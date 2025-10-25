# Templated from https://github.com/pengzhangzhi/faplm/blob/main/faesm/esm.py under MIT license

# Adopted from DPLM for the SDPA attention
# https://github.com/bytedance/dplm/blob/main/src/byprot/models/lm/dplm.py
# which is under license Apache-2.0.

flash_attn_installed = True
try:
    from faesm.fa_utils import RotaryEmbedding as FAEsmRotaryEmbedding
    from faesm.fa_utils import unpad
    from flash_attn import flash_attn_varlen_qkvpacked_func
except ImportError:
    flash_attn_installed = False
    print(
        """
          [Warning] Flash Attention not installed.
          By default we will use Pytorch SDPA attention,
          which is slower than Flash Attention but better than official ESM.
    """
    )
import logging
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from faesm.esm import FAEsmLayer
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers.models.esm.modeling_esm import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    EsmContactPredictionHead,
    EsmEmbeddings,
    EsmEncoder,
    EsmForMaskedLM,
    EsmLMHead,
    EsmModel,
    EsmPooler,
    EsmPreTrainedModel,
)

logger = logging.getLogger(__name__)


class TimestepEmbedderNew(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=1280):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = (
            2
            * 3.14159
            * torch.exp(
                -math.log(max_period)
                * (torch.arange(start=0, end=half, dtype=torch.float32) - half / 3)
                / half
            ).to(device=t.device)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FAEsmEncoder(EsmEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.layer = nn.ModuleList(
            [FAEsmLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.emb_layer_norm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.gradient_checkpointing = False

        # NEW: Add Film layers after each transformer layer
        self.use_film = getattr(config, "use_film", True)
        if self.use_film:
            # Create film layers for each transformer layer
            self.film_layers = nn.ModuleList(
                [
                    nn.Linear(config.conditioning_dim, 2 * config.hidden_size)
                    for _ in range(config.num_hidden_layers)
                ]
            )
            # Initialize with zero weights and biases
            for layer in self.film_layers:
                layer.weight.data.zero_()
                layer.bias.data.zero_()
            # Timestep or schedule embedder
            self.conditioning_embedder = TimestepEmbedderNew(config.conditioning_dim)
            # Zero initialize the final layer to start with no conditioning
            self.conditioning_embedder.mlp[2].weight.data.zero_()
            self.conditioning_embedder.mlp[2].bias.data.zero_()

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        conditioning=None,
        output_pad_fn=None,
        unpad_function=None,
    ):
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        # New lines for embedding the conditioning
        if self.use_film and conditioning is not None:
            # Get embedding of conditioning (S or t)
            if conditioning.dim() == 1:  # (batch_size,)
                # Expand to match sequence length if needed
                batch_size = (
                    hidden_states.size(0)
                    if cu_seqlens is None
                    else cu_seqlens.size(0) - 1
                )
                conditioning_emb = F.silu(self.conditioning_embedder(conditioning))
                conditioning_emb = conditioning_emb.unsqueeze(
                    1
                )  # (batch_size, 1, conditioning_dim)
            else:
                # Already has sequence dimension
                conditioning_emb = F.silu(
                    self.conditioning_embedder(conditioning.reshape(-1))
                ).reshape(conditioning.shape + (-1,))

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states=hidden_states,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states=hidden_states,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            # New lines to apply FiLM modulation after layer processing
            if self.use_film and conditioning is not None:
                film_params = self.film_layers[i](conditioning_emb)
                # Split into scale and shift
                scale, shift = film_params.chunk(2, dim=-1)

                # For Flash Attention compatibility
                if output_pad_fn is not None and cu_seqlens is not None:
                    # First, convert unpadded hidden states to padded format
                    padded_hidden = output_pad_fn(hidden_states)

                    # Apply FiLM in padded space
                    padded_hidden = padded_hidden * (1 + scale) + shift

                    # Now call unpad with correct format
                    hidden_states, cu_seqlens, max_seqlen, _, output_pad_fn = (
                        unpad_function(padded_hidden)
                    )
                else:
                    # Standard application when Flash Attention is disabled
                    hidden_states = hidden_states * (1 + scale) + shift

            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class FAEsmModel(EsmModel):
    def __init__(self, config, add_pooling_layer=True):
        EsmPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = EsmEmbeddings(config)
        self.encoder = FAEsmEncoder(config)

        self.pooler = EsmPooler(config) if add_pooling_layer else None

        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        conditioning: Optional[torch.Tensor] = None,  # NEW parameter
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = encoder_attention_mask

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        self.config.use_fa &= flash_attn_installed

        if self.config.use_fa:
            embedding_output, cu_seqlens, max_seqlen, _, output_pad_fn = unpad(
                embedding_output, attention_mask
            )
        else:
            cu_seqlens = None
            max_seqlen = None
            output_pad_fn = lambda x: x

        encoder_outputs = self.encoder(
            embedding_output,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            conditioning=conditioning,  # NEW: Pass conditioning to encoder
            output_pad_fn=output_pad_fn if self.config.use_fa else None,
            unpad_function=lambda h: unpad(h, attention_mask),
        )
        sequence_output = encoder_outputs[0]

        sequence_output = output_pad_fn(sequence_output)

        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class FAEsmForMaskedLM(EsmForMaskedLM):
    def __init__(self, config, dropout=0.1):
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        config.hidden_dropout_prob = dropout

        # NEW: Make sure use_film is set in config
        config.use_film = getattr(config, "use_film", True)
        config.conditioning_dim = getattr(config, "conditioning_dim", 128)

        EsmPreTrainedModel.__init__(self, config)
        self.esm = FAEsmModel(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)

        self.init_weights()

        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.x_id = tokenizer._token_to_id["X"]

        self.contact_head = None
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        conditioning=None,  # New parameter for S or t
    ):
        attention_mask = input_ids.ne(self.pad_id)
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=output_hidden_states,  # For the hidden states
            conditioning=conditioning,  # NEW: Pass conditioning to model
        )
        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)

        if outputs.hidden_states is not None:
            result = {
                "logits": logits,
                "last_hidden_state": sequence_output,
                "hidden_states": [x.unsqueeze(0) for x in outputs.hidden_states],
            }
        else:
            result = {"logits": logits, "last_hidden_state": sequence_output}
        return result

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        use_fa=True,
        use_film=True,
        conditioning_dim=128,
        load_pretrained_weights=True,
        *model_args,
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        config.use_fa = use_fa
        config.use_film = use_film
        config.conditioning_dim = conditioning_dim

        model = cls(config, *model_args, **kwargs)
        if load_pretrained_weights:
            ptr_model = AutoModelForMaskedLM.from_pretrained(
                pretrained_model_name_or_path
            )
            # NEW: Set strict=False to allow loading weights into a model with additional parameters
            model.load_state_dict(ptr_model.state_dict(), strict=False)
        return model

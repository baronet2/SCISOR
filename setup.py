from setuptools import find_packages, setup

setup(
    name="SCISOR",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "datasets",
        "huggingface_hub",
        "lightning",
        "einops",
        "hydra-core",
        "omegaconf",
        "wandb",
        "transformers",
        "faiss-cpu",
        "pytorch-lightning",
        "evodiff",
    ],
    python_requires=">=3.8.5",
)

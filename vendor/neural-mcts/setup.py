"""
Neural-Guided MCTS for Pokémon Battling
Hybrid model combining Actor-Critic RL priors with MCTS search
"""

from setuptools import setup, find_packages

setup(
    name="neural-mcts",
    version="0.1.0",
    description="Neural-Guided MCTS for Pokémon Battling",
    author="PokeAgent Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "poke-engine>=0.0.46",
        "flask>=3.0.0",
        "requests>=2.32.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "ruff>=0.1.0",
        ],
    },
)

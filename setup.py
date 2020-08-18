import os
from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="metis",
    version="0.1.1",
    author="Frank Odom",
    author_email="frank.odom.iii@gmail.com",
    description="Minimalist library for training RL agents in PyTorch",
    license="GPL-3.0",
    long_description=long_description,
    packages=find_packages(),
    entry_points={"console_scripts": ["metis=metis.cli:main"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
    ]
)

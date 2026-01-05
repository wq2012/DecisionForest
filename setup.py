from setuptools import setup, find_packages

# Read the contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pydecisionforest",
    version="0.1.0",
    description="Python implementation of Decision Tree, Decision Forest, and AdaBoost",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Quan Wang",
    author_email="wangq10@rpi.edu",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    install_requires=[
        "numpy>=1.18.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent

setup(
    name="kdpeak",
    version="0.1.1",
    author="Dominik Otto",
    author_email="dotto@fredhutch.org",
    description="A tool to identify genomic peaks based on kernel density estimation.",
    long_description=(this_directory / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/settylab/kdpeak",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "KDEpy>=1.0.6",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "kdpeak=kdpeak.core:main",
        ],
    },
)

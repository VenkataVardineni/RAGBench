"""Setup script for RAGBench."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="ragbench",
    version="0.1.0",
    description="Evaluation harness for Retrieval-Augmented Generation systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="RAGBench Contributors",
    license="MIT",
    url="https://github.com/VenkataVardineni/RAGBench",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "faiss": [
            "faiss-cpu>=1.7.4; sys_platform != 'darwin'",
            "faiss>=1.7.4; sys_platform == 'darwin'",
        ],
        "spacy": [
            "spacy>=3.5.0",
        ],
        "all": [
            "faiss-cpu>=1.7.4; sys_platform != 'darwin'",
            "faiss>=1.7.4; sys_platform == 'darwin'",
            "spacy>=3.5.0",
            "sentence-transformers>=2.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ragbench-eval=ragbench.eval.run_eval:main",
            "ragbench-report=ragbench.eval.report:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)


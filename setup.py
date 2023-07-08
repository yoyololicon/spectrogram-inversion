
import setuptools
from torch_specinv import __version__, __email__, name, __maintainer__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=name,
    version=__version__,
    author=__maintainer__,
    author_email=__email__,
    description="A pytorch package for Spectrogram Inversion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyololicon/spectrogram-inversion",
    packages=["torch_specinv"],
    install_requires=['torch>=1.6.0', 'tqdm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

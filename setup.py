import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quantagrad",
    version="0.1.0",
    author="Oluwaseun Ale-Alaba",
    author_email="carmichael8821@gmail.com",
    description="An autograd engine with a PyTorch-like neural network library on top.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tony-Ale/quantagrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    
    install_requires=[
        "numpy>=1.26.4",
    ],
)
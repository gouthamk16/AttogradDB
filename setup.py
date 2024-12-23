from setuptools import setup, find_packages

setup(
    name="attogradDB",  # Name of your package
    version="0.2",  # Initial version
    description="A simple vector database for fast similarity search",  # Short description
    long_description=open("README.md").read(),  # Use README.md for long description
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vector-db",  # Your project URL (GitHub, etc.)
    author="Goutham Krishnan",
    license="MIT",  # License type
    packages=find_packages(),  # Automatically find packages in the folder
    install_requires=[
        "numpy",
        "scipy",
        "tiktoken",
        "transformers",
        "accelerate",
        "huggingface_hub",  # Example dependencies
        "PyPDF2",
        "graphing",
        "hnswlib"  
    ],
    classifiers=[
        "Programming Language :: Python :: 3.12.3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',  # Python version requirement
)

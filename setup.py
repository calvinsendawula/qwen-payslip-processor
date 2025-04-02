from setuptools import setup, find_packages

setup(
    name="qwen-payslip-processor",
    version="0.1.4",
    author="Calvin Sendawula",
    author_email="calvinsendawula188@gmail.com",
    description="German payslip processor using Qwen2.5-VL-7B model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/calvinsendawula/qwen-payslip-processor",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11.5",  # Specify Python 3.11.5 as the minimum version
    install_requires=[
        "torch>=2.0.0,<3.0.0",
        "transformers>=4.30.0,<5.0.0",
        "PyMuPDF>=1.22.0,<2.0.0",
        "pillow>=9.3.0,<12.0.0",
        "numpy>=1.24.0,<2.0.0",
        "pyyaml>=6.0,<7.0.0",
        "matplotlib>=3.5.0,<4.0.0",
        "requests>=2.32.0,<3.0.0",  # For API communication
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)

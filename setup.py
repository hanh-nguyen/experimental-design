from setuptools import setup, find_packages

setup(
    name="experimental-design",
    version="0.0.1",
    description="Tests for experimental design",
    author="Hanh Nguyen",
    author_email="myhanh.nguyen1211@gmail.com",
    packages=find_packages("dev"),
    package_dir={"": "dev"},
    install_requires=["pandas", "numpy", "scipy", "pytest"],
)

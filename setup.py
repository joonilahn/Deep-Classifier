#!/usr/bin/env python
from setuptools import find_packages, setup


setup(
    name="deep_classifier",
    version="0.1",
    author="Joonil Ahn",
    description="Deep Learning Based Image Classifier",
    keywords="image classification, deep learning",
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "matplotlib",
        "pandas",
        "seaborn",
        "yacs",
        "efficientnet_pytorch",
        "Pillow>=5.4.1",
        "tensorboardX==2.0",
    ],
    dependency_links=[
        "http://42.1.154.162:8081/repository/pypi-repo/simple",
        "http://42.1.154.162:8081/repository/pypi-repo",
    ],
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
    ],
    include_package_data=True,
    zip_safe=False,
    license="MIT",
)

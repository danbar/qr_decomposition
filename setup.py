from setuptools import setup, find_packages

import qr_decomposition

setup(
    name="qr_decomposition",
    version=qr_decomposition.__version__,
    url="https://github.com/danbar/qr_decomposition/",
    license="MIT License",
    author="Daniel Bartel",
    install_requires=["numpy>=1.9"],
    description="QR decomposition package for Python",
    packages=find_packages(),
    test_suite="nose.collector"
)

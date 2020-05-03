from setuptools import setup

setup(
    packages=["sharepred"],
    entry_points={
        "console_scripts": ["sharepred = sharepred.__main__:main"]},
    install_requires=[],
)

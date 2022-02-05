from setuptools import setup

setup(
    name="dylightful",
    version="0.1.2",
    description="A package to get the most out of Dynophores",
    url="https://github.com/MQSchleich/dylightful",
    author="Julian M. Kleber",
    author_email="julian.m.kleber@gmail.com",
    license="MIT",
    packages=["app"],
    install_requires=["numpy", "pandas", "matplotlib", "tqdm", "hmmlearn"],
    zip_safe=False,
)

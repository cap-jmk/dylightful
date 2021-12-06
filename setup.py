from setuptools import setup

setup(
    name="dylightful",
    version="0.1",
    description="A package supporting rapid python development with GitHub CI",
    url="https://github.com/MQSchleich/dylightful",
    author="Peter Parker",
    author_email="julian.m.kleber@gmail.com",
    license="MIT",
    packages=["app", "dylightful"],
    install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'tqdm',
          'hmmlearn'
      ],
    zip_safe=False,
)

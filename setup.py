import setuptools


def readme():
    with open('README.md', 'r') as f:
        return f.read()


def requirements():
    with open('requirements.txt', 'r') as f:
        return [line.rstrip() for line in f]


with open("README.md", "r") as fh:
    long_description = fh.read()
    setuptools.setup(
        name='ann',
        author="Timo Nicolai",
        description="KTH ANN lab solutions",
        long_description=readme(),
        packages=setuptools.find_packages(),
        install_requires=requirements())

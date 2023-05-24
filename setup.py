from setuptools import find_packages
from setuptools import setup

from dal_toolbox import __version__


def requirements():
    with open("requirements.txt", "r") as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


setup(
    name='dal_toolbox',
    version=__version__,
    description='Deep Active Learning Toolbox.',
    author='Denis Huseljic',
    author_email='dhuselijc@uni-kassel.de',
    url='https://git.ies.uni-kassel.de/dhuseljic/uncertainty-evaluation',
    license='BSD 3-Clause',
    packages=find_packages(),
    install_requires=requirements(),
    extras_require={
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD 3-Clause License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='deep active learning machine learning probabilistic modeling',
    python_requires=">=3.9",
)

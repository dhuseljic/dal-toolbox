import os
import sys

from setuptools import find_packages
from setuptools import setup

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'dal_toolbox')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

setup(
    name='dal_toolbox',
    version=__version__,
    description='Deep Active Learning Toolbox.',
    author='Denis Huseljic',
    author_email='dhuselijc@uni-kassel.de',
    url='https://git.ies.uni-kassel.de/dhuseljic/uncertainty-evaluation',
    license='BSD 3-Clause',
    packages=find_packages(),
    install_requires=[ ],
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
)

import sys
from distutils.core import setup


setup(
    name='sbiambit',
    version='2.0',
    author='Dan Leonte',
    description='SBI for Ambit stochastics',
    install_requires=['corner', 'matplotlib', 'numpy', 'pandas', 'seaborn', 'tqdm']
)

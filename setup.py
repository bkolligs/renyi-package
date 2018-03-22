from setuptools import setup

setup(name='renyi',
      version=0.1,
      description='RMI package',
      url='https://github.com/bkolligs/renyi-package',
      author='Ben Kolligs',
      packages=['renyi'],
      install_requires= ['time', 'numpy', 'datetime', 'mpmath', 'pylab', 'random'],
      zip_safe = False
)
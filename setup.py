from setuptools import setup

setup(
    name='beholder',
    version='0.0.1',
    packages=['beholder'],
    package_data={'beholder': ['resources/*']},
    url='https://github.com/chrisranderson/beholder',
    author='Chris Anderson',
    install_requires=[
      'futures'
    ],
)

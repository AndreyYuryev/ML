from setuptools import setup
import setuptools
from os import path


def readme():
    with open(path.join(path.abspath((path.dirname(__file__))), "README.rst"), "r", encoding="utf-8") as file:
        return file.read()


setup(name='ML',
      author='Andrey Yuryev',
      version='0.1',
      description='ML',
      long_description=readme(),
      long_description_content_type="text/markdown",
      packages=setuptools.find_packages(),
      package_dir={'ml': 'ml'},
      install_requires=['numpy'],
      # install_requires=['numpy', 'selenium', 'openpyxl'],
      include_package_data=True,
      entry_points={
          'console_scripts': ['ml = ml.__main__:main']
      }
      # packages=['qs_check']
      )

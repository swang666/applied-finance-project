"""setup file for the FlaskServer package
"""
import unittest
from setuptools import setup, find_packages
from distutils.command.build_py import build_py


def readme():
    """returns the text of the readme.md file
    """
    with open('README.md') as readme_file:
        return readme_file.read()


def package_test_suite():
    """load the test suit
    """
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('./tests', pattern='test_*.py')
    return test_suite


class SetupNLTK(build_py):
    """This class downloads nltk resources
    """

    # The resources we need for nltk
    RESOURCES = [
        "punkt",
        "stopwords",
        "wordnet"
    ]

    def run(self):
        """This function executes the command
        """
        import nltk
        for resource in self.RESOURCES:
            nltk.download(resource)


setup(name="ucla-topic-analysis",
      version="0.1.0",
      description=("This package is used for finding topics of interest"
                   " financial statements"),
      cmdclass={"nltk": SetupNLTK},
      long_description=readme(),
      url="https://github.com/Ark-Paradigm/UCLA-Topic-Analysis",
      author="Ark Paradigm",
      author_email="founders@arkparadigm.com",
      license=None,
      packages=find_packages(),
      install_requires=[],
      dependency_links=[],
      include_package_data=True,
      test_suite="setup.package_test_suite")

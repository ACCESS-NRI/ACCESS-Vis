import versioneer
from setuptools import setup

setup(
    name="accessvis",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)

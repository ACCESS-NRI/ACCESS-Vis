from setuptools import setup
import versioneer

setup(
    name="accessvis",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)

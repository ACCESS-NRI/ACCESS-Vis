import os
import sys

from . import _version
from .earth import *
from .utils import *

# Add current directory to sys.path because python is deranged
sys.path.append(os.path.dirname(__file__))

__version__ = _version.get_versions()["version"]

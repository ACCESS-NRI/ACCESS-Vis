import sys
import os
#Add current directory to sys.path because python is deranged
sys.path.append(os.path.dirname(__file__))

from .earth import *
from .utils import *
 


from . import _version
__version__ = _version.get_versions()['version']

import numpy
import numpy as np
from error import *
from cube import *

__author__ = "Niels Bassler and Jakob Toftegaard"
__version__ = "1.0"
__email__ = "bassler@phys.au.dk"

class LETCube(Cube):
        def __init__(self,cube = None):
                super(LETCube,self).__init__(cube)

#!/usr/bin/env python

"""npmatrix.py: numpy (sparse) implementation of matrix"""

__author__      = "Tim Van de Cruys"
__email__       = "timvdc@gmail.com"
__status__      = "development"

from util import *
from similarityfunctions import *
import math
import numpy, scipy
import scipy.sparse
import semsimlib

class NPMatrix:
    def __init__(self, matrix):
        assert issubclass(matrix.__class__, semsimlib.Matrix)
        self.instances = matrix.instances
        self.instanceDict = matrix.instanceDict
        self.features = matrix.features
        self.featureDict = matrix.featureDict

        indrow = []
        indcol = []
        val = []
        for i in range(len(matrix.vectorList)):
            lengthi = len(matrix.vectorList[i])
            indrow.extend([i for j in range(lengthi)])
            indcol.extend(matrix.vectorList[i].keys())
            val.extend(matrix.vectorList[i].values())

        self.matrix = scipy.sparse.csr_matrix(
            (numpy.array(val),
             (numpy.array(indrow),
              numpy.array(indcol))
             ),
            shape=(len(self.instances),
                   len(self.features))
            )

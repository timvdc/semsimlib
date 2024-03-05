#!/usr/bin/env python

"""
svdmatrix.py: matrix for singular value decomposition using scipy's
svds function

"""

__author__      = "Tim Van de Cruys"
__email__       = "timvdc@gmail.com"
__status__      = "development"

import math
import numpy
import scipy
import scipy.sparse
import scipy.sparse.linalg
import semsimlib
import warnings

class SVDMatrix:
    def __init__(self, matrix, rdim):
        assert issubclass(matrix.__class__, semsimlib.NPMatrix)
        self.instances = matrix.instances
        self.instanceDict = matrix.instanceDict
        self.features = matrix.features
        self.featureDict = matrix.featureDict

        self.matrix = matrix.matrix
        #Normalize matrix to 1
        #self.matrix.data = self.matrix.data / numpy.sum(self.matrix.data)

        self.ndim = len(self.instances)
        self.vdim = len(self.features)
        self.rdim = rdim
        
        self.U = []
        self.S = []
        self.Vt = []

        #self.Wnorm = []

    def compute(self):
        result = scipy.sparse.linalg.svds(self.matrix, self.rdim)
        self.U, self.S, self.Vt = result
        return None

    def getTopWordsDim(self,ndim,nwords=20):
        #show list of words with highest value for particular
        #dimension
        if hasattr(self, 'Wnorm'):
            raise ValueError('make sure to normalize NMF result')
        dimList = [(self.Wnorm[i,ndim],i) for i in range(len(self.instances))]
        dimList.sort()
        dimList.reverse()
        for i in range(nwords):
            print(self.instances[dimList[i][1]], dimList[i][0])

    def normalize(self):
        self.U = numpy.fliplr(self.U)
        self.S = numpy.fliplr([self.S])[0]
        self.Vt = numpy.fliplr(self.Vt)
        Wnorm = numpy.zeros((self.ndim, self.rdim))
        for i in range(self.ndim):
            Wnorm[i] = self.U[i] / numpy.linalg.norm(self.U[i])
        self.Wnorm = Wnorm
        return None

    def calculateMostSimilar(self, instance, topN = 20):
        if self.Wnorm == []:
            raise ValueError('make sure to normalize NMF result')
        resultVector = numpy.dot(self.Wnorm[self.instanceDict[instance]],self.Wnorm.T)
        sortedCosineList = [ [resultVector[i],i] for i in range(len(resultVector)) ]
        sortedCosineList.sort()
        sortedCosineList.reverse()
        outputList = []
        for i in range(topN):
            outputList.append([self.instances[sortedCosineList[i][1]], sortedCosineList[i][0]])
        return outputList


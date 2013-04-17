#!/usr/bin/env python

"""
nmfmatrix.py: implementation of non-negative matrix
factorization
"""

__author__      = "Tim Van de Cruys"
__email__       = "timvdc@gmail.com"
__status__      = "development"

from util import *
from similarityfunctions import *
import math
import numpy
import scipy
import scipy.sparse
import semsimlib

class NMFMatrix:
    def __init__(self, matrix, rdim):
        assert issubclass(matrix.__class__, semsimlib.NPMatrix)
        self.instances = matrix.instances
        self.instanceDict = matrix.instanceDict
        self.features = matrix.features
        self.featureDict = matrix.featureDict

        self.matrix = matrix.matrix
        #Normalize matrix to 1
        self.matrix.data = self.matrix.data / numpy.sum(self.matrix.data)

        self.ndim = len(self.instances)
        self.vdim = len(self.features)
        self.rdim = rdim
        
        # Initialize matrix using absolute values from gaussian
        # distribution (mean=0,std=1), shifted by 0.01 - this ought to
        # give many small values and a limited number of large values
        self.W = numpy.absolute(
            numpy.random.randn(self.ndim, self.rdim)
            ) + 0.01
        # Normalize: columns sum to 1
        self.W /= self.W.sum(axis=0)[numpy.newaxis, :]
        
        #self.H = numpy.absolute(
        #    numpy.random.randn(self.rdim, self.vdim)
        #    ) + 0.01

        # Construct H from random W and original
        self.H = self.W.transpose() * self.matrix.tocsr()
        # Normalize: columns sum to column-sum of original matrix
        #self.H /= self.H.sum(axis=0) / self.matrix.sum(axis=0)

    def compute(self, niters=25):
        self.divs = []
        for niter in range(niters):
            print 'iteration ' + str(niter + 1)
            reconstruct = numpy.zeros(len(self.matrix.data))
            for i in range(len(self.matrix.row)):
                reconstruct[i] = numpy.dot(self.W[self.matrix.row[i],:],
                                           self.H[:,self.matrix.col[i]])

            Q = self.matrix.data / reconstruct
            mQ = scipy.sparse.coo_matrix((Q,
                                          (self.matrix.row,self.matrix.col)),
                                         shape=(self.ndim,self.vdim))
            self.H = self.H * (self.W.T * mQ)
            #self.H /= self.H.sum(axis=0) / self.matrix.sum(axis=0)
            #self.H /= self.H.sum(axis=0)[numpy.newaxis, :]

            self.W = numpy.maximum(self.W * (mQ * self.H.T),1e-20)
            self.W /= self.W.sum(axis=0)[numpy.newaxis, :]
            div = numpy.sum(self.matrix.data * numpy.log(Q)
                            - self.matrix.data
                            + reconstruct)
            print 'div: ' + str(div)
            self.divs.append(div)
        return reconstruct, mQ

    def get_top_words_dim(self,ndim):
        dimList = [(self.W[i,ndim],i) for i in range(len(self.instances))]
        dimList.sort()
        dimList.reverse()
        for i in range(20):
            print self.instances[dimList[i][1]], dimList[i][0]

#    def normalize():
#    Note: row-sum of H contain p(z)
#    Normalization would normalize rows to 1 and put
#    sums of original H-rows in vector pz

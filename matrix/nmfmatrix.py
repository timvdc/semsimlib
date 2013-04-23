#!/usr/bin/env python

import nmf

"""
nmfmatrix.py: implementation of non-negative matrix
factorization for sparse matrix (kl-divergence)
TODO: sparseness constraints
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
        # keep most values close to zero with a limited number of
        # large values
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

        self.divs = []

    def compute(self, niters=25):
        for niter in range(niters):
            print 'iteration ' + str(niter + 1)

            #reconstructed = self.__reconstruct()
            reconstructed = nmf.reconstruct(self.matrix.row, self.matrix.col, self.matrix.data, self.W, self.H)
            Q = scipy.sparse.coo_matrix((self.matrix.data / reconstructed,
                                          (self.matrix.row,self.matrix.col)),
                                         shape=(self.ndim,self.vdim))
            self.H = self.H * (self.W.T * Q)
            #self.H /= self.H.sum(axis=0) / self.matrix.sum(axis=0)
            #self.H /= self.H.sum(axis=0)[numpy.newaxis, :]

            self.W = numpy.maximum(self.W * (Q * self.H.T),1e-20)
            self.W /= self.W.sum(axis=0)[numpy.newaxis, :]
            div = numpy.sum(self.matrix.data * numpy.log(Q.data)
                            - self.matrix.data
                            + reconstructed)
            print 'div: ' + str(div)
            self.divs.append(div)
        return None

    def __reconstruct(self):
        reconstruct = numpy.zeros(len(self.matrix.data))
        for i in range(len(self.matrix.row)):
           reconstruct[i] = numpy.dot(self.W[self.matrix.row[i],:],
                                      self.H[:,self.matrix.col[i]])
        return reconstruct
    
    def get_top_words_dim(self,ndim,nwords=20):
        #show list of words with highest value for particular
        #dimension
        dimList = [(self.W[i,ndim],i) for i in range(len(self.instances))]
        dimList.sort()
        dimList.reverse()
        for i in range(nwords):
            print self.instances[dimList[i][1]], dimList[i][0]

    def normalize(self):
        #original row-sum of H contain p(z); normalization normalizes
        #rows of H to 1 and puts sum of original H-rows in vector pz
        self.pz = numpy.sum(self.H,axis=1)
        self.H /= self.H.sum(axis=1)[:,numpy.newaxis]

        #sorting in descending order
        sortindices = self.pz.argsort()[::-1]
        self.pz = self.pz[sortindices]
        self.W = self.W[:,sortindices]
        self.H = self.H[sortindices]
        return None

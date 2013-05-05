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
        if issubclass(matrix.__class__, semsimlib.Matrix):
            self.instances = matrix.instances
            self.instanceDict = matrix.instanceDict
            self.features = matrix.features
            self.featureDict = matrix.featureDict

            # indrow = []
            # indcol = []
            # val = []
            # for i in range(len(matrix.vectorList)):
            #     lengthi = len(matrix.vectorList[i])
            #     indrow.extend([i for j in range(lengthi)])
            #     indcol.extend(matrix.vectorList[i].keys())
            #     val.extend(matrix.vectorList[i].values())

            # self.matrix = scipy.sparse.csr_matrix(
            #     (numpy.array(val),
            #      (numpy.array(indrow),
            #       numpy.array(indcol))
            #      ),
            #     shape=(len(self.instances),
            #            len(self.features))
            #     )

            indrow = []
            indcol = []
            data = []
            for i in range(len(matrix.vectorList)):
                for j in matrix.vectorList[i].keys():
                    indrow.append(i)
                    indcol.append(j)
                    data.append(matrix.vectorList[i][j])
            self.matrix = scipy.sparse.coo_matrix(
                (numpy.array(data),
                 (numpy.array(indrow),
                  numpy.array(indcol))
                 ),
                shape = (len(self.instances),
                         len(self.features))
                )

        elif isinstance(matrix, tuple) and len(matrix) == 5:
            indrow,indcol,data,ndim,vdim = matrix
            self.matrix = scipy.sparse.coo_matrix(
                (numpy.array(data),
                 (numpy.array(indrow),
                  numpy.array(indcol))
                 ),
                shape = (ndim,vdim)
                )
            self.instances = [i for i in range(ndim)]
            self.instanceDict = dict.fromkeys(self.instances)
            self.features = [i for i in range(vdim)]
            self.featureDict = dict.fromkeys(self.features)
        
        elif isinstance(matrix, tuple) and len(matrix) == 3:
            dataFile,instanceFile,featureFile = matrix
            self.instances = readFileAsList(instanceFile)
            self.instanceDict = createDictFromList(self.instances)
            self.features = readFileAsList(featureFile)
            self.featureDict = createDictFromList(self.features)

            indrow = []
            indcol = []
            data = []
            for line in file(dataFile):
                line = line.rstrip()
                i,j,k = line.split('\t')
                indrow.append(int(i) - 1)
                indcol.append(int(j) - 1)
                data.append(float(k))

            self.matrix = scipy.sparse.coo_matrix(
                (numpy.array(data),
                 (numpy.array(indrow),
                  numpy.array(indcol))
                 ),
                shape = (len(self.instances), len(self.features))
                )

#    this is where we'll implement the same matrix functions
#    using scipy.sparse
#    def..
    def calculateMostSimilar(self, instance, topN = 20, similarity = 'cosine'):
        if not similarity == 'cosine':
            raise ValueError('similarity function unknown')
        self.matrix = self.matrix.tocsr()
        nInstance = self.instanceDict[instance]
        cosineVector = self.matrix[nInstance].dot(self.matrix.T).todense()
        cosineList = [(cosineVector[0,i],i) for i in range(cosineVector.shape[1])]
        cosineList.sort()
        cosineList.reverse()
        outputList = []
        for i in range(topN):
            outputList.append([self.instances[cosineList[i][1]], cosineList[i][0]])
        return outputList

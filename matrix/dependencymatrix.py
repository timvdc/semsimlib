#!/usr/bin/env python

"""dependencymatrix.py: matrix class for dependency-based context"""

__author__ = "Tim Van de Cruys"
__email__ = "timvdc@gmail.com"
__status__ = "development"

import semsimlib
from matrix import *
from semsimlib.corpusreader import *

class DependencyMatrix(Matrix):
    def __init__(self, instances, features):
        Matrix.__init__(self, instances, features)

    def fill(self, filenames, CorpusReader=DependencyCorpusReader, cleanup=True):
        dependencyStream = CorpusReader(filenames)
        for tripleList in dependencyStream:
            for freq,instance,feature in tripleList:
                if (self.instanceDict.has_key(instance) and
                    self.featureDict.has_key(feature)):
                    nInstance = self.instanceDict[instance]
                    nFeature = self.featureDict[feature]
                    self[nInstance, nFeature] += freq
        if cleanup:
            self.cleanup()
        return

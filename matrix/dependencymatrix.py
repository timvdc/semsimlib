#!/usr/bin/env python

"""dependencymatrix.py: matrix class for dependency-based context"""

__author__ = "Tim Van de Cruys"
__email__ = "timvdc@gmail.com"
__status__ = "development"

import semsimlib
from matrix import *
from semsimlib.corpusreader import *

class DependencyMatrix(Matrix):
    def __init__(self, filenames, instanceFile = None,
                 featureFile = None, instanceCutoff = 20,
                 featureCutoff = 2, valueCutoff = 3,
                 CorpusReader = DependencyCorpusReader,
                 dryRun = True, cleanup = True):
        Matrix.__init__(self, filenames, instanceFile, featureFile,
                        instanceCutoff, featureCutoff, valueCutoff)

        if dryRun:
            dependencyStream = CorpusReader(self.filenames)
            self.dryRun(dependencyStream)

        dependencyStream = CorpusReader(self.filenames)
        self.fill(dependencyStream)

        if cleanup:
            self.applyValueCutoff()
            self.applyInstanceFeatureCutoff()

    def dryRun(self, stream):
        instanceCount = {}
        featureCount = {}
        for tripleList in stream:
            for freq,instance,feature in tripleList:
#                if freq >= self.valueCutoff:
                try:
                    instanceCount[instance] += freq
                except KeyError:
                    instanceCount[instance] = freq
                try:
                    featureCount[feature] += freq
                except KeyError:
                    featureCount[feature] = freq

        instances = [i for i in instanceCount if instanceCount[i] >= self.instanceCutoff]
        features = [i for i in featureCount if featureCount[i] >= self.featureCutoff]
        if self.instances:
            instances = [i for i in instances if self.instanceDict.has_key(i)]
        if self.features:
            features = [i for i in features if self.features.has_key(i)]
        self.instances = instances
        self.features = features
        self.instanceDict = createDictFromList(self.instances)
        self.featureDict = createDictFromList(self.features)
        self.vectorList = [{} for i in range(len(self.instances))]

    def fill(self, stream):
        for tripleList in stream:
            for freq,instance,feature in tripleList:
                if (self.instanceDict.has_key(instance) and
                    self.featureDict.has_key(feature)):
                    nInstance = self.instanceDict[instance]
                    nFeature = self.featureDict[feature]
                    try:
                        self.vectorList[nInstance][nFeature] += freq
                    except KeyError:
                        self.vectorList[nInstance][nFeature] = freq

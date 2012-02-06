#!/usr/bin/env python

"""documentmatrix.py: matrix class for document-based context"""

__author__      = "Tim Van de Cruys"
__email__       = "timvdc@gmail.com"
__status__      = "development"

import semsimlib
from matrix import *
from semsimlib.corpusreader import PlaintextCorpusReader

class DocumentMatrix(Matrix):
    def __init__(self, filenames, instanceFile = None,
                 instanceCutoff = 100,
                 featureCutoff = 2, valueCutoff = 1,
                 CorpusReader = PlaintextCorpusReader,
                 dryRun = True, cleanup = True):
        featureFile = None
        Matrix.__init__(self, filenames, instanceFile,
                        featureFile, instanceCutoff,
                        featureCutoff, valueCutoff)

        if dryRun:
            textStream = CorpusReader(self.filenames)
            self.dryRun(textStream)

        textStream = CorpusReader(self.filenames)
        self.fill(textStream)

        if cleanup:
            self.applyValueCutoff()
            self.applyInstanceFeatureCutoff()

    def dryRun(self, stream):
        
        wordCount = {}
        for wordList in stream:
            for i in range(len(wordList)):
                try:
                    wordCount[wordList[i]] += 1
                except KeyError:
                    wordCount[wordList[i]] = 1

        instances = [ i for i in wordCount if wordCount[i] >= self.instanceCutoff ]
        if self.instances:
            instances = [ i for i in instances if self.instanceDict.has_key(i) ]
        self.instances = instances
        self.instanceDict = createDictFromList(self.instances)
        self.features = []
        self.vectorList = [ {} for i in range(len(self.instances)) ]

    def fill(self, stream):
        docCount = 0
        for wordList in stream:
            self.features.append(docCount)
            nFeature = docCount
            for i in range(len(wordList)):
                if self.instanceDict.has_key(wordList[i]):
                    nInstance = self.instanceDict[wordList[i]]
                    try:
                        self.vectorList[nInstance][nFeature] += 1
                    except KeyError:
                        self.vectorList[nInstance][nFeature] = 1
            docCount += 1

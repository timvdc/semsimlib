#!/usr/bin/env python

"""windowmatrix.py: matrix class for window-based context"""

__author__      = "Tim Van de Cruys"
__email__       = "timvdc@gmail.com"
__status__      = "development"

import semsimlib
from matrix import *
from semsimlib.corpusreader import *

class WindowMatrix(Matrix):
    """Matrix class that extracts a window-based word space model from
    a sentence-based stream input. Use an integer value as
    window-value, or string value 'all' to use the entire sentence
    """
    def __init__(self, filenames, instanceFile = None,
                 featureFile = None, window = 5, instanceCutoff = 20,
                 featureCutoff = 2, valueCutoff = 3,
                 CorpusReader = PlaintextCorpusReader,
                 dryRun = True, cleanup = True):
        Matrix.__init__(self, filenames, instanceFile, featureFile,
                        instanceCutoff, featureCutoff, valueCutoff)
        self.window = window

        if dryRun:
            textStream = CorpusReader(self.filenames)
            self.dryRun(textStream)

        textStream = CorpusReader(self.filenames)
        self.fill(textStream, window)

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
        features = [ i for i in wordCount if wordCount[i] >= self.featureCutoff ]
        if self.instances:
            instances = [ i for i in instances if self.instanceDict.has_key(i) ]
        if self.features:
            features = [ i for i in features if self.features.has_key(i) ]
        self.instances = instances
        self.features = features
        self.instanceDict = createDictFromList(self.instances)
        self.featureDict = createDictFromList(self.features)
        self.vectorList = [ {} for i in range(len(self.instances)) ]

    def fill(self, stream, window):
        for wordList in stream:
            for i in range(len(wordList)):
                if self.instanceDict.has_key(wordList[i]):
                    if window == 'all':
                        contextList = wordList
                    elif type(window) == int:
                        contextList = []
                        if not i == 0:
                            if i <= window:
                                start1 = 0
                            else:
                                start1 = i - window
                            end1 = i
                            contextList.extend(wordList[start1:end1])

                        if not i == len(wordList) - 1:
                            start2 = i + 1
                            if (len(wordList) - 1) <= i + window:
                                end2 = len(wordList)
                            else:
                                end2 = (i + 1) + window
                            contextList.extend(wordList[start2:end2])
                    else:
                        raise ValueError("Window not implemented")
                    contextList = [el for el in contextList if not el == wordList[i]]
                    nInstance = self.instanceDict[wordList[i]]
                    for c in contextList:
                        if self.featureDict.has_key(c):
                            nFeature = self.featureDict[c]
                            try:
                                self.vectorList[nInstance][nFeature] += 1
                            except KeyError:
                                self.vectorList[nInstance][nFeature] = 1

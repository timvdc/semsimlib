#!/usr/bin/env python

"""windowmatrix.py: matrix class for window-based context"""

__author__      = "Tim Van de Cruys"
__email__       = "timvdc@gmail.com"
__status__      = "development"

import semsimlib
from matrix import *
from semsimlib.corpusreader import *

class WindowMatrix(Matrix):
    def __init__(self, instances, features):
        Matrix.__init__(self, instances, features)

    def fill(self, filenames, CorpusReader=PlaintextCorpusReader,
             window, cleanup=True):
        textStream = CorpusReader(filenames)
        for wordList in textStream:
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
                    contextList = [el for el in contextList if not el == wordList[i]]
                    nInstance = self.instanceDict[wordList[i]]
                    for c in contextList:
                        if self.featureDict.has_key(c):
                            nFeature = self.featureDict[c]
                            self[nInstance,nFeature] += 1
        if cleanup:
            self.cleanup()
        return

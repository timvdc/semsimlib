#!/usr/bin/env python

"""windowtensor3.py: 3-way tensor class for window-based context"""

__author__      = "Tim Van de Cruys"
__email__       = "timvdc@gmail.com"
__status__      = "development"

import semsimlib
from ..matrix import *
from .tensor3 import *
from semsimlib.corpusreader import *

class WindowTensor3(Tensor3):

    def __init__(self, filenames, instance1File = None,
                 instance2File = None, instance3File = None,
                 window = 5, instance1Cutoff = 20,
                 instance2Cutoff = 2, instance3Cutoff= 3, valueCutoff = 3,
                 CorpusReader = semsimlib.TMusicCSVCorpusReader,
                 dryRun = True, cleanup = True):
        Tensor3.__init__(self, filenames, instance1File, instance2File,
                 instance3File, instance1Cutoff, instance2Cutoff,
                 instance3Cutoff, valueCutoff)
        self.window = window

        if dryRun:
            textStream = CorpusReader(self.filenames)
            self.dryRun(textStream)

        textStream = CorpusReader(self.filenames)
        self.fill(textStream, window)

        if cleanup:
            #self.applyValueCutoff()
            #self.applyInstance123Cutoff()
            pass

    def dryRun(self, stream):
        mode1_count = {}
        wordCount = {}
        for el_mode1, wordList in stream:
            #print(el_mode1,wordList)
            try:
                mode1_count[el_mode1] += 1
            except KeyError:
                mode1_count[el_mode1] = 1
            for i in range(len(wordList)):
                try:
                    wordCount[wordList[i]] += 1
                except KeyError:
                    wordCount[wordList[i]] = 1

        instances1 = [ i for i in mode1_count if mode1_count[i] >= self.instance1Cutoff ]
        instances2 = [ i for i in wordCount if wordCount[i] >= self.instance2Cutoff ]
        instances3 = [ i for i in wordCount if wordCount[i] >= self.instance3Cutoff ]
        if self.instances1:
            instances1 = [ i for i in instances1 if i in self.instance1Dict ]
        if self.instances2:
            instances2 = [ i for i in instances2 if i in self.instance2Dict ]
        if self.instances3:
            instances3 = [ i for i in instances3 if i in self.instance3Dict ]    
        self.instances1 = instances1
        self.instances2 = instances2
        self.instances3 = instances3

        self.instance1Dict = createDictFromList(self.instances1)
        self.instance2Dict = createDictFromList(self.instances2)
        self.instance3Dict = createDictFromList(self.instances3)
        self.indexList = []
        self.valueList = []

    def fill(self, stream, window):
        for el_mode1, wordList in stream:
            if el_mode1 in self.instance1Dict:
                nInstance1 = self.instance1Dict[el_mode1]
                for i in range(len(wordList)):
                    if wordList[i] in self.instance2Dict:
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
                        nInstance2 = self.instance2Dict[wordList[i]]
                        #print('a', wordList[i], contextList)
                        for c in contextList:
                            if c in self.instance3Dict:
                                nInstance3 = self.instance3Dict[c]
                                #print(nInstance1, nInstance2, nInstance3)
                                try:
                                    self.tensorDict[nInstance1][nInstance2][nInstance3] += 1
                                except KeyError:
                                    try:
                                        self.tensorDict[nInstance1][nInstance2][nInstance3] = 1
                                    except KeyError:
                                        try:
                                            self.tensorDict[nInstance1][nInstance2] = {}
                                            self.tensorDict[nInstance1][nInstance2][nInstance3] = 1
                                        except KeyError:
                                            self.tensorDict[nInstance1] = {}
                                            self.tensorDict[nInstance1][nInstance2] = {}
                                            self.tensorDict[nInstance1][nInstance2][nInstance3] = 1
                            #print(self.tensorDict)

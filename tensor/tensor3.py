#!/usr/bin/env python

"""tensor.py: distributional tensor superclass"""

__author__      = "Tim Van de Cruys"
__email__       = "timvdc@gmail.com"
__status__      = "development"

from semsimlib.matrix.util import *
#from .similarityfunctions import *
#import math
#import fileinput

class Tensor3:
    def __init__(self, filenames, instance1File, instance2File,
                 instance3File, instance1Cutoff, instance2Cutoff,
                 instance3Cutoff, valueCutoff):

        self.filenames = filenames

        if isinstance(instance1File,str):
            self.instances1 = readFileAsList(instance1File)
            self.instance1Dict = createDictFromList(self.instances1)
        elif instance1File == None:
            self.instances1 = None

        if isinstance(instance2File,str):
            self.instances2 = readFileAsList(instance2File)
            self.instance2Dict = createDictFromList(self.instances2)
        elif instance2File == None:
            self.instances2 = None

        if isinstance(instance3File,str):
            self.instances3 = readFileAsList(instance3File)
            self.instance3Dict = createDictFromList(self.instances3)
        elif instance3File == None:
            self.instances3 = None

        self.instance1Cutoff = instance1Cutoff
        self.instance2Cutoff = instance2Cutoff
        self.instance3Cutoff = instance3Cutoff
        self.valueCutoff = valueCutoff

        self.tensorDict = {}
        
        self.weighted = None
        self.normalized = None

    def toCoo(self):
        coordList1 = []
        coordList2 = []
        coordList3 = []
        valueList = []
        for i in self.tensorDict.keys():
            for j in self.tensorDict[i].keys():
                for k in self.tensorDict[i][j].keys():
                    coordList1.append(i)
                    coordList2.append(j)
                    coordList3.append(k)
                    valueList.append(self.tensorDict[i][j][k])
        return [coordList1, coordList2, coordList3], valueList

    # def applyValueCutoff(self):
    #     if self.valueCutoff > 1:
    #         print(" - triple check")
    #         newTensorDict = {}
    #         for i in self.tensorDict:
    #             for j in self.tensorDict[i]:
    #                 for k in self.tensorDict[i][j]:
    #                     if self.tensorDict[i][j][k] >= self.valueCutoff:
    #                         try:
    #                             newTensorDict[i][j][k] = self.tensorDict[i][j][k]
    #                         except KeyError:
    #                             try:
    #                                 newTensorDict[i][j] = {}
    #                                 newTensorDict[i][j][k] = self.tensorDict[i][j][k]
    #                             except KeyError:
    #                                 newTensorDict[i] = {}
    #                                 newTensorDict[i][j] = {}
    #                                 newTensorDict[i][j][k] = self.tensorDict[i][j][k]

    #         self.tensorDict = newTensorDict

    # def applyInstance123Cutoff(self):
    #     while True:
    #         instances1Cleaned = self.__cleanInstances1()
    #         instances2Cleaned = self.__cleanInstances2()
    #         instances3Cleaned = self.__cleanInstances3()

            
    #         if instances1Cleaned and instances2Cleaned and instances3Cleaned:
    #             break

    #     self.instance1Dict = createDictFromList(self.instances1)
    #     self.instance2Dict = createDictFromList(self.instances2)
    #     self.instance3Dict = createDictFromList(self.instances3)

    # def self.__cleanInstances1():
    #     print(" - instance1 check")
    #     featuresCleaned = False
    #     removeFeatures = {}
    #     featureCountDict = [float(0) for i in range(len(self.instances1))]
    #     for i in self.tensorDict.keys()
    #             featureCountDict[i] += 1
    #     for f in range(len(featureCountDict)):
    #         if featureCountDict[f] < self.featureCutoff:
    #             removeFeatures[f] = 1
    #     if not removeFeatures == {}:
    #         featureMappingList = [ i for i in range(len(self.instances1)) ]
    #         featureMappingDict = {}
    #         self.instances1 = [ self.instances1[i]
    #                           for i in range(len(self.instances1))
    #                           if not i in removeFeatures ]
    #         featureMappingList = [ featureMappingList[i]
    #                                for i in range(len(featureMappingList))
    #                                if not i in removeFeatures ]

    #         print(" - shrinking matrix")
    #         for i in range(len(featureMappingList)):
    #             featureMappingDict[featureMappingList[i]] = i
    #         newVectorList = [ {} for i in range(len(self.instances))]
    #         newTensorDict = {}
    #         for i in range(len(self.instances)):
    #             for j in self.vectorList[i].keys():
    #                 try:
    #                     newVectorList[i][featureMappingDict[j]] = self.vectorList[i][j]
    #                 except KeyError:
    #                     if not j in removeFeatures:
    #                         print('foutje')
    #         self.vectorList = newVectorList
    #     else:
    #         featuresCleaned = True
    #     return featuresCleaned

#!/usr/bin/env python

"""tensor.py: distributional tensor superclass"""

__author__      = "Tim Van de Cruys"
__email__       = "timvdc@gmail.com"
__status__      = "development"

from semsimlib.matrix.util import *
#from .similarityfunctions import *
import math
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

    def applyValueCutoff(self):
        if self.valueCutoff > 1:
            print(" - triple check")
            newTensorDict = {}
            for i in self.tensorDict:
                for j in self.tensorDict[i]:
                    for k in self.tensorDict[i][j]:
                        if self.tensorDict[i][j][k] >= self.valueCutoff:
                            try:
                                newTensorDict[i][j][k] = self.tensorDict[i][j][k]
                            except KeyError:
                                try:
                                    newTensorDict[i][j] = {}
                                    newTensorDict[i][j][k] = self.tensorDict[i][j][k]
                                except KeyError:
                                    newTensorDict[i] = {}
                                    newTensorDict[i][j] = {}
                                    newTensorDict[i][j][k] = self.tensorDict[i][j][k]

            self.tensorDict = newTensorDict

    # def applyInstance123Cutoff(self):
    #     while True:
    #         instances1Cleaned = self.__cleanInstances1()
    #         #instances2Cleaned = self.__cleanInstances2()
    #         #instances3Cleaned = self.__cleanInstances3()

            
    #         if instances1Cleaned: #and instances2Cleaned and instances3Cleaned:
    #             break

    #     self.instance1Dict = createDictFromList(self.instances1)
    #     #self.instance2Dict = createDictFromList(self.instances2)
    #     #self.instance3Dict = createDictFromList(self.instances3)

    # def __cleanInstances1(self):
    #     print(" - instance1 check")
    #     featuresCleaned = False
    #     removeFeatures = {}
    #     featureCountDict = [float(0) for i in range(len(self.instances1))]
    #     for i in self.tensorDict.keys():
    #         for j in self.tensorDict[i].keys():
    #             for k in self.tensorDict[i][j].keys():
    #                 featureCountDict[i] += 1
    #     for f in range(len(featureCountDict)):
    #         if featureCountDict[f] < self.instance1Cutoff:
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
    #         #newVectorList = [ {} for i in range(len(self.instances))]
    #         newTensorDict = {}
    #         for i in self.tensorDict.keys():
    #             for j in self.tensorDict[i].keys():
    #                 for k in self.tensorDict[i][j].keys():
    #                     if not i in removeFeatures:
    #                         newTensorDict[featureMappingDict[i]][j][k] = self.tensorDict[i][j][k]
    #                 #except KeyError:
    #                 #    if not j in removeFeatures:
    #                 #        print('foutje')
    #         self.tensorDict = newTensorDict
    #     else:
    #         featuresCleaned = True
    #     return featuresCleaned

# Local weighting functions

    # Logarithmic weighting
    def calculateLogWeighting(self):
        for i in self.tensorDict.keys():
            for j in self.tensorDict[i].keys():
                for k in self.tensorDict[i][j].keys():
                    self.tensorDict[i][j][k] = 1 + math.log(self.tensorDict[i][j][k])


# Global weighting functions

    # Positive Pointwise Mutual Information
    def calculate3PMI(self):
        try:
            self.instance1ProbabilityList
            self.instance2ProbabilityList
            self.instance3ProbabilityList
        except AttributeError:
            self.__calculateMarginalProbabilities()

        tensorDictPMI = {}
        for i in self.tensorDict.keys():
            for j in self.tensorDict[i].keys():
                for k in self.tensorDict[i][j].keys():
                    PMIValue = math.log( ( self.tensorDict[i][j][k] /
                                           self.frequencyTotal ) /
                                         ( self.instance1ProbabilityList[i] *
                                           self.instance2ProbabilityList[j] *
                                           self.instance3ProbabilityList[k] ) )
                    if PMIValue > 0:
                        try:
                            tensorDictPMI[i][j][k] = PMIValue
                        except KeyError:
                            try:
                                tensorDictPMI[i][j] = {}
                                tensorDictPMI[i][j][k] = PMIValue
                            except KeyError:
                                tensorDictPMI[i] = {}
                                tensorDictPMI[i][j] = {}
                                tensorDictPMI[i][j][k] = PMIValue
        self.tensorDict = tensorDictPMI
        #self.applyInstanceFeatureCutoff()

    # def calculate3LMI(self):
    #     try:
    #         self.instanceProbabilityList
    #         self.featureProbabilityList
    #     except AttributeError:
    #         self.__calculateMarginalProbabilities()

    #     vectorListLMI = [ {} for i in range(len(self.instances))]
    #     for i in range(len(self.vectorList)):
    #         for j in self.vectorList[i]:
    #             LMIValue = ( (self.vectorList[i][j] / self.frequencyTotal) *
    #                          math.log( ( self.vectorList[i][j] /
    #                                      self.frequencyTotal ) /
    #                                    ( self.instanceProbabilityList[i] *
    #                                      self.featureProbabilityList[j] ) ) )
    #             if LMIValue > 0:
    #                 vectorListLMI[i][j] = LMIValue
    #     self.vectorList = vectorListLMI
    #     self.applyInstanceFeatureCutoff()

    def __calculateSumFrequencies(self):
        instance1FrequencyList = [float(0) for i in range(len(self.instances1))]
        instance2FrequencyList = [float(0) for i in range(len(self.instances2))]
        instance3FrequencyList = [float(0) for i in range(len(self.instances3))]

        frequencyTotal = float(0)
        for i in self.tensorDict.keys():
            for j in self.tensorDict[i].keys():
                for k in self.tensorDict[i][j].keys():
                    instance1FrequencyList[i] += self.tensorDict[i][j][k]
                    instance2FrequencyList[j] += self.tensorDict[i][j][k]
                    instance3FrequencyList[k] += self.tensorDict[i][j][k]
                    frequencyTotal += self.tensorDict[i][j][k]
        self.instance1FrequencyList = instance1FrequencyList
        self.instance2FrequencyList = instance2FrequencyList
        self.instance3FrequencyList = instance3FrequencyList
        self.frequencyTotal = frequencyTotal       

    def __calculateMarginalProbabilities(self):
        try:
            self.instance1FrequencyList
            self.instance2FrequencyList
            self.instance3FrequencyList

        except AttributeError:
            self.__calculateSumFrequencies()

        instance1ProbabilityList = [ (self.instance1FrequencyList[i] / self.frequencyTotal)
                                    for i in range(len(self.instance1FrequencyList)) ]
        instance2ProbabilityList = [ (self.instance2FrequencyList[i] / self.frequencyTotal)
                                     for i in range(len(self.instance2FrequencyList)) ]
        instance3ProbabilityList = [ (self.instance3FrequencyList[i] / self.frequencyTotal)
                                     for i in range(len(self.instance3FrequencyList)) ]

        self.instance1ProbabilityList = instance1ProbabilityList
        self.instance2ProbabilityList = instance2ProbabilityList
        self.instance3ProbabilityList = instance3ProbabilityList


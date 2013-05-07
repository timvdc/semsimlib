#!/usr/bin/env python

"""matrix.py: distributional matrix superclass"""

__author__      = "Tim Van de Cruys"
__email__       = "timvdc@gmail.com"
__status__      = "development"

import numpy as np
import scipy
from semsimlib.sparse.dod import dod_matrix
from util import *
from similarityfunctions import *
import math
import fileinput

class Matrix:
    def __init__(self, instanceFile, featureFile):

        if isinstance(instanceFile,basestring):
            self.instances = readFileAsList(instanceFile)
            self.instanceDict = createDictFromList(self.instances)
        elif isinstance(instanceFile, list):
            self.instances = instanceFile
            self.instanceDict = createDictFromList(self.instances)
        elif instanceFile == None:
            self.instances = None

        if isinstance(featureFile,basestring):
            self.features = readFileAsList(featureFile)
            self.featureDict = createDictFromList(self.features)
        elif isinstance(featureFile,list):
            self.features = featureFile
            self.featureDict = createDictFromList(self.features)
        elif featureFile == None:
            self.features = None

        self.matrix = dod_matrix(
            (len(self.instances),len(self.features)),
            dtype=np.float64)

        self.weighted = None
        self.normalized = None
    
    def cleanup(self, instanceCutoff = 20, featureCutoff = 2, valueCutoff = 3):
        self.applyValueCutoff(valueCutoff)
        self.applyInstanceFeatureCutoff(instanceCutoff, featureCutoff)
        return

    def applyValueCutoff(self, valueCutoff):
            print " - triple check"
            for i in range(len(self.instances)):
                for j in self.matrix.data[i].keys():
                    if self.matrix[i,j] < valueCutoff:
                        self.matrix[i,j] = 0

    def applyInstanceFeatureCutoff(self, instanceCutoff, featureCutoff):
        while True:
            instancesCleaned = self.__cleanInstances(instanceCutoff)
            featuresCleaned = self.__cleanFeatures(featureCutoff)
            
            if instancesCleaned and featuresCleaned:
                break

        self.instanceDict = createDictFromList(self.instances)
        self.featureDict = createDictFromList(self.features)

    def loadFromCoordinateFormat(self, filename, delimiter = ' '):
        fileStream = fileinput.FileInput(filename,
                                         openhook=fileinput.hook_compressed)
        for line in fileStream:
            line = line.strip()
            freq, ninst, nfeat = line.split(delimiter)
            self.matrix[int(ninst),int(nfeat)] = int(freq)

###########################################
# Weighting functions
###########################################

# Local weighting functions

    # Logarithmic weighting
    def calculateLogWeighting(self):
        for i in range(len(self.vectorList)):
            for j in self.vectorList[i]:
                self.vectorList[i][j] = 1 + math.log(self.vectorList[i][j])


# Global weighting functions

    # Positive Pointwise Mutual Information
    def calculatePMI(self):
        try:
            self.instanceProbabilityList
            self.featureProbabilityList
        except AttributeError:
            self.__calculateMarginalProbabilities()

        vectorListPMI = [ {} for i in range(len(self.instances))]    
        for i in range(len(self.vectorList)):
            for j in self.vectorList[i]:
                PMIValue = math.log( ( self.vectorList[i][j] /
                                       self.frequencyTotal ) /
                                     ( self.instanceProbabilityList[i] *
                                       self.featureProbabilityList[j] ) )
                if PMIValue > 0:
                    vectorListPMI[i][j] = PMIValue
        self.vectorList = vectorListPMI
        self.applyInstanceFeatureCutoff()

    #probability (feat|instance) / prob(feat)
    #cfr. mitchell & lapata (2008, 2010)
    def calculateConditionalProbability(self):
        try:
            self.instanceFrequencyList
            self.featureFrequencyList
            self.frequencyTotal
        except AttributeError:
            self.__calculateSumFrequencies()

        vectorListProb = [ {} for i in range(len(self.instances))]
        for i in range(len(self.vectorList)):
            for j in self.vectorList[i]:
                probValue = ((self.vectorList[i][j] *
                              self.frequencyTotal) /
                             (self.instanceFrequencyList[i] *
                              self.featureFrequencyList[j]))
                vectorListProb[i][j] = probValue
        self.vectorList = vectorListProb

    #logodds
    def calculateLogOdds(self):
        try:
            self.instanceFrequencyList
            self.featureFrequencyList
            self.frequencyTotal
        except AttributeError:
            self.__calculateSumFrequencies()
        vectorListLogOdd = [ {} for i in range(len(self.instances))]
        for i in range(len(self.vectorList)):
            for j in self.vectorList[i]:
                k = self.vectorList[i][j]
                l = self.featureFrequencyList[j] - k
                m = self.instanceFrequencyList[i] - k
                n = self.frequencyTotal - (k + l + m)
                logOddValue = 2 * ( (k * math.log(k)) + (l * math.log(l)) \
                                    + (m * math.log(m)) + (n * math.log(n)) \
                                    - ((k + l) * math.log(k + l)) - ((k + m) * math.log(k + m)) \
                                    - ((l + n) * math.log(l + n)) - ((m + n) * math.log(m + n)) \
                                    + ((k + l + m + n) * math.log(k + l + m + n))
                                    )
                vectorListLogOdd[i][j] = logOddValue
        self.vectorList = vectorListLogOdd

    #Entropy
    def calculateEntropy(self):
        try:
            self.instanceFrequencyList
        except AttributeError:
            self.__calculateSumFrequencies()
        lognDoc = math.log(len(self.features))
        entropyInstances = [float(0) for i in range(len(self.instances))]
        for i in range(len(self.vectorList)):
            for j in self.vectorList[i]:
                pij = self.vectorList[i][j] / self.instanceFrequencyList[i]
                entropyInstances[i] += ( ( pij * math.log(pij) ) / lognDoc )
            entropyInstances[i] += 1
        for i in range(len(self.vectorList)):
            for j in self.vectorList[i]:
                self.vectorList[i][j] *=  entropyInstances[i]


###########################################
# Similarity calculation
###########################################

    def normalize(self, normTo='vnorm'):
        if not self.normalized == None:
            raise ValueError("matrix is already normalized")
        #normalize to vector length of one
        if normTo == 'vnorm':
            for i in range(len(self.vectorList)):
                sumVector = 0
                for j in self.vectorList[i]:
                    sumVector += self.vectorList[i][j] ** 2
                vectorNorm = math.sqrt(sumVector)
                for j in self.vectorList[i]:
                    self.vectorList[i][j] = self.vectorList[i][j] / vectorNorm
            self.normalized = 'vnorm'

        #normalize to 1 - conditional probability p(feature|instance)
        elif normTo == 'prob':
            for i in range(len(self.vectorList)):
                sumVector = 0
                for j in self.vectorList[i]:
                    sumVector += self.vectorList[i][j]
                for j in self.vectorList[i]:
                    self.vectorList[i][j] = self.vectorList[i][j] / float(sumVector)
            self.normalized = 'prob'
        else:
            raise ValueError("normalization parameter '" + normTo + "' not supported")

    def calculateMostSimilar(self, instance, topN = 20, similarity = 'cosine'):
        if similarity == 'cosine':
            if not self.normalized == 'vnorm':
                raise ValueError("matrix should be normalized to vector norm \
for cosine calculations")
            simFunction = calculateCosine
        elif similarity == 'skew':
            if not self.normalized == 'prob':
                raise ValueError("matrix should be normalized to probabilities \
for skew divergence calculations")
            simFunction = calculateSkewDivergence
        elif similarity == 'JS':
            if not self.normalized == 'prob':
                raise ValueError("matrix should be normalized to probabilities \
for Jensen-Shannon divergence calculations")
            simFunction = calculateJSDivergence
        else:
            raise ValueError("similarity function '" + similarity + "' unknown")

        nInstance = self.instanceDict[instance]
        cosineValueList = []
        for i in range(len(self.vectorList)):
            cosineValue = simFunction(self.vectorList[nInstance],self.vectorList[i])
            cosineValueList.append(cosineValue)
        sortedCosineList = [ [cosineValueList[i],i] for i in range(len(cosineValueList)) ]
        sortedCosineList.sort()
        if similarity == 'cosine':
            sortedCosineList.reverse()
        outputList = []
        for i in range(topN):
            outputList.append([self.instances[sortedCosineList[i][1]], sortedCosineList[i][0]])
        return outputList

###########################################
# Output
###########################################

    def dump(self, filename, format='cluto'):
        outFile = file(filename, 'w')
        if format == 'cluto':
            totalValueCount = int(0)
            for i in range(len(self.vectorList)):
                totalValueCount += len(self.vectorList[i])
            outFile.write(str(len(self.instances)) + ' ' +
                          str(len(self.features)) + ' ' +
                          str(totalValueCount) + '\n')
            for i in range(len(self.vectorList)):
                featList = self.vectorList[i].keys()
                featList.sort()
                for j in featList:
                    outFile.write(str(j + 1) + ' ' +
                                  "%.11f " % self.vectorList[i][j])
                outFile.write('\n')
        elif format == 'matlab':
            for i in range(len(self.vectorList)):
                for j in self.vectorList[i].keys():
                    outFile.write(str(i + 1) + '\t' +
                                  str(j + 1) + '\t' +
                                  str(self.vectorList[i][j]) + '\n') 
        else:
            print 'wrong output format'
        outFile.close()


###########################################
# Private functions
###########################################

    def __cleanInstances(self, instanceCutoff):
        print " - instance check"
        instancesCleaned = False
        removeInstances = []
        for i in range(len(self.instances)):
            if len(self.matrix.data[i]) < instanceCutoff:
                removeInstances.append(i)
        self.matrix.data = [self.matrix.data[i] for i in range(len(self.matrix.data))
                       if not i in removeInstances]
        self.instances = [self.instances[i] for i in range(len(self.instances))
                          if not i in removeInstances]
        self.matrix._shape = (len(self.instances),len(self.features))
        if removeInstances == []:
            instancesCleaned = True
        return instancesCleaned

    def __cleanFeatures(self, featureCutoff):
        print " - feature check"
        featuresCleaned = False
        removeFeatures = {}
        featureCountDict = [float(0) for i in range(len(self.features))]
        for i in range(len(self.instances)):
            for j in self.matrix.data[i].keys():
                featureCountDict[j] += 1
        for f in range(len(featureCountDict)):
            if featureCountDict[f] < featureCutoff:
                removeFeatures[f] = 1
        if not removeFeatures == {}:
            featureMappingList = [ i for i in range(len(self.features)) ]
            featureMappingDict = {}
            self.features = [ self.features[i]
                              for i in range(len(self.features))
                              if not removeFeatures.has_key(i) ]
            featureMappingList = [ featureMappingList[i]
                                   for i in range(len(featureMappingList))
                                   if not removeFeatures.has_key(i) ]

            print " - shrinking matrix"
            for i in range(len(featureMappingList)):
                featureMappingDict[featureMappingList[i]] = i
            newMatrix = dod_matrix(
                (len(self.instances),len(self.features)),
                dtype=np.float64)
            for i in range(len(self.matrix.data)):
                for j in self.matrix.data[i].keys():
                    try:
                        newMatrix[i,featureMappingDict[j]] = self.matrix[i,j]
                    except KeyError:
                        # if j is not in featureMappingDict
                        # then it has been removed
                        continue
            self.matrix = newMatrix
        else:
            featuresCleaned = True
        return featuresCleaned

    def __calculateSumFrequencies(self):
        instanceFrequencyList = [float(0) for i in range(len(self.instances))]
        featureFrequencyList = [float(0) for i in range(len(self.features))]
        frequencyTotal = float(0)
        for i in range(len(self.vectorList)):
            for j in self.vectorList[i]:
                instanceFrequencyList[i] += self.vectorList[i][j]
                featureFrequencyList[j] += self.vectorList[i][j]
                frequencyTotal += self.vectorList[i][j]
        self.instanceFrequencyList = instanceFrequencyList
        self.featureFrequencyList = featureFrequencyList
        self.frequencyTotal = frequencyTotal

    def __calculateMarginalProbabilities(self):
        try:
            self.instanceFrequencyList
            self.featureFrequencyList
            self.frequencyTotal
        except AttributeError:
            self.__calculateSumFrequencies()

        instanceProbabilityList = [ (self.instanceFrequencyList[i] / self.frequencyTotal)
                                    for i in range(len(self.instanceFrequencyList)) ]
        featureProbabilityList = [ (self.featureFrequencyList[i] / self.frequencyTotal)
                                   for i in range(len(self.featureFrequencyList)) ]
        self.instanceProbabilityList = instanceProbabilityList
        self.featureProbabilityList = featureProbabilityList

# 
# Semsim matrix class
# Tim Van de Cruys
#

from util import *
from similarityfunctions import *
import math

class Matrix:
    def __init__(self, instanceFile, featureFile, instanceCutoff = 20, featureCutoff = 2, valueCutoff = 1):
        if isinstance(instanceFile,basestring):
            self.instances = readFileAsList(instanceFile)
            self.instanceDict = createDictFromList(self.instances)
        elif instanceFile == None:
            self.instances = None

        if isinstance(featureFile,basestring):
            self.features = readFileAsList(featureFile)
            self.featureDict = createDictFromList(self.features)
        elif featureFile == None:
            self.features = None

        self.instanceCutoff = instanceCutoff
        self.featureCutoff = featureCutoff
        self.valueCutoff = valueCutoff

        self.vectorList = [ {} for i in range(len(self.instances)) ]

    def dryRun(self, stream):
        instanceCount = {}
        for n in self.instances:
            instanceCount[n] = 0
        featureCount = {}
        for f in self.features:
            featureCount[f] = 0
        for freq,feature,instance in stream:
            if freq >= self.valueCutoff:
                if self.instanceDict.has_key(instance) and self.featureDict.has_key(feature):
                    instanceCount[instance] += 1
                    featureCount[feature] += 1

        instances = [ i for i in self.instances if instanceCount[i] >= self.instanceCutoff ]
        features = [ i for i in self.features if featureCount[i] >= self.featureCutoff ]
        self.instances = instances
        self.features = features
        self.instanceDict = createDictFromList(self.instances)
        self.featureDict = createDictFromList(self.features)

    def fill(self, stream):
        for freq,feature,instance in stream:
            if self.instanceDict.has_key(instance) and self.featureDict.has_key(feature):
                nInstance = self.instanceDict[instance]
                nFeature = self.featureDict[feature]
                try:
                    self.vectorList[nInstance][nFeature] += freq
                except KeyError:
                    self.vectorList[nInstance][nFeature] = freq


    def applyValueCutoff(self):
        if self.valueCutoff > 1:
            print " - triple check"
            for i in range(len(self.vectorList)):
                for j in self.vectorList[i].keys():
                    if self.vectorList[i][j] < self.valueCutoff:
                        del self.vectorList[i][j]

    def applyInstanceFeatureCutoff(self):
        while True:
            instancesCleaned = self.__cleanInstances()
            featuresCleaned = self.__cleanFeatures()
            
            if instancesCleaned and featuresCleaned:
                break

        self.instanceDict = createDictFromList(self.instances)
        self.featureDict = createDictFromList(self.features)


###########################################
# Weighting functions
###########################################

# Local weighting functions

    # Logarithmic weighting
    def calculateLogWeighting(self):
        for i in range(len(self.vectorList)):
            for j in self.vectorList[i]:
                self.vectorList[i][j] = math.log(self.vectorList[i][j] + 1)


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

    #logodds
    def calculateLogOdds(self):
        vectorListLogOdd = [ {} for i in range(len(self.instances))]
        for i in range(len(self.vectorList)):
            for j in self.vectorList[i]:
                k = self.vectorList[i][j]
                l = featureFrequencyList[j] - k
                m = instanceFrequencyList[i] - k
                n = frequencyTotal - (k + l + m)
                logOddValue = 2 * ( ( k * math.log(k) ) + ( l * math.log(l) ) \
                                    + ( m * math.log(m) ) + ( n * math.log(n)) \
                                    - ((k+l) * math.log(k+l)) - ((k+m) * math.log(k+m)) \
                                    - ((l+n) * math.log(l+n)) - ((m+n) * math.log(m+n)) \
                                    + ((k+l+m+n) * math.log(k+l+m+n)) )
                vectorListLogOdd[i][j] = logOddValue
        self.vectorList = vectorListLogOdd

    #Entropy
    def calculateEntropy(self):
        lognDoc = math.log(len(self.features))
        entropyInstances = [float(0) for i in range(len(self.instances))]
        for i in range(len(self.vectorList)):
            for j in self.vectorList[i]:
                pij = self.vectorList[i][j] / instanceFrequencyList[i]
                entropyInstances[i] += ( ( pij * math.log(pij) ) / lognDoc )
            entropyInstances[i] += 1
        for i in range(len(self.vectorList)):
            for j in self.vectorList[i]:
                self.vectorList[i][j] *=  entropyInstances[i]


###########################################
# Similarity calculation
###########################################

    def normalizeMatrix(self, normTo='cosine'):
        #Normalize to vector length
        if normTo == 'cosine':
            for i in range(len(self.vectorList)):
                sumVector = 0
                for j in self.vectorList[i]:
                    sumVector += self.vectorList[i][j] ** 2
                vectorNorm = math.sqrt(sumVector)
                for j in self.vectorList[i]:
                    self.vectorList[i][j] = self.vectorList[i][j] / vectorNorm

        #Normalize to 1 (probability vector feat|noun)
        elif normTo == 'prob':
            for i in range(len(self.vectorList)):
                sumVector = 0
                for j in self.vectorList[i]:
                    sumVector += self.vectorList[i][j]
                for j in self.vectorList[i]:
                    self.vectorList[i][j] = self.vectorList[i][j] / float(sumVector)

    def calculateMostSimilar(self, instance, topN = 20, similarity = 'cosine'):
        if similarity == 'cosine':
            simFunction = calculateCosine
        elif similarity == 'skew':
            simFunction = calculatekewDivergence
        elif similarity == 'JS':
            simFunction = calculateJSDivergence
        else:
            print 'unknown similarity function; using cosine..'
            simFunction == calculateCosine

        nInstance = self.instanceDict[instance]
        cosineValueList = []
        for i in range(len(self.vectorList)):
            cosineValue = simFunction(self.vectorList[nInstance],self.vectorList[i])
            cosineValueList.append(cosineValue)
        sortedCosineList = [ [cosineValueList[i],i] for i in range(len(cosineValueList)) ]
        sortedCosineList.sort()
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
            outFile.write(str(len(self.instances)) + ' ' + str(len(self.features)) + ' ' + str(totalValueCount) + '\n')
            for i in range(len(self.vectorList)):
                featList = self.vectorList[i].keys()
                featList.sort()
                for j in featList:
                    outFile.write(str(j + 1) + ' ' + "%.11f " % self.vectorList[i][j])
                outFile.write('\n')
        elif format == 'matlab':
            for i in range(len(self.vectorList)):
                for j in self.vectorList[i].keys():
                    outFile.write(str(i + 1) + '\t' + str(j + 1) + '\t' + str(self.vectorList[i][j]) + '\n') 
        else:
            print 'wrong output format'
        outFile.close()


###########################################
# Private functions
###########################################

    def __cleanInstances(self):
        print " - instance check"
        instancesCleaned = False
        removeInstances = []
        for i in range(len(self.instances)):
            if len(self.vectorList[i]) < self.instanceCutoff:
                removeInstances.append(i)
        self.vectorList = [ self.vectorList[i] for i in range(len(self.vectorList)) if not i in removeInstances ]
        self.instances = [ self.instances[i] for i in range(len(self.instances)) if not i in removeInstances ]
        if removeInstances == []:
            instancesCleaned = True
        return instancesCleaned

    def __cleanFeatures(self):
        print " - feature check"
        featuresCleaned = False
        removeFeatures = {}
        featureCountDict = [float(0) for i in range(len(self.features))]
        for i in range(len(self.instances)):
            for j in self.vectorList[i].keys():
                featureCountDict[j] += 1
        for f in range(len(featureCountDict)):
            if featureCountDict[f] < self.featureCutoff:
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
            newVectorList = [ {} for i in range(len(self.instances))]
            for i in range(len(self.vectorList)):
                for j in self.vectorList[i].keys():
                    try:
                        newVectorList[i][featureMappingDict[j]] = self.vectorList[i][j]
                    except KeyError:
                        if not removeFeatures.has_key(j):
                            print 'foutje'
            self.vectorList = newVectorList
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

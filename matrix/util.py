def readFileAsList(inputFile):
    instanceList = []
    for line in file(inputFile):
        line = line.rstrip()
        instanceList.append(line)
    return instanceList

def createDictFromList(instanceList):
    instanceDict = {}
    for i in range(len(instanceList)):
        instanceDict[instanceList[i]] = i
    return instanceDict

def dryRun_window(stream, instanceCutoff, featureCutoff):
    wordFrequencyDict = {}
    for wordList in stream:
        for word in wordList:
            try:
                wordFrequencyDicy[word] += 1
            except KeyError:
                wordFrequencyDict[word] = 1

    instances = [i for i in wordFrequencyDict
                 if wordFrequencyDict[i] >= instanceCutoff]
    features = [i for i in wordCount
                if wordCount[i] >= featureCutoff]
    return instances, features

def dryRun_dep(stream, instanceCutoff, featureCutoff):
    instanceCount = {}
    featureCount = {}
    for tripleList in stream:
        for freq,instance,feature in tripleList:
            try:
                instanceCount[instance] += freq
            except KeyError:
                instanceCount[instance] = freq
            try:
                featureCount[feature] += freq
            except KeyError:
                featureCount[feature] = freq

    instances = [i for i in instanceCount
                 if instanceCount[i] >= self.instanceCutoff]
    features = [i for i in featureCount
                if featureCount[i] >= self.featureCutoff]
    return instances, features

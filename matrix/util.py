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


from .corpusreader import *

class UKWacCorpusReader(CorpusReader):
    def __init__(self, fileids):
        CorpusReader.__init__(self, fileids)

    def __iter__(self):
        return self

    def next(self):
        stopFlag = False
        tokenList = []
        while not stopFlag:
            line = self.fileStream.next()
            line = line.rstrip()
            if line == '</s>':
                stopFlag = True
            elif line == '<s>' or line == '</text>' or line.startswith('<text'):
                continue
            else:
                token = line.split('\t')
                tokenList.append(token)
        return tokenList

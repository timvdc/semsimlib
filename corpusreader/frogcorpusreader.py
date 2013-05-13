from corpusreader import *
import re

posMatch = re.compile(r'([A-Z]+)\(.*')

class FrogCorpusReader(CorpusReader):
    def __init__(self, fileids):
        CorpusReader.__init__(self, fileids)

    def __iter__(self):
        return self

    def next(self):
        sentenceList = []
        while True:
            line = self.fileStream.next()
            line = line.rstrip()
            if not line:
                return sentenceList
            else:
                tokenFields = line.split('\t')
                lemma = tokenFields[2]
                pos = tokenFields[4]
                posMatcher = posMatch.match(pos)
                broadPos = posMatcher.group(1)
                sentenceList.append(lemma + '/' + broadPos)

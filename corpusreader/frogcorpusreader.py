from .corpusreader import *
import re

posMatch = re.compile(r'([A-Z]+)\(.*')

class FrogCorpusReader(CorpusReader):
    def __init__(self, fileids):
        CorpusReader.__init__(self, fileids)

    def __iter__(self):
        return self

    def __next__(self):
        sentenceList = []
        while True:
            line = next(self.fileStream)
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

from .corpusreader import *

class POSCorpusReader(CorpusReader):
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
                pos = tokenFields[3]
                sentenceList.append(lemma + '/' + pos)

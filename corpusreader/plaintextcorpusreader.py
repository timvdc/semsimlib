from corpusreader import *

class PlaintextCorpusReader(CorpusReader):
    def __init__(self, fileids):
        CorpusReader.__init__(self, fileids)

    def __iter__(self):
        return self

    def next(self):
        line = self.fileStream.next()
        line = line.rstrip()
        words = line.split(' ')
        return words

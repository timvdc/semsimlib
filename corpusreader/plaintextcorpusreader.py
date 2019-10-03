from .corpusreader import *

class PlaintextCorpusReader(CorpusReader):
    def __init__(self, fileids):
        CorpusReader.__init__(self, fileids)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            line = self.fileStream.__next__()
        except UnicodeDecodeError:
            return []
        else:
            line = line.rstrip()
            words = line.split(' ')
            return words

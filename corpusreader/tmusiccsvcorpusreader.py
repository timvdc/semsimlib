from .corpusreader import *

class TMusicCSVCorpusReader(CorpusReader):
    def __init__(self, fileids):
        CorpusReader.__init__(self, fileids)
        next(self.fileStream) #skip first line

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.fileStream)
        line = line.strip()
        CSVList = line.split(',')
        author = CSVList[4]
        lyrics = CSVList[6]
        words = lyrics.split(' ')
        return (author, words)

from corpusreader import *

class DependencyCorpusReader(CorpusReader):
    def __init__(self, fileids, separator='#'):
        CorpusReader.__init__(self, fileids)
        self.separator = separator

    def __iter__(self):
        return self

    def next(self):
        errorFlag = True
        while errorFlag:
            line = self.fileStream.next()
            line = line.rstrip()
            try:
                freq, dep1, rel, dep2 = line.split(self.separator)
            except ValueError:
                print "ValueError in line: " + line
                continue
            else:
                errorFlag = False
        return int(freq), dep1, rel + '#' + dep2

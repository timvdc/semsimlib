from corpusreader import *
import re
import sys
import fileinput

depMatch = re.compile(r'([^\(]+)\((.+)-([0-9]{1,5})\'*, (.+)-([0-9]{1,5})\'*\)')

class UKWacParsedCorpusReader(CorpusReader):
    def __init__(self, fileids_final):
        if isinstance(fileids_final, basestring):
            fileids_deps = [os.path.abspath(fileids_final.replace('final','deps'))]
            fileids_final = [os.path.abspath(fileids_final)]
        elif isinstance(fileids_final, list):
            fileids_final = [os.path.abspath(id_final)
                             for id_final in fileids_final]
            fileids_deps = [os.path.abspath(id_deps.replace('final','deps'))
                            for id_deps in fileids_final]
        else:
            raise AssertionError('fileids have to be strings or list of strings')

        self._fileids_final = fileids_final
        self._fileids_deps = fileids_deps

        self.fileStream_final = fileinput.FileInput(self._fileids_final,
                                                    openhook=fileinput.hook_compressed)
        self.fileStream_deps = fileinput.FileInput(self._fileids_deps,
                                                   openhook=fileinput.hook_compressed)


    def __iter__(self):
        return self

    # Number of filthy hacks; this will all look much nicer
    # once parsed ukWaC is encoded in proper XML
    def next(self):
        tokenListAll = []
        sentCount = 0
        returnFlag = False
        while not returnFlag:
            stopFlag = False
            tokenList = []
            while not stopFlag:
                line = self.fileStream_final.next()
                line = line.rstrip()
                if line == '</doc>':
                    stopFlag = True
                    returnFlag = True
                elif line.startswith('<doc'):
                    continue
                elif line == '':
                    sentCount += 1
                    stopFlag = True
                else:
                    tokens = line.split('\t')
                    tokenList.append([tokens[2], tokens[3], []])
                if sentCount >= 5:
                    stopFlag = True
                    returnFlag = True

            stopFlag_deps = False
            depList = []
            while not stopFlag_deps:
                line = self.fileStream_deps.next()
                line = line.rstrip()
                if line == '</doc>':
                    stopFlag_deps = True
                elif line.startswith('<doc'):
                    continue
                elif line == '':
                    stopFlag_deps = True
                else:
                    depList.append(line)
            for dep in depList:
                depMatchObject = depMatch.match(dep)
                try:
                    rel = depMatchObject.group(1)
                    w1 = depMatchObject.group(2)
                    pos1 = int(depMatchObject.group(3)) - 1
                    w2 = depMatchObject.group(4)
                    pos2 = int(depMatchObject.group(5)) - 1
                except AttributeError:
                    continue
                else:
                    if not pos1 == pos2:
                        try:
                            tokenList[pos1][2].append([rel,tokenList[pos2][0],
                                                       tokenList[pos2][1]])
                            if rel.startswith('conj'):
                                tokenList[pos2][2].append([rel,
                                                           tokenList[pos1][0],
                                                           tokenList[pos1][1]])
                            else:
                                tokenList[pos2][2].append([rel + '-1',
                                                           tokenList[pos1][0],
                                                           tokenList[pos1][1]])
                        except IndexError:
                            sys.stderr.write('err ' + str(tokenList) + '\n')
            tokenListAll.extend(tokenList)
            outputList = []
            for token in tokenListAll:
                for relation in token[2]:
                    outputList.append((1,
                                       token[0] + '/' + token[1],
                                       relation[0] + '#' +
                                       relation[1] + '/' +
                                       relation[2]))
        return outputList

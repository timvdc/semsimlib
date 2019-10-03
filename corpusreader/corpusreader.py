from .util import *
import fileinput

class CorpusReader:
    def __init__(self, fileids):
#        if not isinstance(root, basestring):
#            raise AssertionError('root has to be string')
#        self._root = root
        
        if isinstance(fileids, str):
            fileids = [ os.path.abspath(fileids) ]

        elif isinstance(fileids, list):
            fileids = [ os.path.abspath(fileid) for fileid in fileids ]
        else:
            raise AssertionError('fileids has to be string or list')

        self._fileids = fileids

        self.fileStream = fileinput.FileInput(self._fileids,
                                              openhook=fileinput.hook_compressed)


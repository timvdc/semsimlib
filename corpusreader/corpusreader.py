from .util import *
import fileinput

import os

def hook_compressed_text(filename, mode, encoding='utf8'):
    ext = os.path.splitext(filename)[1]
    if ext == '.gz':
        import gzip
        return gzip.open(filename, mode + 't', encoding=encoding)
    elif ext == '.bz2':
        import bz2
        return bz2.open(filename, mode + 't', encoding=encoding)
    else:
        return open(filename, mode + 't', encoding=encoding)

class CorpusReader:
    def __init__(self, fileids):
        
        if isinstance(fileids, str):
            fileids = [ os.path.abspath(fileids) ]

        elif isinstance(fileids, list):
            fileids = [ os.path.abspath(fileid) for fileid in fileids ]
        else:
            raise AssertionError('fileids has to be string or list')

        self._fileids = fileids

        self.fileStream = fileinput.FileInput(self._fileids,
                                              openhook=hook_compressed_text)


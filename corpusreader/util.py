import os
import fnmatch

def find_corpus_fileids(root, regexp):
    if isinstance(root, str):
        items = []
        for dirname, subdirs, fileids in os.walk(root):
            items.extend([os.path.abspath(os.path.join(dirname, fileid))
                          for fileid in fileids
                          if fnmatch.fnmatch(fileid, regexp)])
        return sorted(items) 
    else: 
        raise AssertionError("Don't know how to handle %r" % root)

# 
# Semsim depmatrix class
# Tim Van de Cruys
#

from matrix import *
import semsimlib

class DependencyMatrix(Matrix):
    def __init__(self, dependencyFilenames, instanceFile = None,
                 featureFile = None, instanceCutoff = 20, featureCutoff = 2,
                 valueCutoff = 3, dryRun = True, cleanup = True):
        Matrix.__init__(self, instanceFile, featureFile)
        self.dependencyFilenames = dependencyFilenames
        self.instanceCutoff = instanceCutoff
        self.featureCutoff = featureCutoff
        self.valueCutoff = valueCutoff

        if dryRun:
            dependencyStream = semsimlib.DependencyCorpusReader(self.dependencyFilenames)
            self.dryRun(dependencyStream)

        dependencyStream = semsimlib.DependencyCorpusReader(self.dependencyFilenames)
        self.fill(dependencyStream)

        if cleanup:
            self.applyValueCutoff()
            self.applyInstanceFeatureCutoff()

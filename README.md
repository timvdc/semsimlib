# semsimlib - a python library for semantic similarity

v2.0

python3

tim.vandecruys@irit.fr

www.timvandecruys.be

## Introduction

The module contains a number of classes for the construction of
co-occurrence matrices according to the principles of distributional
similarity.

Required modules: `numpy`, `scipy`, `cython` (for fast nmf computations)

## Code example

```
import semsimlib

m = semsimlib.WindowMatrix('corpus_input.txt')

m.calculatePMI()

m_np = semsimlib.NPMatrix(m)
m_nmf = semsimlib.NMFMatrix(m_np,rdim=50)

m_nmf.compute()
m_nmf.normalize()

```

## Cython NMF

In order to compile the cython code, run

```
python setup.py build_ext --inplace
```

and copy the resulting `.so` file to the `nmf` directory
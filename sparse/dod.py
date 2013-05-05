"""List of Dictionaries based matrix"""

__docformat__ = "restructuredtext en"

__all__ = ['dod_matrix', 'isspmatrix_dod']

from itertools import izip

import numpy as np
import scipy.sparse

from scipy.sparse.base import spmatrix, isspmatrix
from scipy.sparse.sputils import isdense, getdtype, isshape, isintlike, isscalarlike, upcast

try:
    from operator import isSequenceType as _is_sequence
except ImportError:
    def _is_sequence(x):
        return (hasattr(x, '__len__') or hasattr(x, '__next__')
                or hasattr(x, 'next'))

class dod_matrix(spmatrix, dict):
    """
    List Of Dictionaries based sparse matrix.

    The format is called dod (dictionary of dictionaries)
    in order to smoothly cooperate with scipy.sparse,
    but the first index is actually implemented as a list

    This is an efficient structure for constructing sparse
    matrices incrementally.

    This can be instantiated in several ways:
        dok_matrix(D)
            with a dense matrix, D

        dok_matrix(S)
            with a sparse matrix, S

        dok_matrix((M,N), [dtype])
            create the matrix with initial shape (M,N)
            dtype is optional, defaulting to dtype='d'

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Allows for efficient O(1) access of individual elements.
    Duplicates are not allowed.
    Can be efficiently converted to a coo_matrix once constructed.

    Examples
    --------
    >>> from scipy.sparse import *
    >>> from scipy import *
    >>> S = dok_matrix((5,5), dtype=float32)
    >>> for i in range(5):
    >>>     for j in range(5):
    >>>         S[i,j] = i+j # Update element

    """

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        dict.__init__(self)
        spmatrix.__init__(self)

        self.dtype = getdtype(dtype, default=float)
        if isspmatrix(arg1): # Sparse ctor
            if isspmatrix_dod(arg1) and copy:
                arg1 = arg1.copy()
            else:
                arg1 = arg1.todod()

            if dtype is not None:
                arg1 = arg1.astype(dtype)

            self.update(arg1)
            self.shape = arg1.shape
            self.dtype = arg1.dtype
        elif isinstance(arg1, tuple) and isshape(arg1): # (M,N)
            M, N = arg1
            self.shape = (M, N)
            self.data = [ {} for i in range(M)]
        else: # Dense ctor
            raise ValueError("Dense constructor not implemented")

    def getnnz(self):
        return sum([len(rowvals) for rowvals in self.data])
    nnz = property(fget=getnnz)

    def __len__(self):
        raise ValueError("ambiguous")

    def _get1(self, i, j):

        if i < 0:
            i += self.shape[0]
        if i < 0 or i >= self.shape[0]:
            raise IndexError('row index out of bounds')

        if j < 0:
            j += self.shape[1]
        if j < 0 or j >= self.shape[1]:
            raise IndexError('column index out of bounds')

        try:
            return self.data[i][j]
        except KeyError:
            return self.dtype.type(0)

    def __getitem__(self, index):
        """Return the element(s) index=(i, j)
"""
        try:
            i, j = index
        except (AssertionError, TypeError):
            raise IndexError('invalid index')

        if not np.isscalar(i) and np.isscalar(j):
            # warn('Indexing into a lil_matrix with multiple indices is slow. '
            #      'Pre-converting to CSC or CSR beforehand is more efficient.',
            #      SparseEfficiencyWarning)
            raise ValueError('no slicing implemented')

        if np.isscalar(i):
            if np.isscalar(j):
                return self._get1(i, j)

    def _insertat2(self, i, j, x):
        """ helper for __setitem__: insert a value in the given row/data at
        column j. """

        if i < 0:
            i += self.shape[0]
        if i < 0 or i >= self.shape[0]:
            raise IndexError('row index out of bounds')

        if j < 0: #handle negative column indices
            j += self.shape[1]
        if j < 0 or j >= self.shape[1]:
            raise IndexError('column index out of bounds')

        if not np.isscalar(x):
            raise ValueError('setting an array element with a sequence')
        try:
            x = self.dtype.type(x)
        except:
            raise TypeError('Unable to convert value (%s) to dtype [%s]' % (x,self.dtype.name))

        if x != 0:
            self.data[i][j] = x
        else:
            del self.data[i][j]

    def __setitem__(self, index, x):
        try:
            i, j = index
        except (ValueError, TypeError):
            raise IndexError('invalid index')

        # shortcut for common case of single entry assign:
        if np.isscalar(x) and np.isscalar(i) and np.isscalar(j):
            self._insertat2(i, j, x)
            return

    # def __add__(self, other):
    #     # First check if argument is a scalar
    #     if isscalarlike(other):
    #         new = dok_matrix(self.shape, dtype=self.dtype)
    #         # Add this scalar to every element.
    #         M, N = self.shape
    #         for i in xrange(M):
    #             for j in xrange(N):
    #                 aij = self.get((i, j), 0) + other
    #                 if aij != 0:
    #                     new[i, j] = aij
    #         #new.dtype.char = self.dtype.char
    #     elif isinstance(other, dok_matrix):
    #         if other.shape != self.shape:
    #             raise ValueError("matrix dimensions are not equal")
    #         # We could alternatively set the dimensions to the the largest of
    #         # the two matrices to be summed.  Would this be a good idea?
    #         new = dok_matrix(self.shape, dtype=self.dtype)
    #         new.update(self)
    #         for key in other.keys():
    #             new[key] += other[key]
    #     elif isspmatrix(other):
    #         csc = self.tocsc()
    #         new = csc + other
    #     elif isdense(other):
    #         new = self.todense() + other
    #     else:
    #         raise TypeError("data type not understood")
    #     return new

    # def __radd__(self, other):
    #     # First check if argument is a scalar
    #     if isscalarlike(other):
    #         new = dok_matrix(self.shape, dtype=self.dtype)
    #         # Add this scalar to every element.
    #         M, N = self.shape
    #         for i in xrange(M):
    #             for j in xrange(N):
    #                 aij = self.get((i, j), 0) + other
    #                 if aij != 0:
    #                     new[i, j] = aij
    #     elif isinstance(other, dok_matrix):
    #         if other.shape != self.shape:
    #             raise ValueError("matrix dimensions are not equal")
    #         new = dok_matrix(self.shape, dtype=self.dtype)
    #         new.update(self)
    #         for key in other:
    #             new[key] += other[key]
    #     elif isspmatrix(other):
    #         csc = self.tocsc()
    #         new = csc + other
    #     elif isdense(other):
    #         new = other + self.todense()
    #     else:
    #         raise TypeError("data type not understood")
    #     return new

    # def __neg__(self):
    #     new = dok_matrix(self.shape, dtype=self.dtype)
    #     for key in self.keys():
    #         new[key] = -self[key]
    #     return new

    # def _mul_scalar(self, other):
    #     # Multiply this scalar by every element.
    #     new = dok_matrix(self.shape, dtype=self.dtype)
    #     for (key, val) in self.iteritems():
    #         new[key] = val * other
    #     return new

    # def _mul_vector(self, other):
    #     #matrix * vector
    #     result = np.zeros( self.shape[0], dtype=upcast(self.dtype,other.dtype) )
    #     for (i,j),v in self.iteritems():
    #         result[i] += v * other[j]
    #     return result

    # def _mul_multivector(self, other):
    #     #matrix * multivector
    #     M,N = self.shape
    #     n_vecs = other.shape[1] #number of column vectors
    #     result = np.zeros( (M,n_vecs), dtype=upcast(self.dtype,other.dtype) )
    #     for (i,j),v in self.iteritems():
    #         result[i,:] += v * other[j,:]
    #     return result

    # def __imul__(self, other):
    #     if isscalarlike(other):
    #         # Multiply this scalar by every element.
    #         for (key, val) in self.iteritems():
    #             self[key] = val * other
    #         #new.dtype.char = self.dtype.char
    #         return self
    #     else:
    #         return NotImplementedError


    # def __truediv__(self, other):
    #     if isscalarlike(other):
    #         new = dok_matrix(self.shape, dtype=self.dtype)
    #         # Multiply this scalar by every element.
    #         for (key, val) in self.iteritems():
    #             new[key] = val / other
    #         #new.dtype.char = self.dtype.char
    #         return new
    #     else:
    #         return self.tocsr() / other


    # def __itruediv__(self, other):
    #     if isscalarlike(other):
    #         # Multiply this scalar by every element.
    #         for (key, val) in self.iteritems():
    #             self[key] = val / other
    #         return self
    #     else:
    #         return NotImplementedError

    # # What should len(sparse) return? For consistency with dense matrices,
    # # perhaps it should be the number of rows?  For now it returns the number
    # # of non-zeros.

    # def transpose(self):
    #     """ Return the transpose
    #     """
    #     M, N = self.shape
    #     new = dok_matrix((N, M), dtype=self.dtype)
    #     for key, value in self.iteritems():
    #         new[key[1], key[0]] = value
    #     return new

    # def conjtransp(self):
    #     """ Return the conjugate transpose
    #     """
    #     M, N = self.shape
    #     new = dok_matrix((N, M), dtype=self.dtype)
    #     for key, value in self.iteritems():
    #         new[key[1], key[0]] = np.conj(value)
    #     return new

    def copy(self):
        new = dod_matrix(self.shape, dtype=self.dtype)
        new.update(self)
        return new

    # def take(self, cols_or_rows, columns=1):
    #     # Extract columns or rows as indictated from matrix
    #     # assume cols_or_rows is sorted
    #     new = dok_matrix(dtype=self.dtype)    # what should the dimensions be ?!
    #     indx = int((columns == 1))
    #     N = len(cols_or_rows)
    #     if indx: # columns
    #         for key in self.keys():
    #             num = np.searchsorted(cols_or_rows, key[1])
    #             if num < N:
    #                 newkey = (key[0], num)
    #                 new[newkey] = self[key]
    #     else:
    #         for key in self.keys():
    #             num = np.searchsorted(cols_or_rows, key[0])
    #             if num < N:
    #                 newkey = (num, key[1])
    #                 new[newkey] = self[key]
    #     return new

    # def split(self, cols_or_rows, columns=1):
    #     # Similar to take but returns two arrays, the extracted columns plus
    #     # the resulting array.  Assumes cols_or_rows is sorted
    #     base = dok_matrix()
    #     ext = dok_matrix()
    #     indx = int((columns == 1))
    #     if indx:
    #         for key in self.keys():
    #             num = np.searchsorted(cols_or_rows, key[1])
    #             if cols_or_rows[num] == key[1]:
    #                 newkey = (key[0], num)
    #                 ext[newkey] = self[key]
    #             else:
    #                 newkey = (key[0], key[1]-num)
    #                 base[newkey] = self[key]
    #     else:
    #         for key in self.keys():
    #             num = np.searchsorted(cols_or_rows, key[0])
    #             if cols_or_rows[num] == key[0]:
    #                 newkey = (num, key[1])
    #                 ext[newkey] = self[key]
    #             else:
    #                 newkey = (key[0]-num, key[1])
    #                 base[newkey] = self[key]
    #     return base, ext

    def tocoo(self):
        """ Return a copy of this matrix in COOrdinate format"""
        from scipy.sparse.coo import coo_matrix
        if self.nnz == 0:
            return coo_matrix(self.shape, dtype=self.dtype)
        else:
            data = []
            indices = []
            for i in range(self.shape[0]):
                for j in self.data[i].keys():
                    data.append(self.data[i][j])
                    indices.append((i,j))
            data    = np.asarray(data, dtype=self.dtype)
            indices = np.asarray(indices, dtype=np.intc).T
            print data
            print indices
            return coo_matrix((data,indices), shape=self.shape, dtype=self.dtype)

    def todod(self,copy=False):
        if copy:
            return self.copy()
        else:
            return self

    def tocsr(self):
        """ Return a copy of this matrix in Compressed Sparse Row format"""
        return self.tocoo().tocsr()

    def tocsc(self):
        """ Return a copy of this matrix in Compressed Sparse Column format"""
        return self.tocoo().tocsc()

    def tolil(self):
        return self.tocoo().tolil()

    # def toarray(self, order=None, out=None):
    #     """See the docstring for `spmatrix.toarray`."""
    #     return self.tocoo().toarray(order=order, out=out)

    # def resize(self, shape):
    #     """ Resize the matrix in-place to dimensions given by 'shape'.

    #     Any non-zero elements that lie outside the new shape are removed.
    #     """
    #     if not isshape(shape):
    #         raise TypeError("dimensions must be a 2-tuple of positive"
    #                          " integers")
    #     newM, newN = shape
    #     M, N = self.shape
    #     if newM < M or newN < N:
    #         # Remove all elements outside new dimensions
    #         for (i, j) in self.keys():
    #             if i >= newM or j >= newN:
    #                 del self[i, j]
    #     self._shape = shape



def isspmatrix_dod(x):
    return isinstance(x, dod_matrix)

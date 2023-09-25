#cython: language_level=3
import numpy as np
cimport numpy as np
from copy import deepcopy as deepcopy
DTYPE = np.float32
INF= np.inf
np.import_array()

def onecycle_cover(np.ndarray[float, ndim=2] distance_matrix):
    cdef int max_idx = np.argmax(distance_matrix)
    cdef int row =0
    cdef int col =0
    cdef int first_row = 0
    cdef int idx = 0
    cdef np.ndarray[int, ndim=1] sorted_indices = np.zeros(distance_matrix.shape[0], dtype=np.int32)
    np.fill_diagonal(distance_matrix, -INF)
    row, col = max_idx//distance_matrix.shape[0], max_idx % distance_matrix.shape[0]
    first_row = deepcopy(row)
    sorted_indices[row] = col
    distance_matrix[:, row] = -INF
    while idx < distance_matrix.shape[0] - 2:
        idx += 1
        row = col
        col = np.argmax(distance_matrix[row])
        sorted_indices[row] = col
        distance_matrix[:, row] = -INF
    sorted_indices[col] = first_row
    return sorted_indices

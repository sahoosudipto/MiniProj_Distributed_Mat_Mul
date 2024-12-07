
import unittest
import numpy as np
from dist_matrix_mult import serial_matrix_mult, distributed_matrix_mult
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class TestMatrixMult(unittest.TestCase):

    def test_serial_mult(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C_expected = np.array([[19, 22], [43, 50]])
        C_actual = serial_matrix_mult(A, B)
        np.testing.assert_array_equal(C_actual, C_expected)

    def test_distributed_mult(self):
        m = 4
        n = 4
        p = 4
        A = np.random.rand(m, n)
        B = np.random.rand(n, p)

        if rank == 0:
            C_serial = serial_matrix_mult(A, B)

        C_dist = distributed_matrix_mult(A, B, m, n, p)

        if rank == 0:
            np.testing.assert_allclose(C_serial, C_dist, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    unittest.main()

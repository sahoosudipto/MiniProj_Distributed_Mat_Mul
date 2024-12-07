from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def serial_matrix_mult(A, B):
    return np.dot(A, B)

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def distribute_matrix(matrix, comm, root=0, axis=0):  # Add axis parameter
    """Distributes a matrix across processes."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        # Split the matrix along the specified axis
        matrix_chunks = np.array_split(matrix, size, axis=axis)  
    else:
        matrix_chunks = None

    local_chunk = comm.scatter(matrix_chunks, root=root)
    return local_chunk


def distributed_matrix_mult(A, B, m, n, p):
    """Performs distributed matrix multiplication."""
    from mpi4py import MPI
    import numpy as np

    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribute A by rows and B by columns
    local_A = distribute_matrix(A, comm, axis=0)  # Each process gets a subset of rows of A
    local_B = distribute_matrix(B.T, comm, axis=0).T  # Each process gets a subset of cols of B, transposed back

    # Perform local computation
    local_C = np.dot(local_A, local_B)

    # Prepare the output matrix on the root process
    if rank == 0:
        C = np.zeros((m, p))  # Root process will gather all parts into this matrix
    else:
        C = None

    # Gather all local_C parts into the root process
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty((m, p))  # Allocate a buffer for gathering
    comm.Gather(local_C, recvbuf, root=0)

    # Root process assembles the matrix from chunks
    if rank == 0:
        rows_per_proc = m // size
        extra_rows = m % size
        offset = 0
        for i in range(size):
            rows = rows_per_proc + (1 if i < extra_rows else 0)
            C[offset:offset+rows, :] = recvbuf[offset:offset+rows, :]
            offset += rows
        return C
    else:
        return None


# def distributed_matrix_mult(A, B, m, n, p):
#     """
#     Performs distributed matrix multiplication using MPI.

#     Args:
#         A: Numpy array representing the first matrix.
#         B: Numpy array representing the second matrix.
#         m: Number of rows in matrix A.
#         n: Number of columns in matrix A (and rows in matrix B).
#         p: Number of columns in matrix B.

#     Returns:
#         C: Numpy array representing the result matrix.
#            Only rank 0 will have the complete result.
#     """

#     # Calculate local rows for each process, accounting for uneven distribution
#     local_rows = m // size + (rank < m % size)  
#     start_row = sum(m // size + (i < m % size) for i in range(rank))
#     end_row = start_row + local_rows

#     # Scatter matrix A (using slicing for uneven distribution)
#     local_A = np.empty((local_rows, n))
#     comm.Scatterv([A, (local_rows * np.ones(size, dtype=int), 
#                        np.arange(size) * (m // size + (np.arange(size) < m % size)) * n), MPI.DOUBLE], local_A, root=0)

#     # Broadcast matrix B
#     B = comm.bcast(B, root=0)

#     # Perform local matrix multiplication
#     local_C = np.dot(local_A, B)

#     # Gather results (using Gatherv for uneven distribution)
#     counts = (m // size + (np.arange(size) < m % size)) * p
#     displacements = np.cumsum(counts) - counts
#     C = np.empty((m, p))
#     comm.Gatherv(local_C, [C, (counts, displacements), MPI.DOUBLE], root=0)

#     return C

if __name__ == "__main__":
    m = 500  # Rows of matrix A
    n = 500  # Columns of matrix A / Rows of matrix B
    p = 500  # Columns of matrix B

    if rank == 0:
        # Generate matrices A and B
        A = np.random.rand(m, n)
        B = np.random.rand(n, p)

        # Serial execution
        start_time = time.time()
        C_serial = serial_matrix_mult(A, B)
        serial_time = time.time() - start_time
        print(f"Serial execution time: {serial_time:.4f} seconds")

        # Distributed execution
        start_time = time.time()
        C_dist = distributed_matrix_mult(A, B, m, n, p)
        dist_time = time.time() - start_time
        print(f"Distributed execution time: {dist_time:.4f} seconds")

        # Calculate speedup and efficiency
        speedup = serial_time / dist_time
        efficiency = speedup / size
        print(f"Speedup: {speedup:.4f}")
        print(f"Efficiency: {efficiency:.4f}")

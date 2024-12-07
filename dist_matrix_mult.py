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

    # Distribute matrices A (split by rows) and B (split by columns)
    local_A = distribute_matrix(A, comm, axis=0)  # Split A by rows
    local_B = distribute_matrix(B.T, comm, axis=0).T  # Split B transpose and transpose back to column layout

    # Ensure the shapes align for the dot product
    local_C = np.dot(local_A, local_B)

    # Gather results from all processes
    C = None
    if rank == 0:
        C = np.empty((m, p))  # Create an empty C matrix on rank 0
    comm.Gather(local_C, C, root=0)  # Gather into the C matrix
    
    return C  # Return the complete C matrix from all processes


    # if rank == 0:
    #     # Concatenate the gathered results along axis 0 (rows)
    #     C = np.concatenate(gathered_results, axis=0)
    #     return C
    # else:
    #     return None


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

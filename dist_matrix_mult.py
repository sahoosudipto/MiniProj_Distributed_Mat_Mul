from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def serial_matrix_mult(A, B):
    """Perform serial matrix multiplication."""
    return np.dot(A, B)


def distribute_matrix(matrix, axis=0, root=0):
    """Distributes a matrix across processes along the specified axis."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        # Split the matrix into chunks along the specified axis
        chunks = np.array_split(matrix, size, axis=axis)
    else:
        chunks = None

    # Scatter the chunks across all processes
    local_chunk = comm.scatter(chunks, root=root)
    return local_chunk


def distributed_matrix_mult(A, B, m, n, p):
    """Performs distributed matrix multiplication using MPI."""
    # Distribute matrix A by rows and matrix B entirely to all processes
    local_A = distribute_matrix(A, axis=0)
    local_B = comm.bcast(B, root=0)  # Broadcast matrix B to all processes

    # Perform local computation of the chunk
    local_C = np.dot(local_A, local_B)

    # Gather the local results into the global matrix
    gathered_C = comm.gather(local_C, root=0)

    if rank == 0:
        # Combine all gathered chunks into the final result
        C = np.vstack(gathered_C)
        return C
    else:
        return None


# if __name__ == "__main__":
#     m = 500  # Rows of matrix A
#     n = 500  # Columns of matrix A / Rows of matrix B
#     p = 500  # Columns of matrix B

#     if rank == 0:
#         # Generate matrices A and B
#         A = np.random.rand(m, n)
#         B = np.random.rand(n, p)

#         # Serial execution
#         start_time = time.time()
#         C_serial = serial_matrix_mult(A, B)
#         serial_time = time.time() - start_time
#         print(f"Serial execution time: {serial_time:.4f} seconds")

#         # Distributed execution
#         start_time = time.time()
#         C_dist = distributed_matrix_mult(A, B, m, n, p)
#         dist_time = time.time() - start_time
#         print(f"Distributed execution time: {dist_time:.4f} seconds")

#         # Calculate speedup and efficiency
#         speedup = serial_time / dist_time
#         efficiency = speedup / size
#         print(f"Speedup: {speedup:.4f}")
#         print(f"Efficiency: {efficiency:.4f}")

import time
from dist_matrix_mult import serial_matrix_mult, distributed_matrix_mult
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_benchmark(m, n, p):
    # Generate matrices A and B on all processes
    A = np.random.rand(m, n)  
    B = np.random.rand(n, p)

    if rank == 0:
        # Serial execution
        start_time = time.time()
        serial_matrix_mult(A, B)
        serial_time = time.time() - start_time
    else:
        serial_time = None

    # Distributed execution (all processes run this)
    start_time = time.time()
    distributed_matrix_mult(A, B, m, n, p)  # Make sure this function is correct
    dist_time = time.time() - start_time

    serial_time = comm.bcast(serial_time, root=0)

    return serial_time, dist_time

if __name__ == "__main__":
    # Matrix dimensions
    m = 2000  # Increased matrix size
    n = 2000
    p = 2000

    serial_time, dist_time = run_benchmark(m, n, p)

    if rank == 0:
        # Calculate speedup and efficiency
        speedup = serial_time / dist_time
        efficiency = speedup / size

        # Print results
        print(f"Matrix dimensions: {m}x{n}, {n}x{p}")
        print(f"Number of processes: {size}")
        print(f"Serial execution time: {serial_time:.4f} seconds")
        print(f"Distributed execution time: {dist_time:.4f} seconds")
        print(f"Speedup: {speedup:.4f}")
        print(f"Efficiency: {efficiency:.4f}")

"""Test avec mesure de performance"""
from mpi4py import MPI
import numpy as np
import time

def bucket_sort_parallel_with_test():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n = 100000000
    
    if rank == 0:
        data = np.random.random(n)
        start_seq = time.time()
        sequential_sorted = np.sort(data.copy())
        time_seq = time.time() - start_seq
    else:
        data = None
        time_seq = None
    
    start_parallel = time.time()
    
    bucket_min = rank / size
    bucket_max = (rank + 1) / size
    
    data = comm.bcast(data, root=0)
    
    local_bucket = data[(data >= bucket_min) & (data < bucket_max)]
    if rank == size - 1:
        local_bucket = data[(data >= bucket_min) & (data <= bucket_max)]
    
    local_sorted = np.sort(local_bucket)
    all_sorted_buckets = comm.gather(local_sorted, root=0)
    
    time_parallel = time.time() - start_parallel
    
    if rank == 0:
        final_sorted = np.concatenate(all_sorted_buckets)
        is_sorted = np.all(final_sorted[:-1] <= final_sorted[1:])
        
        print(f"n={n}, processus={size}")
        print(f"Sequentiel: {time_seq:.4f}s")
        print(f"Parallele: {time_parallel:.4f}s")
        print(f"Speedup: {time_seq/time_parallel:.2f}x")
        print(f"Trie: {is_sorted}")
        
        return final_sorted
    
    return None


if __name__ == "__main__":
    bucket_sort_parallel_with_test()

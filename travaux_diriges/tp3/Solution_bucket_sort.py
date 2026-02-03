"""Bucket Sort parallele avec MPI"""
from mpi4py import MPI
import numpy as np

def bucket_sort_parallel():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n = 1000
    
    if rank == 0:
        data = np.random.random(n)
    else:
        data = None
    
    bucket_min = rank / size
    bucket_max = (rank + 1) / size
    
    data = comm.bcast(data, root=0)
    
    local_bucket = data[(data >= bucket_min) & (data < bucket_max)]
    if rank == size - 1:
        local_bucket = data[(data >= bucket_min) & (data <= bucket_max)]
    
    local_sorted = np.sort(local_bucket)
    all_sorted_buckets = comm.gather(local_sorted, root=0)
    
    if rank == 0:
        final_sorted = np.concatenate(all_sorted_buckets)
        is_sorted = np.all(final_sorted[:-1] <= final_sorted[1:])
        print(f"n={n}, trie={is_sorted}")
        return final_sorted
    
    return None


if __name__ == "__main__":
    bucket_sort_parallel()

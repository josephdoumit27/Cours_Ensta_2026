"""Bucket Sort avec entiers"""
from mpi4py import MPI
import numpy as np

def bucket_sort_integers():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n = 1000
    min_val = 0
    max_val = 1000
    
    if rank == 0:
        data = np.random.randint(min_val, max_val + 1, n)
        
        range_size = max_val - min_val + 1
        bucket_size = range_size / size
        
        buckets = [[] for _ in range(size)]
        for value in data:
            bucket_idx = int((value - min_val) / bucket_size)
            if bucket_idx >= size:
                bucket_idx = size - 1
            buckets[bucket_idx].append(value)
        
        buckets = [np.array(b, dtype=np.int32) for b in buckets]
    else:
        buckets = None
    
    local_bucket = comm.scatter(buckets, root=0)
    local_sorted = np.sort(local_bucket)
    all_sorted_buckets = comm.gather(local_sorted, root=0)
    
    if rank == 0:
        final_sorted = np.concatenate(all_sorted_buckets)
        is_sorted = np.all(final_sorted[:-1] <= final_sorted[1:])
        print(f"n={n}, trie={is_sorted}")
        return final_sorted
    
    return None

if __name__ == "__main__":
    bucket_sort_integers()

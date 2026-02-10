"""

Joseph DOUMIT BADER TARABAY

Bucket Sort Parallele avec MPI
TD n°3 - Programmation Parallele

"""

from mpi4py import MPI
import numpy as np
import time


def bucket_sort_parallel(n=1000000):
    
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Demarrage du bucket sort avec {n} elements sur {size} processus")
        data = np.random.random(n)
        start_time = time.time()
    else:
        data = None
    
    data = comm.bcast(data, root=0)
    
    bucket_min = rank / size
    bucket_max = (rank + 1) / size
    
    if rank == size - 1:
        local_bucket = data[(data >= bucket_min) & (data <= bucket_max)]
    else:
        local_bucket = data[(data >= bucket_min) & (data < bucket_max)]
    
    local_sorted = np.sort(local_bucket)
    
    all_buckets = comm.gather(local_sorted, root=0)
    
    if rank == 0:
        result = np.concatenate(all_buckets)
        elapsed = time.time() - start_time
        is_sorted = np.all(result[:-1] <= result[1:])
        
        print(f"Temps d'execution: {elapsed:.4f}s")
        print(f"RResultat correct: {is_sorted}")
        
        return result
    
    return None


def test_performance():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000]
    results = []
    
    for n in sizes:
        # Mesure du temps parallèle
        start_parallel = time.time()
        result_parallel = bucket_sort_parallel(n)
        time_parallel = time.time() - start_parallel
        
        if rank == 0:
            data_seq = np.random.random(n)
            start_seq = time.time()
            result_seq = np.sort(data_seq)
            time_seq = time.time() - start_seq
            
            speedup = time_seq / time_parallel
            results.append((n, time_seq, time_parallel, speedup))
    
    # Affichage des tableaux récapitulatifs
    if rank == 0:
        print("\n" + "="*60)
        print("TEMPS D'EXECUTION SEQUENTIEL")
        print("="*60)
        print(f"\n{'Nb donnees':>15} | {'Temps seq':>12}")
        print("-"*32)
        for n, time_seq, _, _ in results:
            print(f"{n:>15,} | {time_seq:>10.4f}s")
        print("="*60)
        
        print("\n" + "="*60)
        print(f"RESULTATS - {size} PROCESSUS")
        print("="*60)
        print(f"\n{'Nb donnees':>15} | {'Speedup':>10}")
        print("-"*30)
        for n, _, _, speedup in results:
            print(f"{n:>15,} | {speedup:>9.2f}x")
        print("="*60)


if __name__ == "__main__":
    test_performance()

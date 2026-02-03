"""Test et visualisation des performances du bucket sort"""
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def run_bucket_sort(n, num_proc):
    """Execute bucket sort avec MPI et retourne le temps"""
    script = f"""
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = {n}

if rank == 0:
    data = np.random.random(n)
else:
    data = None

start = time.time()

bucket_min = rank / size
bucket_max = (rank + 1) / size

data = comm.bcast(data, root=0)

local_bucket = data[(data >= bucket_min) & (data < bucket_max)]
if rank == size - 1:
    local_bucket = data[(data >= bucket_min) & (data <= bucket_max)]

local_sorted = np.sort(local_bucket)
all_sorted_buckets = comm.gather(local_sorted, root=0)

elapsed = time.time() - start

if rank == 0:
    print(elapsed)
"""
    
    with open('_temp_bucket_sort.py', 'w') as f:
        f.write(script)
    
    try:
        result = subprocess.run(
            ['mpiexec', '-n', str(num_proc), 'python', '_temp_bucket_sort.py'],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0])
        else:
            return None
    except:
        return None
    finally:
        if os.path.exists('_temp_bucket_sort.py'):
            os.remove('_temp_bucket_sort.py')

def sequential_sort(n):
    """Tri sequentiel pour reference"""
    data = np.random.random(n)
    start = time.time()
    np.sort(data)
    return time.time() - start

if __name__ == "__main__":
    sizes = [100, 1000, 10000, 100000, 1000000, 10000000]
    processes = [1, 2, 4, 8]
    
    results = {p: [] for p in processes}
    
    print("Test des performances du bucket sort parallele")
    print("=" * 60)
    
    for n in sizes:
        print(f"\nTaille n = {n}")
        
        for num_proc in processes:
            if num_proc == 1:
                t = sequential_sort(n)
                print(f"  Sequentiel: {t:.4f}s")
            else:
                t = run_bucket_sort(n, num_proc)
                if t:
                    print(f"  {num_proc} proc: {t:.4f}s")
                else:
                    t = None
                    print(f"  {num_proc} proc: ERREUR")
            
            results[num_proc].append(t)
    
    # Tracer les resultats
    plt.figure(figsize=(12, 6))
    
    # Graphique 1: Temps en fonction de la taille
    plt.subplot(1, 2, 1)
    for num_proc in processes:
        valid_times = [(s, t) for s, t in zip(sizes, results[num_proc]) if t is not None]
        if valid_times:
            x, y = zip(*valid_times)
            label = 'Sequentiel' if num_proc == 1 else f'{num_proc} processus'
            plt.plot(x, y, 'o-', label=label, linewidth=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Taille du tableau')
    plt.ylabel('Temps (s)')
    plt.title('Temps d\'execution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graphique 2: Speedup
    plt.subplot(1, 2, 2)
    seq_times = results[1]
    for num_proc in [2, 4, 8]:
        speedups = []
        valid_sizes = []
        for i, (t_seq, t_par) in enumerate(zip(seq_times, results[num_proc])):
            if t_seq and t_par and t_par > 0:
                speedups.append(t_seq / t_par)
                valid_sizes.append(sizes[i])
        if speedups:
            plt.plot(valid_sizes, speedups, 'o-', label=f'{num_proc} processus', linewidth=2)
    
    plt.plot(sizes, [1]*len(sizes), 'k--', alpha=0.3, label='Lineaire ideal')
    plt.xscale('log')
    plt.xlabel('Taille du tableau')
    plt.ylabel('Speedup')
    plt.title('Speedup par rapport au sequentiel')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bucket_sort_performance.png', dpi=150)
    print(f"\n\nGraphique sauvegarde: bucket_sort_performance.png")
    plt.show()

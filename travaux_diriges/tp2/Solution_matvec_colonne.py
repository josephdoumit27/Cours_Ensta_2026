import numpy as np
from mpi4py import MPI
from time import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

dim = 120

if dim % nbp != 0:
    if rank == 0:
        print(f"Erreur: La dimension {dim} n'est pas divisible par le nombre de processus {nbp}")
    exit(1)

Nloc = dim // nbp

start_col = rank * Nloc
end_col = (rank + 1) * Nloc

A_local = np.array([[(i+j) % dim + 1. for i in range(start_col, end_col)] for j in range(dim)])

if rank == 0:
    print(f"Dimension de la matrice: {dim}x{dim}")
    print(f"Nombre de processus: {nbp}")
    print(f"Nombre de colonnes par processus (Nloc): {Nloc}")

u = np.array([i+1. for i in range(dim)])

u_local = u[start_col:end_col]

deb = time()
v_local = A_local.dot(u_local)
fin = time()

if rank == 0:
    print(f"Temps de calcul local: {fin-deb}")

v = np.zeros(dim, dtype=np.double)
comm.Allreduce(v_local, v, op=MPI.SUM)

if rank == 0:
    print(f"\nVérification sur le processus 0:")
    print(f"u = {u}")
    print(f"v = {v}")
    
    A_complete = np.array([[(i+j) % dim + 1. for i in range(dim)] for j in range(dim)])
    v_sequential = A_complete.dot(u)
    diff = np.linalg.norm(v - v_sequential)
    print(f"\nDifférence avec le calcul séquentiel: {diff}")
    if diff < 1e-10:
        print("[OK] Résultat correct!")
    else:
        print("[ERREUR] Erreur dans le calcul!")

import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height

local_lines = []
for y in range(height):
    if y % nbp == rank:
        local_lines.append(y)

local_height = len(local_lines)
local_convergence = np.empty((width, local_height), dtype=np.double)

deb = time()
for idx, y in enumerate(local_lines):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        local_convergence[x, idx] = mandelbrot_set.convergence(c, smooth=True)
fin = time()

if rank == 0:
    print(f"Temps du calcul de l'ensemble de Mandelbrot (avec {nbp} processus, interleaving) : {fin-deb}")

if rank == 0:
    convergence = np.empty((width, height), dtype=np.double)
    for idx, y in enumerate(local_lines):
        convergence[:, y] = local_convergence[:, idx]
    
    for p in range(1, nbp):
        p_lines = [y for y in range(height) if y % nbp == p]
        p_height = len(p_lines)
        
        recv_buffer = np.empty((width, p_height), dtype=np.double)
        comm.Recv(recv_buffer, source=p, tag=0)
        
        for idx, y in enumerate(p_lines):
            convergence[:, y] = recv_buffer[:, idx]
    
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin-deb}")
    image.save("mandelbrot_statique.png")
    print("Image sauvegardée sous 'mandelbrot_statique.png'")
else:
    comm.Send(local_convergence, dest=0, tag=0)

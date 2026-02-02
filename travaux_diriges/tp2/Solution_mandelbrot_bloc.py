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

lines_per_proc = height // nbp
remainder = height % nbp
if rank < remainder:
    local_height = lines_per_proc + 1
    start_y = rank * local_height
else:
    local_height = lines_per_proc
    start_y = rank * lines_per_proc + remainder

end_y = start_y + local_height

local_convergence = np.empty((width, local_height), dtype=np.double)

deb = time()
for idx, y in enumerate(range(start_y, end_y)):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        local_convergence[x, idx] = mandelbrot_set.convergence(c, smooth=True)
fin = time()

if rank == 0:
    print(f"Temps du calcul de l'ensemble de Mandelbrot (avec {nbp} processus) : {fin-deb}")

if rank == 0:
    convergence = np.empty((width, height), dtype=np.double)
    convergence[:, start_y:end_y] = local_convergence
    
    for p in range(1, nbp):
        if p < remainder:
            p_height = lines_per_proc + 1
            p_start = p * p_height
        else:
            p_height = lines_per_proc
            p_start = p * lines_per_proc + remainder
        
        p_end = p_start + p_height
        recv_buffer = np.empty((width, p_height), dtype=np.double)
        comm.Recv(recv_buffer, source=p, tag=0)
        convergence[:, p_start:p_end] = recv_buffer
    
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin-deb}")
    image.save("mandelbrot_bloc.png")
    print("Image sauvegardée sous 'mandelbrot_bloc.png'")
else:
    comm.Send(local_convergence, dest=0, tag=0)

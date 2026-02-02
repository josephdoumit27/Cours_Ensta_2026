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

TAG_WORK = 1
TAG_RESULT = 2
TAG_STOP = 3

deb_total = time()

if rank == 0:
    convergence = np.empty((width, height), dtype=np.double)
    next_line = 0
    finished_workers = 0
    
    for worker in range(1, nbp):
        if next_line < height:
            comm.send(next_line, dest=worker, tag=TAG_WORK)
            next_line += 1
        else:
            comm.send(-1, dest=worker, tag=TAG_STOP)
            finished_workers += 1
    
    while finished_workers < nbp - 1:
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
        worker = status.Get_source()
        
        y_line = result['line']
        line_data = result['data']
        convergence[:, y_line] = line_data
        
        if next_line < height:
            comm.send(next_line, dest=worker, tag=TAG_WORK)
            next_line += 1
        else:
            comm.send(-1, dest=worker, tag=TAG_STOP)
            finished_workers += 1
    
    fin_total = time()
    print(f"Temps du calcul de l'ensemble de Mandelbrot (maître-esclave avec {nbp} processus) : {fin_total-deb_total}")
    
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin-deb}")
    image.save("mandelbrot_maitre_esclave.png")
    print("Image sauvegardée sous 'mandelbrot_maitre_esclave.png'")

else:
    while True:
        y_line = comm.recv(source=0, tag=MPI.ANY_TAG, status=MPI.Status())
        
        if y_line == -1:
            break
        
        line_data = np.empty(width, dtype=np.double)
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY * y_line)
            line_data[x] = mandelbrot_set.convergence(c, smooth=True)
        
        result = {'line': y_line, 'data': line_data}
        comm.send(result, dest=0, tag=TAG_RESULT)

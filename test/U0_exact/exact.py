import numpy as np

Lx = 4 * 3
Ly = 4
N = Lx * Ly

bcx = False
bcy = True

H = np.zeros((N, N), dtype=np.float64)
for i in range(N):
    ix = i % Lx
    iy = i // Lx
    ixp = iy * Lx + (ix+1) % Lx
    iyp = ((iy+1) % Ly) * Lx + ix
    if bcx or ix+1 < Lx:
        H[i, ixp] += -1.0
    if bcy or iy+1 < Ly:
        H[i, iyp] += -1.0
H = H + H.T

eig_single = np.linalg.eigvals(H)
eig_single.sort()
#print(eig_single)
Nf = Lx*Ly//2
print(2*sum(eig_single[:Nf]))
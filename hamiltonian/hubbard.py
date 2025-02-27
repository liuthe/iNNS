import netket as nk
import netket.experimental as nkx

from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc

def Hubbard(t: float, U: float, 
            dimensions: list[int, int], 
            boundaries: list[bool, bool], 
            filling: tuple[int, int]) -> tuple:
    """The function generates the hilber space and the Hamiltonian for the Hubbard model

    Args:
        t (float): hopping strength
        U (float): interaction strength
        dimensions (list[int, int]): size of the lattice
        boundaries (list[bool, bool]): boundary conditions, True for periodic, False for open
        filling (tuple[int, int]): number of up and down electrons

    Returns:
        tuple: hilbert space, Hamiltonian, graph
    """
    basis_vectors = [[1,0],[0,1]]
    #Define the graph 
    graph = nk.graph.Lattice(basis_vectors=basis_vectors, extent = dimensions, pbc = boundaries)
    #Define the Hilbert space
    N = graph.n_nodes
    hi = nk.hilbert.SpinOrbitalFermions(N, s=1/2, n_fermions_per_spin=filling)
    #Define the Hamiltonian
    H = 0.0
    for (i, j) in graph.edges():
        H -= t * (cdag(hi,i,1) * c(hi,j,1) + cdag(hi,j,1) * c(hi,i,1))
        H -= t * (cdag(hi,i,-1) * c(hi,j,-1) + cdag(hi,j,-1) * c(hi,i,-1))
    for i in graph.nodes():
        H += U * nc(hi,i,1) * nc(hi,i,-1)
    return hi, H, graph

def Hubbard_extend(t: float, U: float, 
            dimensions: list[int, int], 
            boundaries: list[bool, bool], 
            filling: tuple[int, int]):
    # Define the graph
    basis_vectors = [[1,0],[0,1]]
    graph = nk.graph.Lattice(basis_vectors=basis_vectors, extent = dimensions, pbc = boundaries)
    exchange_graph = nk.graph.disjoint_union(graph, graph, graph, graph, graph, graph)
    # Define the Hilbert space
    N = graph.n_nodes
    hi_help = nk.hilbert.SpinOrbitalFermions(N, s=5/2, n_fermions_per_spin=(filling[0],filling[1],filling[0],filling[1],filling[0],filling[1]))
    H_help = 0.0
    for (i, j) in graph.edges():
        for sz in [-5, -3, -1, 1, 3, 5]:
            H_help -= t * (cdag(hi_help,i,sz) * c(hi_help,j,sz) + cdag(hi_help,j,sz) * c(hi_help,i,sz))
    for i in graph.nodes():
        H_help += U * nc(hi_help,i,-5) * nc(hi_help,i,-3)
        H_help += U * nc(hi_help,i,-1) * nc(hi_help,i,1)
        H_help += U * nc(hi_help,i,3) * nc(hi_help,i,5)
    # We copy H_help and add some hopping terms between the different spins
    H_full = H_help.copy()
    Lx, Ly = dimensions
    for ny in range(Ly):
        i = (Lx-1)*Ly + ny
        j = ny
        H_full -= t * (cdag(hi_help,i,-5) * c(hi_help,j,-1) + cdag(hi_help,j,-1) * c(hi_help,i,-5))
        H_full -= t * (cdag(hi_help,i,-3) * c(hi_help,j, 1) + cdag(hi_help,j, 1) * c(hi_help,i,-3))
        H_full -= t * (cdag(hi_help,i,-1) * c(hi_help,j, 3) + cdag(hi_help,j, 3) * c(hi_help,i,-1))
        H_full -= t * (cdag(hi_help,i, 1) * c(hi_help,j, 5) + cdag(hi_help,j, 5) * c(hi_help,i, 1))
    return hi_help, H_full, exchange_graph
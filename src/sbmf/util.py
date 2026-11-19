"""
util.py -- numerical helpers used by the BMF models and df_util / bae_util.

Trimmed from the original research util.py to the 20 definitions the package
actually uses; the rest is in the git history.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.special as spc
import scipy.sparse as sprs
from scipy.optimize import linear_sum_assignment
from scipy.optimize import linprog as lp
import networkx as nx


@dataclass
class LexOrder:
    """
    Lexicographic order of symmetric pairs, (i,j), j < i
    
    j = __|_0_1_2_3_
    i = 0 | - - - -
        1 | 0 - - -
        2 | 1 2 - -
        3 | 3 4 5 -
        
    """
    diag: bool = False
    
    def __call__(self,i,j):
        """
        (i,j) -> n
        n = i*(i-1)/2 + j
        """

        i1 = np.where(i>=j, i, j)
        i2 = np.where(i<j, i, j)
        if self.diag:
            i1 += 1

        n = i1*(i1-1)/2 + i2
        if not self.diag:
            return np.where(i==j, -1, n).astype(int)
        else:
            return n

    def inv(self, n):
        """
        n -> (i,j)
        i = floor( (1 + sqrt(1 + 8n))/2 )
        j = n - (i*(i-1)/2)
        """
        i = np.floor((1 - 2*self.diag + np.sqrt(1+8*n))/2).astype(int) 
        j = (n - (i+self.diag)*(i-1+self.diag)//2).astype(int)
        return i, j



def center(K, n=None):
    K_1 = K[...,:n].mean(-1,keepdims=True)
    K_2 = K[...,:n,:].mean(-2,keepdims=True)
    K_ = K[...,:n,:n].mean((-1,-2),keepdims=True)
    return K - K_1 - K_2 + K_



def center_kernel(K):
    return K - K.mean(-2,keepdims=True) - K.mean(-1,keepdims=True) + K.mean((-1,-2),keepdims=True)



def mat(x, incl_diag=False):
    
    inds = LexOrder()

    aye, jay = inds.inv(np.arange(x.shape[-1]))
    
    N = int(aye.max())+1
    
    A = np.zeros((*x.shape[:-1], N, N))
    
    A[...,aye,jay] = x
    A[...,jay,aye] = x
    
    return A + incl_diag*np.expand_dims(np.eye(N), tuple(range(np.ndim(x)-1)))



def vec(A, incl_diag=False):

    inds = LexOrder()

    N = A.shape[-1]
    trind = inds.inv(np.arange(0.5*N*(N-1), dtype=int))
    
    return A[...,trind[0], trind[1]]



def outers(V):
    """
    Compute outer products of columns of V
    """
    return np.einsum('...ik,...jk->...kij', V, V)



def spouters(i, j, v=None, incl_diag=True, shape=None):
    """
    Outer product of columns of a sparse (i,j,v) matrix ... 
    since scipy.sparse doesn't support einsum

    values v are optional, default assumes they are all 1

    the row indices of the output are in the lexicographic order, 
    unless incl_diag is true
    """

    if shape is None:
        shape = (np.max(i)+1, np.max(j)+1)

    idx = np.argsort(j) # I think this is redundant?
    j = j[idx]
    i = i[idx]

    inds = LexOrder()

    junq, c = np.unique(j, return_counts=True)
    c2 = spc.binom(c,2).astype(int)

    ## numpy shenanigans
    pairs = np.arange(np.sum(c2))
    pairs -= np.repeat(np.cumsum(c2)-c2, c2)
    aye, jay = inds.inv(pairs) # indices of item pairs

    shift = np.repeat(np.cumsum(c)-c, c2)
    aye += shift 
    jay += shift

    j_kron = np.repeat(junq, c2) # features 
    i_kron = inds(i[aye], i[jay]) # pairs of items

    if incl_diag:
        ## append diagonal
        j_kron = np.append(j_kron, j)
        i_kron = np.append(i_kron, i + int(spc.binom(shape[0], 2)) )

    if v is not None:
        v_kron = v[aye]*v[jay]
    else:
        v_kron = np.ones(len(i_kron))

    return i_kron.astype(int), j_kron.astype(int), v_kron



def bipartite_incidence(adj):
    """
    Convert bipartite adjacency matrix to an incidence matrix
    """

    P, N = adj.shape

    inds = LexOrder()

    left = np.arange(P) 
    right = np.arange(P, P+N)
    aye, jay = np.where(adj)
    
    ne = len(aye)
    edges = inds(left[aye], right[jay])

    cols = np.repeat(np.arange(len(edges)), 2)
    rows = np.empty((2*ne,), dtype=int)
    rows[0::2] = left[aye]
    rows[1::2] = right[jay]

    return sprs.coo_matrix((np.ones(2*ne), (rows, cols))), edges



def unbalanced_assignment(cost, adj=None, one_sided=False):
    """
    cost is shape (P, N) with P <= N
    adj is the same shape and binary (default all ones)

    returns row_indices, col_indices

    Solves the assignment problem, generalizing the classic
    problem to the case of unbalanced sets. Unlike standard
    modifications or reductions, which find a one-sided 
    perfect matching (bidirectionally one-to-one), this ensures
    that every item participates (bidirectionally onto). So, an 
    item in the smaller set can connect with multiple in the larger,
    but an item in the larger can still only connect with one.

    """

    inds = LexOrder()

    ## left set is at most smaller than right set
    P, N = cost.shape

    if ((P == N) or one_sided) and (adj is None): 
        # scipy is certainly faster than me
        return linear_sum_assignment(cost)

    if N < P: 
        transpose = True
        C = cost.T
        P, N = C.shape
        if adj is not None:   # FIX: a supplied adjacency was left un-transposed
            adj = adj.T
    else:
        transpose = False
        C = cost

    ## bipartite incidence matrix
    if adj is None:
        adj = np.ones((P,N))
    
    Inc, edges = bipartite_incidence(adj)
    ne = len(edges)
    aye, jay = np.where(adj)

    ## modify for slack variables
    I = sprs.coo_matrix( (-np.ones(P), (np.arange(P), np.arange(P)) ), shape=(P+N,P))
    A = sprs.bmat([[Inc, I]], format='csr')
    b = np.ones(P+N)

    ## solve LP
    c = np.append(C[aye,jay], np.zeros(P))
    prog = lp(c, A_eq=A, b_eq=b)
    
    r, l = inds.inv(edges[prog.x[:ne] > 1e-6])

    if transpose:
        return r-P, l
    else:
        return l, r-P



def topsort(graph):
    """
    Returns a the topological sort of graph, a partial ordering. 
    Graph is represented by a dictionary giving the neighbors 
    of each node (i.e. `graph[i]` are the neighbors of `i`)
    Must be a DAG, otherwise returns an error
    """

    nodes = []
    seen = set()        # fully processed nodes
    path = []           # nodes on the current DFS path (for cycle detection)

    def visit(node):

        if node in seen:
            return
        if node in path:
            raise Exception(
                'Check your graph, it has a cycle!'
                f'Specifically, this one: {path+[node]}'
                )

        path.append(node)

        for neighbor in graph[node]:
            visit(neighbor)

        path.pop()
        seen.add(node)
        nodes.insert(0, node)

    for node in graph.keys():
        visit(node)

    return nodes



def depth(graph):
    """
    Depth of each node in the transitive reduction of `graph`
    """

    depths = np.zeros(len(graph))

    for node in topsort(graph):
        for neighbor in graph[node]:
            depths[neighbor] = max(depths[neighbor], depths[node]+1)

    return depths



def generate_parametric_no_chain_tree(N, rho=1, alpha=1.0):
    """
    N: Number of nodes
    rho: (0, 1] Control for number of internal nodes M.
    alpha: (>0) Control for branching concentration. 
           High alpha = uniform branching; Low alpha = hub-heavy.
           
    Modified from Gemini
    """
    if N < 3:
        # A tree with N < 3 cannot avoid degree 1 unless it's just a single node.
        return nx.path_graph(N, create_using=nx.DiGraph())

    # 1. Determine M (Number of internal nodes)
    # We use a Binomial draw so there is still some variance, centered on rho
    M_max = (N - 1) // 2
    M = np.random.binomial(M_max - 1, rho) + 1 # Ensure at least 1 hub
    
    # 2. Determine Degrees using Dirichlet-Multinomial
    # Surplus edges beyond the minimum 2 per internal node
    surplus_count = (N - 1) - (2 * M)
    
    # Generate weights for the M nodes
    weights = np.random.dirichlet([alpha] * M)
    
    # Distribute surplus edges based on weights
    surplus_dist = np.random.multinomial(surplus_count, weights)
    internal_degrees = 2 + surplus_dist
    
    # 3. Assemble the full degree sequence
    degrees = np.zeros(N, dtype=int)
    degrees[:M] = internal_degrees
    np.random.shuffle(degrees)
    
    # 4. Standard Cyclic Lemma + BFS Construction
    steps = degrees - 1
    prefix_sums = np.cumsum(steps)
    shift = (np.argmin(prefix_sums) + 1) % N
    valid_degrees = np.roll(degrees, -shift)
    
    nodes = np.random.permutation(N)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    
    queue = [nodes[0]]
    child_idx = 1
    for d in valid_degrees:
        if not queue: break
        parent = queue.pop(0)
        for _ in range(d):
            if child_idx < N:
                child = nodes[child_idx]
                G.add_edge(parent, child)
                queue.append(child)
                child_idx += 1
                
    return G



def generate_signed_block_matrix(G, tau=0, min_size=2):
    """
    Converts a directed tree G into the adjacency matrix of a signed graph.
    - Each node i in G becomes a clique of size max(1, out_degree(i)).
    - Internal clique edges are -1.
    - Directed edges (i -> j) become +1 edges from one node in C_i to all in C_j.
    
    Modified from Gemini
    """
    # 1. Calculate clique sizes and start indices
    nodes = list(G.nodes())
    out_degrees = {u: G.out_degree(u) for u in nodes}
    
    clique_sizes = {u: max(min_size, out_degrees[u]+np.random.poisson(tau)) for u in nodes}
    
    # Map tree node ID to the start index in the adjacency matrix
    start_indices = {}
    current_idx = 0
    for u in nodes:
        start_indices[u] = current_idx
        current_idx += clique_sizes[u]
    
    N_total = current_idx
    adj = np.zeros((N_total, N_total), dtype=int)
    
    # 2. Fill cliques with negative edges
    for u in nodes:
        size = clique_sizes[u]
        start = start_indices[u]
        if size > 1:
            # Create a clique of -1s
            for i in range(size):
                for j in range(i + 1, size):
                    adj[start + i, start + j] = -1
                    adj[start + j, start + i] = -1
                    
    # 3. Fill directed edges with positive connections
    for u in nodes:
        start_u = start_indices[u]
        # Get successors (children)
        successors = list(G.successors(u))
        
        for k, v in enumerate(successors):
            # Anchor node is the k-th node in clique C_u
            anchor_node_idx = start_u + k
            
            # Target is the entire clique C_v
            start_v = start_indices[v]
            size_v = clique_sizes[v]
            
            for m in range(size_v):
                target_node_idx = start_v + m
                adj[anchor_node_idx, target_node_idx] = 1
                adj[target_node_idx, anchor_node_idx] = 1
                
    node_map = {k: np.arange(start_indices[k], start_indices[k]+clique_sizes[k]) for k in nodes}

    return adj, node_map



def F2(n, sym=False):
    """
    All elements of F2(n), i.e. all n-bit binary vectors
    if sym is True, removes the 1-x symmetry (i.e. returns 2^(n-1) vectors)
    """

    N = n-sym
    F = np.mod(np.arange(2**int(N))[:,None]//(2**np.arange(int(N))[None,:]),2)
    if sym:
        F = np.hstack([np.zeros((2**N,1)), F])

    return F



def lof(S):
    """
    Left-order-form of binary matrix S
    """
    return S[:,np.lexsort(-S[::-1])]



def find_clique(A, first=0):
    """
    Find a single maximal clique in A
    """

    N = len(A)

    clq = [first]

    for i in range(N):
        if (i not in clq) and np.all(A[clq,i]):
            clq.append(i)

    return clq



def norms(x,y,p=2):
    """Assume features are in first axis"""
    return la.norm(x[...,None] - y[...,None,:], p, axis=0)



def dot2dist(K):
    """
    Dot product to squared distances
    """

    n = K.shape[-1]
    norms = K[..., np.arange(n), np.arange(n)]

    return (np.abs(norms[...,None] + norms[...,None,:] - 2*K))



def qform(Q, x, y=None):
    """
    Quadratic component of the surface equation
    assumes features are in last dimension
    """
    
    if y is None:
        y = x


    Qx = np.einsum('ij,...j->...i', Q, x)
    yQx = (Qx*y).sum(-1, keepdims=True)

    return yQx



def group_sum(X, groups, axis=-1):
    """
    Take the group-wise sum of X along axis

    X is array
    groups is a 1-d array
    
    returns array of shape (X.shape[:axis:], len(unique(groups)) )
    """
    
    ## Organize info
    grps, inv = np.unique(groups, return_inverse=True)
    shp = list(X.shape)
    shp[axis] = len(grps)

    ## Prepare output
    Xavg = np.zeros(shp)
    Xavg = np.swapaxes(Xavg, 0, axis)

    ## Use magical function `at'
    np.add.at(Xavg, inv, np.swapaxes(X, 0, axis))

    return np.swapaxes(Xavg, 0, axis)


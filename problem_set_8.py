# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 08:11:51 2019
Problem Set 8
Problem 1: Count Triangles in a Graph Using Adjacency Matrix
Problem 2: Distance Between All Pairs of Vertices

"""

#Problem 1
#Imports
from numpy import dot
from numpy.linalg import matrix_power
import numpy as np
import networkx as nx


def adj_mat(g):
    """Returns a dense adjacency matrix for a graph g"""
    return np.array(nx.adjacency_matrix(g).todense(), dtype=np.float64)

def count_triangles(A):
    """Triangle in a graph is a 3-step paths from each vertex to itself.
        Its ordering on the vertices are in 1-1 correspondence with directed
        paths. Thus, the number of triangles in a graph is the trace of A^3 
        divded by 6 due counting the triangle twice in each direction for each
        vertex in the triangle."""
   
    A_3 = matrix_power(A, 3)
    A_trace = np.trace(A_3)
    num_triangles = A_trace /6
    return num_triangles

#Code Test
from nose.tools import ok_
ok_(0 == count_triangles(adj_mat(nx.petersen_graph())))
ok_(10 == count_triangles(adj_mat(nx.complete_graph(5))))
ok_(9 == count_triangles(adj_mat(nx.wheel_graph(10))))
ok_(1412 == count_triangles(adj_mat(nx.wheel_graph(1413))))


#Problem 2
def seidel_apsp(A):
    """Returns n x n matrix with Dij being the distance in G between i and j
    by computing the matrix B that is the 'square' of G and if it is equivalent
    to the adjacency matrix A of G, then it returns the matrix B. If not, 
    recursively find the B that is equivalent to A, then find a diagonal
    matrix Z that has degree of vertex i in G for each i, and return
    2*D - P. """
    
    A_2 = matrix_power(A, 2)
    
    B_1 = A == 1
    B_1 = B_1.astype(np.int)

    B_2 = A_2 >= 1
    B_2 = B_2.astype(np.int)
    B = B_1 + B_2
    
    B = B > 0 #B > 0 is computed for Aij =1 or (A^2)ij = 1
    B = B.astype(np.int)
    np.fill_diagonal(B, 0)

    if np.array_equal(A, B):
        return A
    
    else:
        # Recursively compute the distance between all pairs of 
        # vertices in the graph
        D = seidel_apsp(B)
       
        # Z is N * N diagonal matrix of the sum of the degrees of vertex i 
        # across the coloumns
        Z = np.diagflat(np.sum(A, axis = 1))
        
        DA = dot(D, A)
        DZ = dot(D, Z)
       
        P = DA < DZ
        P = P.astype(np.int)
        
        return np.subtract(np.multiply(2, D), P)

#Code Test
ok_(np.all(np.array(
        [[0, 1, 2, 2, 1, 1, 2, 2, 2, 2],
       [1, 0, 1, 2, 2, 2, 1, 2, 2, 2],
       [2, 1, 0, 1, 2, 2, 2, 1, 2, 2],
       [2, 2, 1, 0, 1, 2, 2, 2, 1, 2],
       [1, 2, 2, 1, 0, 2, 2, 2, 2, 1],
       [1, 2, 2, 2, 2, 0, 2, 1, 1, 2],
       [2, 1, 2, 2, 2, 2, 0, 2, 1, 1],
       [2, 2, 1, 2, 2, 1, 2, 0, 2, 1],
       [2, 2, 2, 1, 2, 1, 1, 2, 0, 2],
       [2, 2, 2, 2, 1, 2, 1, 1, 2, 0]]) 
    == seidel_apsp(adj_mat(nx.petersen_graph()))))

ok_(np.all(np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
       [1, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
       [1, 2, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
       [1, 2, 2, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2],
       [1, 2, 2, 2, 1, 0, 1, 2, 2, 2, 2, 2, 2],
       [1, 2, 2, 2, 2, 1, 0, 1, 2, 2, 2, 2, 2],
       [1, 2, 2, 2, 2, 2, 1, 0, 1, 2, 2, 2, 2],
       [1, 2, 2, 2, 2, 2, 2, 1, 0, 1, 2, 2, 2],
       [1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 1, 2, 2],
       [1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 1, 2],
       [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 1],
       [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0]])
    == seidel_apsp(adj_mat(nx.wheel_graph(13)))))

ok_(np.all(np.array([[0, 1, 2, 3, 4, 3, 2, 1],
       [1, 0, 1, 2, 3, 4, 3, 2],
       [2, 1, 0, 1, 2, 3, 4, 3],
       [3, 2, 1, 0, 1, 2, 3, 4],
       [4, 3, 2, 1, 0, 1, 2, 3],
       [3, 4, 3, 2, 1, 0, 1, 2],
       [2, 3, 4, 3, 2, 1, 0, 1],
       [1, 2, 3, 4, 3, 2, 1, 0]]) 
    == seidel_apsp(adj_mat(nx.cycle_graph(8)))))

seidel_apsp(adj_mat(nx.wheel_graph(1413))) #With many vertices


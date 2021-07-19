# distutils: language=c++
#Импорт необходимых билиотек
import Gpu_graph_thust as gp
from libcpp.vector cimport vector 
from libcpp.pair cimport pair
from libcpp.string cimport string

#Вызов С++ Функций
cdef extern from "graph.h":
    ctypedef long long unsigned int l_int_t
    ctypedef vector[pair[int,pair[l_int_t,l_int_t]]] l_vec_int_t
    ctypedef pair[vector[int],l_vec_int_t] matrix_edges_t;
    ctypedef pair[vector[int],matrix_edges_t] nodes_matrix_edges_t;
    
    vector[int] _Floid_Warshell "Floid_Warshell" (vector[int] matrix,l_int_t l)
    vector[int] _OpenMP_Floid_Warshell "OpenMP_Floid_Warshell" (vector[int] matrix,l_int_t l,int number_of_threads)
    l_vec_int_t _Prima "Prima" (vector[int] matrix,l_int_t l)
    l_vec_int_t _OpenMP_Prima "OpenMP_Prima" (vector[int] matrix,l_int_t l,int Number_of_threads)
    l_vec_int_t _Kruskal "Kruskal" (vector[vector[int]]Edges, vector[int]Nodes)
    l_vec_int_t _OpenMP_Kruskal "OpenMP_Kruskal" (vector[vector[int]]Edges, vector[int]Nodes)
    vector[int] _New_Nodes "New_Nodes" (l_int_t l)
    matrix_edges_t _New_Matrix_Edges "New_Matrix_Edges" (l_int_t l,int probability)
    nodes_matrix_edges_t _Import_Graph "Import_Graph" (string s)
    int _Print_NOT "Print_NOT"()
    vector[int] _Dijkstra "Dijkstra" (vector[int] matrix,l_int_t l,int start) 
    vector[int] _OpenMP_Dijkstra "OpenMP_Dijkstra" (vector[int] matrix,l_int_t l,int number_of_threads,int start) ;
                  
#Обертка С++ функций с помощью Python синтаксиса 
def Floid_Warshell(a,l):
    cdef vector[int] x=a
    return _Floid_Warshell(x,l)

def OpenMP_Floid_Warshell(a,l,t):
    cdef vector[int] x=a
    return _OpenMP_Floid_Warshell(x,l,t)

def Prima(a,l):
    cdef vector[int] x=a
    return _Prima(x,l)

def OpenMP_Prima(a,l,t):
    cdef vector[int] x=a
    return _OpenMP_Prima(x,l,t)

def Kruskal(e,n):
    cdef vector[vector[int]] x=e
    cdef vector[int] m=n
    return _Kruskal(x,m)

def OpenMP_Kruskal(e,n):
    cdef vector[vector[int]] x=e
    cdef vector[int] m=n
    return _OpenMP_Kruskal(x,m)

def New_Matrix_Edges(l,p):
    return _New_Matrix_Edges(l,p)

def New_Nodes(l):
    return _New_Nodes(l)

def Import_Graph(name):
    py_unicode_object=name
    cdef string name_string = <string> py_unicode_object.encode('utf-8')
    return _Import_Graph(name_string)

def Print_NOT():
    return _Print_NOT()

def Dijkstra(a,l,s):
    cdef vector[int] x=a
    return _Dijkstra(x,l,s)

def OpenMP_Dijkstra(a,l,n,s):
    cdef vector[int] x=a
    return _OpenMP_Dijkstra(x,l,n,s)
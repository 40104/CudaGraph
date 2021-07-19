//
// Created by One on 25.05.2021.
//

#ifndef UNTITLED12_GRAPH_H
#define UNTITLED12_GRAPH_H
// Испольемые библиотеки
#include <vector>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <string>
#include <string.h>
#include <fstream>
#include <algorithm>
using namespace std;

// Использование псевдонимов типов данных

typedef long long unsigned int l_int_t;
typedef vector<pair<int,pair<l_int_t,l_int_t>>> l_vec_int_t;
typedef pair<vector<int>,l_vec_int_t> matrix_edges_t;
typedef pair<vector<int>,matrix_edges_t> nodes_matrix_edges_t;

// Описание функций библиотеки
vector<int> Floid_Warshell (vector<int> matrix,l_int_t l);
vector<int> OpenMP_Floid_Warshell (vector<int> matrix,l_int_t l,int number_of_threads);
l_vec_int_t Prima(vector<int> matrix,l_int_t l);
l_vec_int_t OpenMP_Prima(vector<int> matrix,l_int_t l,int Number_of_threads);
l_vec_int_t Kruskal(vector<vector<int>>Edges, vector<int>Nodes);
l_vec_int_t OpenMP_Kruskal (vector<vector<int>>Edges, vector<int>Nodes);
vector<int> New_Nodes(l_int_t l);
matrix_edges_t New_Matrix_Edges(l_int_t l,int probability);
nodes_matrix_edges_t Import_Graph(string s);
void Print_NOT(int number_of_thread);
vector<int> Dijkstra(vector<int> matrix,l_int_t l,int start);
vector<int> OpenMP_Dijkstra(vector<int> matrix,l_int_t l,int number_of_threads,int start) ;

#endif //UNTITLED12_GRAPH_H

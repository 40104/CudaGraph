

#include "class.h"

//Алгоритм Флойда-Уоршелла однопток
vector<int> Floid_Warshell (vector<int> matrix,l_int_t l)
{
    for (l_int_t k = 0; k < l; k++)
    {
        for (l_int_t i = 0; i < l; i++)
        {
            for (l_int_t j = 0; j < l; j++)
            {
                matrix[i * l + j]=(matrix[i * l + j]<=matrix[i * l + k]+matrix[k * l + j]) ? matrix[i * l + j] : (matrix[i * l + k]+matrix[k * l + j]);
            }
        }
    }
    return matrix;
}

//Алгоритм Флойда-Уоршелла Многопоток
vector<int> OpenMP_Floid_Warshell (vector<int> matrix,l_int_t l,int number_of_threads)
{
    for (l_int_t k = 0; k < l; k++)
    {
        omp_set_dynamic(0);
        omp_set_num_threads(number_of_threads);
#pragma omp parallel for
        for (l_int_t i=0; i < l; i++)
        {
            for (l_int_t j = 0; j < l; j++)
            {
                matrix[i * l + j]=matrix[i * l + j]<=matrix[i * l + k]+matrix[k * l + j]?matrix[i * l + j]:matrix[i * l + k]+matrix[k * l + j];
            }
        }
    }
    return matrix;
}

//Алгоритм Прима однопток
l_vec_int_t Prima(vector<int> matrix,l_int_t l)
{
    vector <bool> visited(l, false);
    l_vec_int_t Tree;


    visited[0] = true;
    l_int_t counter = 1;
    l_int_t min;
    l_int_t i_min;
    l_int_t j_min;
    l_int_t INF = 99999;
    while (counter < l) {
        min = INF;
        i_min = INF;
        j_min = INF;
        for (l_int_t i = 0;i < l;i++) {
            if (visited[i]) {
                for (l_int_t j = 0;j < l;j++) {
                    if (!visited[j] && matrix[i * l + j] != INF && matrix[i * l + j] < min) {
                        min = matrix[i * l + j];
                        i_min = i;
                        j_min = j;
                    }
                }
            }
        }
        Tree.push_back(make_pair(matrix[i_min * l + j_min],make_pair(i_min,j_min)));
        visited[j_min] = true;
        counter++;
    }
    return Tree;
}

//Алгоритм Прима многопоток
l_vec_int_t OpenMP_Prima(vector<int> matrix,l_int_t l,int Number_of_threads)
{
    vector <bool> visited(l, false);
    l_vec_int_t Tree;

    visited[0] = true;
    l_int_t counter = 1, min,i_min,j_min,INF = 99999;

    while (counter < l) {
        min = INF;
        i_min = INF;
        j_min = INF;
        for (l_int_t i = 0;i < l;i++) {
            if (visited[i]) {
                omp_set_dynamic(0);
                omp_set_num_threads(Number_of_threads);
#pragma omp parallel
                {
                    l_int_t index_local=0;
                    l_int_t min_local=INF;

#pragma omp for
                    for (l_int_t j = 0; j < l; j++) {
                        if (!visited[j] && matrix[i * l + j] != INF && matrix[i * l + j] < min_local) {
                            min_local = matrix[i * l + j];
                            index_local = j;
                        }
                    }
#pragma omp critical
                    if (min_local < min) {
                        min = min_local;
                        i_min = i;
                        j_min = index_local;
                    }
                }
            }
        }
        Tree.push_back(make_pair(matrix[i_min * l + j_min],make_pair(i_min,j_min)));
        visited[j_min] = true;
        counter++;
    }
    return Tree;
}

//Алгоритм быстрой сортировки
void Quick_Sort(int* edges, l_int_t low, l_int_t high, int** nodes)
{
    l_int_t i = low;
    l_int_t j = high;
    l_int_t mid = edges[(i + j) / 2];

    while (i <= j)
    {
        while (edges[i] < mid)
            i++;
        while (edges[j] > mid)
            j--;
        if (i <= j)
        {
            swap(edges[i], edges[j]);
            swap(nodes[i], nodes[j]);
            i++;
            j--;
        }
    }
    if (j > low)
        Quick_Sort(edges, low, j, nodes);
    if (i < high)
        Quick_Sort(edges, i, high, nodes);
}

//Алгоритм Краскала однопток
l_vec_int_t Kruskal(vector<vector<int>>Edges, vector<int>Nodes)
{
    l_int_t l=Nodes.size();
    l_int_t c=Edges.size();

    int* edges;
    edges = new int[c];
    int** nodes;
    nodes = new int* [c];
    for (l_int_t i = 0;i < c;i++) { nodes[i] = new int[2]; }

    for (l_int_t i = 0;i < c;i++) {
        edges[i] = Edges[i][0];
        nodes[i][0] = Edges[i][1];
        nodes[i][1] = Edges[i][2];
    }

    l_vec_int_t Tree;
    l_int_t i_node, j_node, val;

    Quick_Sort(edges, 0, c - 1, nodes);

    for (l_int_t iterator = 0; iterator < c; iterator++) {
        i_node = nodes[iterator][0];
        j_node = nodes[iterator][1];
        val = edges[iterator];
        if (Nodes[i_node] != Nodes[j_node])
        {
            Tree.push_back(make_pair(val,make_pair(i_node,j_node)));
            int old_id = Nodes[j_node], new_id = Nodes[i_node];
            for (l_int_t j = 0; j < l; ++j)
                if (Nodes[j] == old_id)
                    Nodes[j] = new_id;
        }
    }
    return Tree;
}
//Алгоритм многопточной быстрой сортировки
void Quick_Sort_OpenMP(int* edges, l_int_t low, l_int_t high, int** nodes)
{
    l_int_t i = low;
    l_int_t j = high;
    l_int_t mid = edges[(i + j) / 2];

    do {
        while (edges[i] < mid) { i++; }
        while (edges[j] > mid) { j--; }

        if (i <= j) {
            swap(edges[i], edges[j]);
            swap(nodes[i], nodes[j]);
            i++;
            j--;
        }
    } while (i <= j);

#pragma omp parallel sections
    {
#pragma omp section
        if (j > low) {
            Quick_Sort_OpenMP(edges, low, j, nodes);
        }
#pragma omp section
        if (i > high) {
            Quick_Sort_OpenMP(edges, i, high, nodes);
        }
    }
}

//Алгоритм Краскала многопоток
l_vec_int_t OpenMP_Kruskal (vector<vector<int>>Edges, vector<int>Nodes)
{
    l_int_t l=Nodes.size();
    l_int_t c=Edges.size();

    int* edges;
    edges = new int[c];
    int** nodes;
    nodes = new int* [c];
    for (l_int_t i = 0;i < c;i++) { nodes[i] = new int[2]; }

    for (l_int_t i = 0;i < c;i++) {
        edges[i] = Edges[i][0];
        nodes[i][0] = Edges[i][1];
        nodes[i][1] = Edges[i][2];
    }

    l_vec_int_t Tree;
    l_int_t i_node, j_node, val;
    omp_set_num_threads(2);

#pragma omp parallel shared(edges)
    {
#pragma omp single //nowait
        {
            Quick_Sort_OpenMP(edges, 0, c - 1, nodes);
        }
    }

    for (l_int_t iterator = 0; iterator < c; iterator++) {
        i_node = nodes[iterator][0];
        j_node = nodes[iterator][1];
        val = edges[iterator];
        if (Nodes[i_node] != Nodes[j_node])
        {
            Tree.push_back(make_pair(val,make_pair(i_node,j_node)));
            int old_id = Nodes[j_node], new_id = Nodes[i_node];
            for (int j = 0; j < l; ++j) {
                if (Nodes[j] == old_id) {
                    Nodes[j] = new_id;
                }
            }
        }
    }
    return Tree;
}
 //Функция создания новых вершин
vector<int> New_Nodes(l_int_t l){
    vector<int>new_nodes;
    int k=0;
    for (l_int_t i=0;i<l;i++){
        new_nodes.push_back(k);
        k++;
    }
    return new_nodes;
}

//Функция генерации ребер графа
matrix_edges_t New_Matrix_Edges(l_int_t l,int probability){
    int INF=99999;
    srand(time(NULL));
    int r;
    vector<int> matrix(l*l);
    l_vec_int_t edges;
    for (l_int_t i = 0;i < l;i++) {
        for (l_int_t j = i;j < l;j++) {
            if (i == j) {
                matrix[i * l + j] = INF;
            }
            else if (rand() % 100 < probability) {
                r= rand() % 10 + 1;
                matrix[i * l + j] = r;
                edges.push_back(make_pair(r,make_pair(i,j)));
            }
            else {
                matrix[i * l + j] = INF;
            }
            matrix[j * l + i] = matrix[i * l + j];
        }
    }
    return make_pair(matrix,edges);
}

//Функция импорта графа из txt файла
nodes_matrix_edges_t Import_Graph(string s) {
    char val;
    int INF=99999;
    l_int_t m = 0;
    l_vec_int_t new_edges;
    vector<int> new_matrix;
    vector<int> new_nodes;

    fstream file(s);

    while (!file.eof()) {
        file.get(val);
        if (val == '\n') { break; }
        else { m++; }
    }

    file.clear();
    file.seekg(0, ios::beg);

    l_int_t n = 1;
    while (!file.eof()) {
        file.get(val);
        if (val == '\n') { n++; }
    }
    file.clear();
    file.seekg(0, ios::beg);

    if (m != n) {
        cout << "Eror! Matrix is not square!" << endl;
        return make_pair(New_Nodes(n), make_pair(new_matrix,new_edges));
    }
    else {

        while (file >> val) new_matrix.push_back(val == '_' ? INF : static_cast<int>(val) - '0');

        for (l_int_t i = 0;i < n;i++) {
            for (l_int_t j = 0;j < n;j++) {
                if (new_matrix[i * n + j] != INF) {
                    new_edges.push_back(make_pair(new_matrix[i * n + j], make_pair(i,j)));
                }
            }
        }
        return make_pair(New_Nodes(n), make_pair(new_matrix,new_edges));
    }
}

//Функция выводит колличество доступных потоков центрального процессора(Используется для отладки)
int Print_NOT() {
    return omp_get_num_procs();
}

//Алгоритм Дейкстры однопоток
vector<int> Dijkstra(vector<int> matrix,l_int_t l,int start){
    int INF=99999;

    vector<bool> visited(l, false);
    vector<int> pos(l, INF);
    pos[start] = 0;
    int min;
    l_int_t index_min;
    for (l_int_t i = 0; i < l; i++) {
        min = INF;
        for (l_int_t j = 0; j < l; j++) {
            if (!visited[j] && pos[j] < min) {
                min = pos[j];
                index_min=j;

            }
        }

        visited[index_min] = true;
        for (l_int_t j = 0; j < l; j++) {
            if (!visited[j] && matrix[index_min * l + j] > 0 && pos[index_min] != INF && pos[index_min] + matrix[index_min * l + j] < pos[j]) {
                pos[j] = pos[index_min] + matrix[index_min * l + j];
            }
        }
    }
    return pos;
}

//Алгоритм Дейкстры многопоток
vector<int> OpenMP_Dijkstra(vector<int> matrix,l_int_t l,int number_of_threads,int start) {
    int INF = 99999;
    vector<bool> visited(l, false);
    vector<int> pos(l, INF);
    pos[start] = 0;
    int min;
    l_int_t index_min;
    for (l_int_t i = 0; i < l; i++) {
        min = INF;
        omp_set_num_threads(number_of_threads);
#pragma omp parallel
        {
            l_int_t index_local;
            int min_local = INF;
#pragma omp for
            for (l_int_t j = 0; j < l; j++) {
                if (!visited[j] && pos[j] < min_local) {
                    min_local = pos[j];
                    index_local = j;
                }
            }
#pragma omp critical
            if (min_local < min) {
                min = min_local;
                index_min = index_local;
            }
        }

        visited[index_min] = true;
#pragma omp parallel for
        for (l_int_t j = 0; j < l; j++) {
            if (!visited[j] && matrix[index_min * l + j] > 0 && pos[index_min] != INF && pos[index_min] + matrix[index_min * l + j] < pos[j]) {
                pos[j] = pos[index_min] + matrix[index_min * l + j];
            }
        }
    }
    return pos;
}






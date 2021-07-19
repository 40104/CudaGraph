// Испольемые библиотеки
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/device_free.h>
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\sort.h>
#include <thrust\iterator\zip_iterator.h>
#include <thrust/gather.h>

#include <string>
#include <string.h>
#include <memory.h>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <thread>
#include <array>
#include <vector>
#include <future>
#include <queue>
#include <functional>
#include <chrono>
#include <time.h> 
#include <omp.h>
#include <stdlib.h>
#include <windows.h>
#include <fstream>
#include <math.h>
using namespace std;

// Ядро kernel
__global__ void  GPU_FW(int* mxx, const int a, const int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a) {
        for (int j = 0;j < a;j++)
        {
            int t = mxx[i * a + K] + mxx[K * a + j];
            mxx[i * a + j] = (mxx[i * a + j] <= t ? mxx[i * a + j] : t);
        }
    }
}


//Создание класса
class Graph
{
    //Хранение значений графа
    vector<int> Nodes;
    vector<int> Matrix;
    vector<vector<int> > Edges;
    const int INF = 99999;

// Набор функций
public:
    Graph() {
        cout << "Start"<<endl;
    }

    //Фукнция генерации графа
    void Generate_Graph(int len, int probability,bool print) {
        auto start_time = clock();
        New_Matrix(len,probability);
        New_Nodes();
     
        if (print) { Print_Graph(); }

        cout << "Time generation new graph: " << (clock() - start_time) / 1000.0 << endl;
        cout << endl;
    }

    //Функция импорта графа
    void Import_Graph(string str, bool print) {
        auto start_time = clock();

        Import_new_Matrix(str);
        New_Nodes();
 
        if (print) { Print_Graph(); }

        cout << "Time import new graph: " << (clock() - start_time) / 1000.0 << endl;
        cout << endl;
    }

    void Import_new_Matrix(string s) {
        char val;
        int m = 0;
        fstream file(s);

        while (!file.eof()) {
            file.get(val);
            if (val == '\n') { break; }
            else { m++; }
        }

        file.clear();
        file.seekg(0, ios::beg);

        int n = 1;
        while (!file.eof()) {
            file.get(val);
            if (val == '\n') { n++; }
        }
        file.clear();
        file.seekg(0, ios::beg);

        if (m != n) {
            cout << "Matrix is not square!" << endl;
        }
        else {
            vector<vector<int>> new_edges;
            vector<int> new_matrix;

            while (file >> val) new_matrix.push_back(val == '_' ? INF : static_cast<int>(val) - '0');

            for (int i = 0;i < n;i++) {
                for (int j = 0;j < n;j++) {
                    if (new_matrix[i * n + j] != INF) {
                        new_edges.push_back({ new_matrix[i * n + j], i, j });
                    }
                }
            }
            Edges = new_edges;
            Matrix = new_matrix;
        }

    }

    //Функции похволяющие пользователю получить значения графов
    vector<int> Get_Nodes() { return Nodes; }
    vector<int> Get_Matrix() { return Matrix; }
    vector<vector<int>> Get_Edges() { return Edges; }
    int Get_Nubmer_Of_Nodes() { return Nodes.size();}
    int Get_Nubmer_Of_Edges() { return Edges.size();}

    //Функция вывода графа
    void Print_Graph()
    {
        const int Nuber_of_Nodes = Nodes.size();
        const int Nuber_of_Edges = Edges.size();
        for (int i = 0;i < Nuber_of_Nodes;i++) {
            cout << endl;
            for (int j = 0;j < Nuber_of_Nodes;j++) {
                ((Matrix[i * Nuber_of_Nodes + j] == INF) ? cout << "0" << " " : cout << Matrix[i * Nuber_of_Nodes + j] << " ");
            }
        }
        cout << endl;
        cout << endl;
        for (int i = 0;i < Nuber_of_Edges;i++) { cout<<"Value edge:"<< Edges[i][0] << " " << "Node 1:"<< Edges[i][1]<< " " << "Node 2:" << Edges[i][2]<<endl; }
        cout << endl;
    }

    void New_Nodes()
    {
        const double Nuber_of_Nodes = sqrt(Matrix.size());
        Nodes.reserve(Nuber_of_Nodes);
        for (int i = 0;i < Nuber_of_Nodes;i++) {
            Nodes.push_back(i);
        }
    }

    void New_Matrix(int len,int probability)
    {
        const int Nuber_of_Nodes = len;
        int V;
        srand(time(NULL));
        Matrix.resize(Nuber_of_Nodes * Nuber_of_Nodes);
        for (int i = 0;i < Nuber_of_Nodes;i++) {
            for (int j = i;j < Nuber_of_Nodes;j++) {
                if (i == j) {
                    Matrix[i * Nuber_of_Nodes + j] = INF;
                }
                else if (rand() % 100 < probability) {
                    V= rand() % 100 + 1;
                    Matrix[i * Nuber_of_Nodes + j] = V;
                    Edges.push_back({V,i,j});
                }
                else {
                    Matrix[i * Nuber_of_Nodes + j] = INF;
                }
                Matrix[j * Nuber_of_Nodes + i] = Matrix[i * Nuber_of_Nodes + j];
            }
        }
    }
    //Вывод информации об устройстве
    void info() {
        int deviceCount, device;
        int gpuDeviceCount = 0;
        struct cudaDeviceProp properties;
        cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
        if (cudaResultCode != cudaSuccess)
            deviceCount = 0;
        for (device = 0; device < deviceCount; ++device) {
            cudaGetDeviceProperties(&properties, device);
            if (properties.major != 9999)
                if (device == 0)
                {
                    printf("multiProcessorCount %d\n", properties.multiProcessorCount);
                    printf("maxThreadsPerMultiProcessor %d\n", properties.maxThreadsPerMultiProcessor);
                }
        }
    }
    
    //Функция Флойда-Уоршелла на центральном процессоре
    void Floid_Warshell(bool print)
    {
        const int Nuber_of_Nodes = Nodes.size();
        double start = omp_get_wtime();
        vector <int> Matrix_Lower_Len = Matrix;
        for (int k = 0; k < Nuber_of_Nodes; k++) {
            for (int i = 0; i < Nuber_of_Nodes; i++) {
                for (int j = 0; j < Nuber_of_Nodes; j++) {
                    Matrix_Lower_Len[i * Nuber_of_Nodes + j] = min(Matrix_Lower_Len[i * Nuber_of_Nodes + j], Matrix_Lower_Len[i * Nuber_of_Nodes + k] + Matrix_Lower_Len[k * Nuber_of_Nodes + j]);
                }
            }
        }
        
        cout << "Algorithm of OPENMP_Floid_Warshell : " << endl;
        double end = omp_get_wtime();
        printf("Time = %.16g\n", end - start);

        if (print) {
            cout << "Number of nodes: " << Nuber_of_Nodes << endl;
            for (int i = 0;i < Nuber_of_Nodes;i++) {
                cout << endl;
                for (int j = 0;j < Nuber_of_Nodes;j++) { ((Matrix_Lower_Len[i * Nuber_of_Nodes + j] == INF) ? cout << "0" << " " : cout << Matrix_Lower_Len[i * Nuber_of_Nodes + j] << " "); }
            }
            cout << endl;
        }
    }

    //Функция Флойда-Уоршелла на графическром процессоре
    void GPU_Floid_Warshell(bool print) {
        const int Nuber_of_Nodes = Nodes.size();
        int* dev_m;
        cudaMalloc((void**)&dev_m, Nuber_of_Nodes * Nuber_of_Nodes * sizeof(int));

        int* Matrix_Lower_Len = new int[Nuber_of_Nodes * Nuber_of_Nodes];
        for (int i = 0;i < Nuber_of_Nodes * Nuber_of_Nodes;i++) { Matrix_Lower_Len[i] = Matrix[i];}
        cout << endl;

        double start = omp_get_wtime();

        cudaMemcpy(dev_m, Matrix_Lower_Len, Nuber_of_Nodes * Nuber_of_Nodes * sizeof(int), cudaMemcpyHostToDevice);

        int block_size = 1024;
        int grid_size = (int)ceil((float)Nuber_of_Nodes / block_size);

        for (int k = 0;k < Nuber_of_Nodes;k++) {
            GPU_FW << <grid_size, block_size >> > (dev_m, Nuber_of_Nodes, k);

        }
        //cudaDeviceSynchronize();

        cudaMemcpy(Matrix_Lower_Len, dev_m, Nuber_of_Nodes * Nuber_of_Nodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(dev_m);
       
        cout << "Algorithm of GPU_Floid_Warshell : " << endl;
        double end = omp_get_wtime();
        printf("Time = %.16g\n", end - start);
        cout << "Number of nodes: " << Nuber_of_Nodes << endl;

        if (print) {
            
            for (int i = 0;i < Nuber_of_Nodes;i++) {
                cout << endl;
                for (int j = 0;j < Nuber_of_Nodes;j++) { ((Matrix_Lower_Len[i * Nuber_of_Nodes + j] == INF) ? cout << "0" << " " : cout << Matrix_Lower_Len[i * Nuber_of_Nodes + j] << " "); }
            }
            cout << endl;
        }
      
    }
    
    //Функция Прима на центральном процессоре
    void Prima(bool print) {
        const int Nuber_of_Nodes = Nodes.size();
        double start = omp_get_wtime();
        vector<int> Matrix_spaning_tree = Matrix;
        vector <bool> visited(Nuber_of_Nodes,false);
        vector<vector<int>> Tree;

        visited[0] = true;
        int counter = 1;
        int min;
        int i_min;
        int j_min;

        while (counter < Nuber_of_Nodes) {
            min = INF;
            i_min = INF;
            j_min = INF;
            for (int i = 0;i < Nuber_of_Nodes;i++) {
                if (visited[i]) {
                    for (int j = 0;j < Nuber_of_Nodes;j++) {
                        if (!visited[j] && Matrix_spaning_tree[i * Nuber_of_Nodes + j] != INF && Matrix_spaning_tree[i * Nuber_of_Nodes + j] < min) {
                            min = Matrix_spaning_tree[i * Nuber_of_Nodes + j];
                            i_min = i;
                            j_min = j;
                        }
                    }
                }
            }
            Tree.push_back({ Matrix_spaning_tree[i_min * Nuber_of_Nodes + j_min],i_min,j_min });
            visited[j_min] = true;
            counter++;
        }
        cout << "Algorithm of Prima : " << endl;
        double end = omp_get_wtime();
        printf("Time = %.16g\n", end - start);
        
        if (print) {
            cout << "Rezult : " << endl;
            for (int i=0;i< Tree.size();i++){ cout << Tree[i][0] << " " << Tree[i][1] << " " << Tree[i][2] << "\n"; } 
            cout << endl;
        }
    }

    //Функция Прима на графическом процессоре
    void GPU_Prima(bool print) {
        const int Nuber_of_Nodes = Nodes.size();
        vector<vector<int>> Tree;
        vector<vector<vector<int> > > edge_list(Nuber_of_Nodes);
        for (int i = 0; i < Edges.size(); ++i) {
            int node1 = Edges[i][1];
            int node2 = Edges[i][2];
            int weight = Edges[i][0];
            edge_list[node1].push_back({ node2, weight });
            edge_list[node2].push_back({ node1, weight });
        }

        double start = omp_get_wtime();

        long long int edge_sum = 0;
        int count = 1;
        int new_node;

        vector<int> weights(Nuber_of_Nodes, INF);
        vector <bool> visited(Nuber_of_Nodes, false);
        vector <vector <int>> nodes(Nuber_of_Nodes);

        nodes[0] = { 0,0 };
        visited[0] = true;

        thrust::device_vector<int> device_weights(weights.begin(), weights.end());
        thrust::device_ptr<int> ptr = device_weights.data();

        while (count < Nuber_of_Nodes) {

            for (int i = 0;i < edge_list.size(); i++) {
                if (visited[i]) {
                    for (int j = 0;j < edge_list[i].size(); j++) {
                        new_node = edge_list[i][j][0];
                        if (!visited[new_node] && weights[new_node] > edge_list[i][j][1]) {
                            weights[new_node] = edge_list[i][j][1];
                            nodes[new_node] = { i,new_node };
                        }
                    }
                }
            }

            device_weights = weights;

            int min_index = thrust::min_element(ptr, ptr + Nuber_of_Nodes) - ptr;

            edge_sum += weights[min_index];
            Tree.push_back({ weights[min_index],nodes[min_index][0],nodes[min_index][1] });
            //cout << count<< " " << weights[min_index] << " " << nodes[min_index][0] << " " << nodes[min_index][1] << endl;
            weights[min_index] = INF;
            visited[nodes[min_index][1]] = true;
            count++;
        }
        cout << "Algorithm of Prima_GPU : " << endl;
        double end = omp_get_wtime();
        printf("Time = %.16g\n", end - start);

        if (print) {
            cout << "Rezult : " << endl;
            for (int i = 0;i < Tree.size(); i++) { cout << Tree[i][0] << " " << Tree[i][1] << " " << Tree[i][2] << endl; }
            cout << endl;
        }
    }

    //Функция Краскала на центральном процессоре
    void Kruskal(bool print) {
        const int Nuber_of_Nodes = Nodes.size();
        const int Nuber_of_Edges = Edges.size();

        double start = omp_get_wtime();

        vector<vector<int> > Kruskal_list_of_edges = Edges;
        vector<bool> visited(Nuber_of_Nodes, false);
        vector<vector<int>> Tree;

        for (int i = 0;i < Nuber_of_Edges;i++) {
            for (int j = 0;j < Nuber_of_Edges;j++) {
                if (Kruskal_list_of_edges[j][0] > Kruskal_list_of_edges[i][0]) {
                    swap(Kruskal_list_of_edges[i], Kruskal_list_of_edges[j]);
                }
            }
        }
        
        int counter = 0;
        int iterator = 0;
        int i_node = 0;
        int j_node = 0;
        int val = 0;

        while (counter < Nuber_of_Nodes && iterator < Nuber_of_Edges ){
            i_node = Kruskal_list_of_edges[iterator][1];
            j_node = Kruskal_list_of_edges[iterator][2];
            val = Kruskal_list_of_edges[iterator][0];
            if ((!visited[i_node]) || (!visited[j_node])) {
                Tree.push_back({ val,i_node,j_node });
                if (!visited[i_node]) {
                    visited[i_node] = true;
                    counter++;
                }
                if (!visited[j_node]) {
                    visited[j_node] = true;
                    counter++;
                }
            }
            iterator++;
        }


        cout << "Algorithm of Kruskal : " << endl;
        double end = omp_get_wtime();
        printf("Time = %.16g\n", end - start);

        if (print) {
            cout << "Rezult : " << endl;
            for (int i = 0;i < Tree.size(); i++) { cout << Tree[i][0] << " " << Tree[i][1] << " " << Tree[i][2] << endl; }
            cout << endl;
        }
    }

    //Функция Краскала на графическом процессоре
    void Kruskal_GPU(bool print) {
        const int Nuber_of_Nodes = Nodes.size();
        const int Nuber_of_Edges = Edges.size();

        double start = omp_get_wtime();

        vector<bool> visited(Nuber_of_Nodes, false);
        vector<vector<int>> Tree;
        
        
        thrust::host_vector<int> weights(Nuber_of_Edges);
        thrust::host_vector<int> i_nodes(Nuber_of_Edges);
        thrust::host_vector<int> j_nodes(Nuber_of_Edges);
        
        for (int k = 0; k < Nuber_of_Edges; k++) {
            weights[k] = Edges[k][0];
            i_nodes[k] = Edges[k][1];
            j_nodes[k] = Edges[k][2];
        }
        
        thrust::device_vector<int> d_weights(weights);
        thrust::device_vector<int> d_i_nodes(i_nodes);
        thrust::device_vector<int> d_j_nodes(j_nodes);

        thrust::counting_iterator<int> iter(0);
        thrust::device_vector<int> indices(Nuber_of_Edges);
        thrust::copy(iter, iter + indices.size(), indices.begin());

        thrust::sort_by_key(d_weights.begin(), d_weights.end(), indices.begin());

        thrust::gather(indices.begin(), indices.end(), d_i_nodes.begin(), d_i_nodes.begin());
        thrust::gather(indices.begin(), indices.end(), d_j_nodes.begin(), d_j_nodes.begin());

        weights = d_weights;
        i_nodes = d_i_nodes;
        j_nodes = d_j_nodes;

        int counter = 0;
        int iterator = 0;
        int i_node = 0;
        int j_node = 0;
        int val = 0;
        while (counter < Nuber_of_Nodes && iterator < Nuber_of_Edges) {
            i_node = i_nodes[iterator];
            j_node = j_nodes[iterator];
            val = weights[iterator];
            if ((!visited[i_node]) || (!visited[j_node])) {
                Tree.push_back({ val,i_node,j_node });
                if (!visited[i_node]) {
                    visited[i_node] = true;
                    counter++;
                }
                if (!visited[j_node]) {
                    visited[j_node] = true;
                    counter++;
                }
            }
            iterator++;
        }
        cout << "Algorithm of Kruskal_GPU : " << endl;
        double end = omp_get_wtime();
        printf("Time = %.16g\n", end - start);
        cout << "Nuber_of_Nodes:" << Nuber_of_Nodes << endl;
        d_weights.clear();
        d_i_nodes.clear();
        d_j_nodes.clear();
        indices.clear();
        if (print) {
            cout << "Rezult : " << endl;
            for (int i = 0;i < Tree.size(); i++) { cout << Tree[i][0] << " " << Tree[i][1] << " " << Tree[i][2] << endl; }
            cout << endl;
        }
    }


    //Функция Дейкстры на графическом процессоре
    void GPU_Dijkstra(bool print,int st) {

        const int Nuber_of_Nodes = Nodes.size();

        double start = omp_get_wtime();
        
        vector<bool> visited(Nuber_of_Nodes, false);
        vector<int> pos(Nuber_of_Nodes, INF);
        vector<int> find_min(Nuber_of_Nodes, INF);
        thrust::device_vector<int> device_weights(find_min.begin(), find_min.end());
        thrust::device_ptr<int> ptr = device_weights.data();
        pos[st] = 0;
        
        for (int i = 0;i < Nuber_of_Nodes;i++) {
            
            for (int j = 0;j < Nuber_of_Nodes;j++) {
                
                if (!visited[j]) {
                    
                    find_min[j] = pos[j];
                }
            }

            device_weights = find_min;
            
            int index_min = thrust::min_element(ptr, ptr + Nuber_of_Nodes) - ptr;
            
            device_weights[index_min] = INF;
            find_min[index_min] = INF;
            visited[index_min] = true;
            
            for (int j = 0;j < Nuber_of_Nodes;j++) {
                if (!visited[j] && Matrix[index_min * Nuber_of_Nodes + j] > 0 && pos[index_min] != INF && pos[index_min] + Matrix[index_min * Nuber_of_Nodes + j] < pos[j]) {
                    
                    pos[j] = pos[index_min] + Matrix[index_min * Nuber_of_Nodes + j];
                }
            }
            
            device_weights.clear();
        }
        cout << "GPU_Dijkstra : " << endl;
        double end = omp_get_wtime();
        printf("Time = %.16g\n", end - start);
        cout << "Nuber_of_Nodes:" << Nuber_of_Nodes << endl;

        if (print) {
            cout << "Rezult : " << endl;
            for (int i = 0;i < Nuber_of_Nodes;i++) {
                cout << pos[i] << " ";
            }
            cout << endl;
        }
    }
};

    
    int main()
    {
        Graph graph2 = Graph();
        graph2.Import_Graph("MX.txt", false);
        graph2.Kruskal_GPU(false);
        return 1;
    }

    

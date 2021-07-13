
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
/*
__global__ void GPU_PR(int* mst,bool* vis,const int new_j, const int n,const int Max,int* re)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    vis[new_j] = true;


    if (vis[i]&& i<n) {
        for (int j = 0;j < n;j++) {
            if (!vis[j] && mst[i * n + j] != Max && mst[i * n + j] < re[0]) {
                re[0] = mst[i * n + j];
                re[1] = i;
                re[2] = j;
            }
        }
    }

}
*/


class Graph
{
    vector<int> Nodes;
    vector<int> Matrix;
    vector<vector<int> > Edges;
    const int INF = 99999;

public:
    Graph() {
        cout << "Start"<<endl;
    }

    void Generate_Graph(int len, int probability,bool print) {
        auto start_time = clock();
        New_Matrix(len,probability);
        New_Nodes();
     
        if (print) { Print_Graph(); }

        cout << "Time generation new graph: " << (clock() - start_time) / 1000.0 << endl;
        cout << endl;
    }

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

    vector<int> Get_Nodes() { return Nodes; }
    vector<int> Get_Matrix() { return Matrix; }
    vector<vector<int>> Get_Edges() { return Edges; }
    int Get_Nubmer_Of_Nodes() { return Nodes.size();}
    int Get_Nubmer_Of_Edges() { return Edges.size();}


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

    void Print_NOT(int number_of_thread) {
        int id;
#pragma omp parallel private(id) num_threads(number_of_thread)
        {
            id = omp_get_thread_num();
            printf("%d: Hello World!\n", id);
        }
    }

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
    
    void OpenMP_Floid_Warshell(int number_of_thread, bool print)
    {
        const int Nuber_of_Nodes = Nodes.size();
        double start = omp_get_wtime();
        vector <int> Matrix_Lower_Len = Matrix;

        for (int k = 0; k < Nuber_of_Nodes; k++) {
            omp_set_dynamic(0);
#pragma omp parallel for num_threads(number_of_thread)
            for (int i = 0; i < Nuber_of_Nodes; i++) {
                auto v = Matrix_Lower_Len[i * Nuber_of_Nodes + k];
                for (int j = 0; j < Nuber_of_Nodes; j++) {
                    auto val = v + Matrix_Lower_Len[k * Nuber_of_Nodes + j];
                    if (Matrix_Lower_Len[i * Nuber_of_Nodes + j] > val) {
                        Matrix_Lower_Len[i * Nuber_of_Nodes + j] = val;
                    }
                }
            }
        }

        cout << "Algorithm of OPENMP_Floid_Warshell : " << endl;
        double end = omp_get_wtime();
        printf("Time = %.16g\n", end - start);
        cout << "Number of nodes: " << Nuber_of_Nodes << endl;
        cout << "Number of threads: " << number_of_thread << endl;

        if (print) {

            for (int i = 0;i < Nuber_of_Nodes;i++) {
                cout << endl;
                for (int j = 0;j < Nuber_of_Nodes;j++) { ((Matrix_Lower_Len[i * Nuber_of_Nodes + j] == INF) ? cout << "0" << " " : cout << Matrix_Lower_Len[i * Nuber_of_Nodes + j] << " "); }
            }
            cout << endl;
        }
    }

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

    void OPENMP_Prima(int number_of_thread, bool print) {
        const int Nuber_of_Nodes = Nodes.size();
        double start = omp_get_wtime();
        vector<int> Matrix_spaning_tree = Matrix;
        vector <bool> visited(Nuber_of_Nodes, false);
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
            for (int i = 0;i < Nuber_of_Nodes;i++)
            {
                if (visited[i])
                {
#pragma omp parallel 
                    { vector<int> rezult_mp{ INF,0, 0 };
#pragma omp for num_threads(number_of_thread)
                    for (int j = 0;j < Nuber_of_Nodes;j++) {
                        if (!visited[j] && Matrix_spaning_tree[i * Nuber_of_Nodes + j] != INF && Matrix_spaning_tree[i * Nuber_of_Nodes + j] < rezult_mp[0]) {
                            rezult_mp[0] = Matrix_spaning_tree[i * Nuber_of_Nodes + j];
                            rezult_mp[1] = i;
                            rezult_mp[2] = j;
                        }
                    }
#pragma omp critical
                    if (rezult_mp[0] < min) {
                        min = rezult_mp[0];
                        i_min = rezult_mp[1];
                        j_min = rezult_mp[2];
                    }

                    }

                }
            }
            Tree.push_back({ min,i_min,j_min });
            visited[j_min] = true;
            counter++;
        }

        cout << "Algorithm of Prima_OPENMP : " << endl;
        double end = omp_get_wtime();
        printf("Time = %.16g\n", end - start);

        if (print) { 
            cout << "Rezult : " << endl;
            for (int i = 0;i < Tree.size(); i++) { cout << Tree[i][0] << " " << Tree[i][1] << " " << Tree[i][2] << endl; }
            cout << endl;
        }
    }

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

    void quick_sort(int* edges, int n, int** nodes) {
        int i = 0;
        int j = n;
        float pivot = edges[n / 2];

        do {
            while (edges[i] < pivot) { i++; }
            while (edges[j] > pivot) { j--; }

            if (i <= j) {
                swap(edges[i], edges[j]);
                swap(nodes[i], nodes[j]);
                i++;
                j--;
            }
        } while (i <= j);

#pragma omp task shared(edges)
        {
            if (j > 0) { quick_sort(edges, j, nodes); }
        } 
#pragma omp task shared(edges)
        {
            if (j > 0) { quick_sort(edges + i, n - i, nodes + i); }
        } 
#pragma omp taskwait
    }

    void Kruskal_OPENMP(bool print) {
        const int Nuber_of_Nodes = Nodes.size();
        const int Nuber_of_Edges = Edges.size();
        double start = omp_get_wtime();

        vector<bool> visited(Nuber_of_Nodes, false);
        vector<vector<int>> Tree;
        int* edges;
        edges = new int[Nuber_of_Edges];
        int** nodes;
        nodes = new int* [Nuber_of_Edges];
        for (int i = 0;i < Nuber_of_Edges;i++) { nodes[i] = new int[2]; }

        for (int i = 0;i < Nuber_of_Edges;i++) {
            edges[i] = Edges[i][0];
            nodes[i][0] = Edges[i][1];
            nodes[i][1] = Edges[i][2];
        }

#pragma omp parallel shared(edges)
        {
#pragma omp single nowait 
            {
                quick_sort(edges, Nuber_of_Edges - 1, nodes);
            } 
        } 

        int counter = 0;
        int iterator = 0;
        int i_node = 0;
        int j_node = 0;
        int val = 0;
        while (counter < Nuber_of_Nodes && iterator < Nuber_of_Edges) {
            i_node = nodes[iterator][0];
            j_node = nodes[iterator][1];
            val = edges[iterator];
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


        cout << "Algorithm of Kruskal_OPENMP : " << endl;
        double end = omp_get_wtime();
        printf("Time = %.16g\n", end - start);

        if (print) {
            cout << "Rezult : " << endl;
            for (int i = 0;i < Tree.size(); i++) { cout << Tree[i][0] << " " << Tree[i][1] << " " << Tree[i][2] << endl; }
            cout << endl;
        }
    }

    void Kruskal_GPU(bool print) {
        const int Nuber_of_Nodes = Nodes.size();
        const int Nuber_of_Edges = Edges.size();

        double start = omp_get_wtime();

        vector<bool> visited(Nuber_of_Nodes, false);
        vector<vector<int>> Tree;
        /*
        thrust::host_vector<int> weights(Edges[0].begin(), Edges[0].end());
        thrust::host_vector<int> i_nodes(Edges[1].begin(), Edges[1].end());
        thrust::host_vector<int> j_nodes(Edges[2].begin(), Edges[2].end());
        */
        
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

    void GPU_Modified_Dijkstra(bool print) {
        vector<vector<int>> returned;
        const int Nuber_of_Nodes = Nodes.size();
        const int Nuber_of_Edges = Edges.size();

        double start = omp_get_wtime();
        
        for (int k = 0;k < Nuber_of_Nodes;k++) {
            vector<bool> visited(Nuber_of_Nodes, false);
            vector<int> pos(Nuber_of_Nodes, INF);
            vector<int> find_min(Nuber_of_Nodes, INF);
            thrust::device_vector<int> device_weights(find_min.begin(), find_min.end());
            thrust::device_ptr<int> ptr = device_weights.data();
            pos[k] = 0;
            int min;
            int index_min;
            
            for (int i = 0;i < Nuber_of_Nodes;i++) {
                min = INF;
                
                for (int j = 0;j < Nuber_of_Nodes;j++) {
                    if (!visited[j]) {
                        
                        find_min[j] = pos[j];
                    }
                }
                
                device_weights = find_min;
                
                int index_min = thrust::min_element(ptr, ptr + Nuber_of_Nodes) - ptr;
                device_weights[index_min] = INF;
                visited[index_min] = true;
                
                for (int j = 0;j < Nuber_of_Nodes;j++) {
                    
                    if (!visited[j] && Matrix[index_min * Nuber_of_Nodes + j] > 0 && pos[index_min] != INF &&
                        pos[index_min] + Matrix[index_min * Nuber_of_Nodes + j] < pos[j]) {
                        pos[j] = pos[index_min] + Matrix[index_min * Nuber_of_Nodes + j];
                    }
                }
                device_weights.clear();
                
            }
            returned.push_back(pos);
        }
        cout << "GPU_Modified_Dijkstra : " << endl;
        double end = omp_get_wtime();
        printf("Time = %.16g\n", end - start);
        cout << "Nuber_of_Nodes:" << Nuber_of_Nodes << endl;
        
        if (print) {
            cout << "Rezult : " << endl;
            for (int i = 0;i < Nuber_of_Nodes;i++) {
                for (int j = 0;j < Nuber_of_Nodes;j++) {
                    cout << returned[i][j] << " ";
                }
                cout << endl;
            }
            
            cout << endl;
        }
    }

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
    

    /*
    void Equal() {
        cout << " ";
        for (int i = 0; i < Len; i++) {
            for (int j = 0; j < Len; j++) {
                if (Matrix_Low_Len_CPU_Open_MP[i * Len + j]!=Matrix_Low_Len_GPU[i * Len + j]) {
                    cout <<"Matrix_Low_Len_CPU_Open_MP: "<<"i="<<i<<"j="<<j<<"M[i][j]="<< Matrix_Low_Len_CPU_Open_MP[i * Len + j];
                    cout <<"Matrix_Low_Len_GPU: " << "i=" << i << "j=" << j << "M[i][j]=" << Matrix_Low_Len_GPU[i * Len + j];
                }
            }
        }
        cout << " ";
    }

*/

    /*
    void GPU_Prima(bool print) {
        double start = omp_get_wtime();
        vector<int> Matrix_spaning_tree = Matrix;
        vector <bool> visited(Nuber_of_Nodes, false);
        vector<vector<int>> Tree;
        vector<int> weights(Nuber_of_Nodes * Nuber_of_Nodes,INF);
        vector<vector<int>> nodes(Nuber_of_Nodes * Nuber_of_Nodes, { INF,INF });
        
        visited[0] = true;
        int counter = 1;
        int min;
        int i_min;
        int j_min;
        int k;
        int min_index;

        thrust::device_vector<int> device_weights(weights.begin(), weights.end());
        thrust::device_ptr<int> ptr = device_weights.data();

        while (counter < Nuber_of_Nodes) {
            k = 0;
            for (int i = 0;i < Nuber_of_Nodes;i++) {
                if (visited[i]) {
                    for (int j = 0;j < Nuber_of_Nodes;j++) {
                        if (!visited[j] && Matrix_spaning_tree[i * Nuber_of_Nodes + j] != INF) {
                            weights[k] = Matrix_spaning_tree[i * Nuber_of_Nodes + j];
                            nodes[k][0] = i;
                            nodes[k][1] = j;
                            k++;
                        }
                    }
                }
            }

            device_weights = weights;

            min_index = thrust::min_element(ptr, ptr + k) - ptr;

            i_min = nodes[min_index][0];
            j_min = nodes[min_index][1];

            Tree.push_back({ Matrix_spaning_tree[i_min * Len + j_min],i_min,j_min });

            visited[j_min] = true;

            fill(weights.begin(), weights.end() , INF);

            for (int x = 0;x < Len * Len;x++) { nodes[x] = {INF,INF}; }
            //fill(nodes.begin(), nodes.end(), { INF, INF});
            counter++;
        }
        cout << "Time algorithm of Pr : " << endl;
        //for (int i = 0;i < Tree.size(); i++) { cout << Tree[i][0] << " " << Tree[i][1] << " " << Tree[i][2] << "\n"; }
        double end = omp_get_wtime();
        printf("Time = %.16g\n", end - start);

    }
    
    void PRP_GPU() {
        
        vector<vector<int>> Tree;

        int* dev_matrix;
        cudaMalloc((void**)&dev_matrix, N * N * sizeof(int));
        
        bool* dev_visited;
        cudaMalloc((void**)&dev_visited,  N * sizeof(bool));
        
        int* dev_min;
        cudaMalloc((void**)&dev_min, 3 * sizeof(int));
        
        int* mx = new int[N * N];
        for (int i = 0;i < Len * Len;i++) { mx[i] = Matrix[i]; }
        
        
        bool* visit = new bool[Len];
        for (int i = 0;i < Len ;i++) { visit[i] = false; }
        visit[0] = true;
        for (int i = 0;i < Len;i++) { cout << visit[i] << " "; }

        
        int* h_min = new int[3];
        for (int i = 0;i < 3;i++) { h_min[i] = INF; }
        
        
        cudaMemcpy(dev_matrix, mx, N * N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_visited, visit, N * sizeof(bool), cudaMemcpyHostToDevice);
        

        int block_size = 16;
        int grid_size = (int)ceil((float)Len / block_size);

        double start = omp_get_wtime();
        int j_min = 0;
        for (int counter = 1;counter < Len;counter++) 
        {
            cudaMemcpy(dev_min, h_min, 3 * sizeof(int), cudaMemcpyHostToDevice);
            GPU_PR << <grid_size, block_size >> > (dev_matrix, dev_visited, j_min,Len,INF, dev_min);
            cudaDeviceSynchronize();
            cudaMemcpy(h_min, dev_min, 3 * sizeof(int), cudaMemcpyDeviceToHost);
            cout << h_min[0] << " " << h_min[1] << " " << h_min[2] << endl;
            Tree.push_back({ h_min[0],h_min[1],h_min[2] });
            j_min = h_min[2];
            counter++;
        }
       
        cout << "Time algorithm of Pr_GPU : " << endl;
        //for (int i = 0;i < Tree.size(); i++) { cout << Tree[i][0] << " " << Tree[i][1] << " " << Tree[i][2] << "\n"; }
        double end = omp_get_wtime();
        printf("Time = %.16g\n", end - start);
        
    }
    */

    
/*
    void Prima()
    {
        auto start_time = clock();
        vector <vector <int>> Tree;
        //Tree.resize(Len);
        //for (int i = 0;i < Len;i++) { Tree[i].resize(3); }
        vector<int> Matrix_spaning_tree = Matrix;
        int counter = 0;
        int min;
        int i_min = 0;
        int j_min = 0;

        vector <bool> selected;
        for (int i = 0;i < Len;i++) { selected.push_back(false); }
        selected[0] = true;

        while (counter < Len - 1) {

            min = INF;
            i_min = 0;
            j_min = 0;

            for (int i = 0; i < Len; i++) {
                if (selected[i]) {
                    for (int j = 0; j < Len; j++) {
                        if (!selected[j] && Matrix_spaning_tree[i * Len + j] != INF) {
                            if (min > Matrix_spaning_tree[i*Len+j]) {
                                min = Matrix_spaning_tree[i * Len + j];
                                i_min = i;
                                j_min = j;
                            }
                        }
                    }
                }
            }

            Tree.push_back({ min,i_min,j_min });
            selected[j_min] = true;
            counter++;
        }
        cout << "Time algorithm of Prima : " << (clock() - start_time) / 1000.0 << endl;
        cout << "\n";
        //for (int i = 0;i < Tree.size(); i++) { cout << Tree[i][0] << " " << Tree[i][1] << " " << Tree[i][2] << "\n"; }
        
    }

    void OpenMP_Prima(int number_of_thread)
    {
        auto start_time = clock();
        vector <vector <int>> Tree;
        //Tree.resize(Len);
        //for (int i = 0;i < Len;i++) { Tree[i].resize(3); }
        vector <int> Matrix_spaning_tree = Matrix;
        int counter = 0;
        vector<bool> visited;
        visited.reserve(Len);
        for (int i = 0;i < Len;i++) { visited.push_back(false); }
        visited[0] = true;
        vector<int> rezult{ INF, 0,0 };
        int min, i_min, j_min;

        while (counter < Len - 1) {
            min = INF;
            i_min = 0;
            j_min = 0;
            for (int i = 0; i < Len; i++) {
                if (visited[i]) {
#pragma omp parallel 
                    {
                        vector<int> rezult_mp{ INF, 0 };
#pragma omp for 
                        for (int j = 0; j < Len; j++) {
                            //cout << omp_get_thread_num();
                            if (!visited[j] && Matrix[i*Len+j] != INF && Matrix[i * Len + j] < rezult_mp[0]) {
                                rezult_mp[0] = Matrix[i * Len + j];
                                rezult_mp[1] = j;
                            }
                        }

                        //int id = omp_get_thread_num();
                        //printf("%d: Hello World!\n", id);

#pragma omp critical

                        if (rezult_mp[0] < rezult[0]) {
                            min = rezult_mp[0];
                            i_min = i;
                            j_min = rezult_mp[1];
                        }
                    }
                }
            }

            Tree.push_back({ min,i_min,j_min });
            visited[j_min] = true;
            counter++;
        }

        cout << "Time algorithm of OpemMP_Prima: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << endl;
        cout << "Number of threads: " << number_of_thread << endl;
        cout << "Number of threads: " << omp_get_num_procs() << endl;
        cout << "Number of nodes: " << Len << endl;
        cout << endl;

        //for (int i = 0;i < Tree.size(); i++) { cout << Tree[i][0] << " " << Tree[i][1] << " " << Tree[i][2] << "\n"; }
        
    }

    void GPU_Prima(int number_of_thread)
    {
        auto start_time = clock();
        vector <vector <int>> Tree;
        Tree.resize(Len);
        for (int i = 0;i < Len;i++) { Tree[i].resize(3); }
        vector <int> Matrix_spaning_tree = Matrix;
        int counter = 0;
        vector<bool> visited;
        visited.reserve(Len);
        for (int i = 0;i < Len;i++) { visited.push_back(false); }
        visited[0] = true;
        vector<int> rezult{ INF, 0,0 };
        int min, i_min, j_min;

        while (counter < Len - 1) {
            min = INF;
            i_min = 0;
            j_min = 0;
            for (int i = 0; i < Len; i++) {
                if (visited[i]) {
#pragma omp parallel num_threads(number_of_thread)
                    {
                        vector<int> rezult_mp{ INF, 0 };
#pragma omp for 
                        for (int j = 0; j < Len; j++) {
                            if (!visited[j] && Matrix[i * Len + j] != INF && Matrix[i * Len + j] < rezult_mp[0]) {
                                rezult_mp[0] = Matrix[i * Len + j];
                                rezult_mp[1] = j;
                            }
                        }

                        //int id = omp_get_thread_num();
                        //printf("%d: Hello World!\n", id);

#pragma omp critical

                        if (rezult_mp[0] < rezult[0]) {
                            min = rezult_mp[0];
                            i_min = i;
                            j_min = rezult_mp[1];
                        }
                    }
                }
            }

            Tree.push_back({ min,i_min,j_min });
            visited[rezult[2]] = true;
            counter++;
        }

        cout << "Time algorithm of OpemMP_Prima: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << endl;
        cout << "Number of threads: " << number_of_thread << endl;
        cout << "Number of threads: " << omp_get_num_procs() << endl;
        cout << "Number of nodes: " << Len << endl;
        cout << endl;

        for (int i = 0;i < Len; i++) {
            cout << rezult[0]<<" " << rezult[1] << " " << rezult[2] << "\n";
        }
    }
      */



};

    
    int main()
    {
        //Graph graph = Graph();
        //graph.Import_Graph("MX.txt",false);
        //graph.GPU_Prima(false);
        //graph.Generate_Graph(1000, 90, false);
        //graph.Print_Graph();
        //graph.Floid_Warshell(true);
        //graph.OpenMP_Floid_Warshell(2);
        //graph.OpenMP_Floid_Warshell(4,false);
        //cout << 1 << endl;
        //graph.GPU_Floid_Warshell(false);
        //cout << 2 << endl;
        //graph.Equal();
        //graph.info();
        //graph.Prima(true);
        //graph.OPENMP_Prima(4, false);
        //graph.GPU_Prima( false);

        //graph.Kruskal(true);
        //graph.Kruskal_OPENMP(true);
        //graph.Kruskal_GPU(false);

        //graph.PRP();
        //graph.PRP_MP(4);
        //graph.PRP_GPU();
        //graph.test();
        //graph.GPU_THR();
        //graph.GPU_PRP();
        //graph.Kruskal();
        //graph.Kruskal_GPU();
        //graph.Kruskal_GPU();
        //graph.Kruskal_OPENMP();

        //graph.Print_NOT(4);
        /*
        Graph graph3 = Graph();
        graph3.Generate_Graph(2000, 90, false);
        graph3.Kruskal_GPU(false);
        graph3.Kruskal_GPU(false);
        */
        //graph3.GPU_Floid_Warshell(false);
        //graph3.GPU_Dijkstra(false,0);
        //graph3.GPU_Modified_Dijkstra(false);
        //graph3.Kruskal(false);
        //graph3.Kruskal_OPENMP(false);
        //graph3.Kruskal_GPU(false);
        //graph3.GPU_Floid_Warshell(false);
        //graph3.Floid_Warshell(false);
        //graph3.OpenMP_Floid_Warshell(4,false);
     
        /*
        Graph graph2 = Graph();
        graph2.Import_Graph("MX.txt", false);
        graph2.Kruskal_GPU(false);
        */
       
       
        
       
        
        /*
       
        Graph graph1 = Graph();
        graph1.Generate_Graph(200, 90, false);
        graph1.GPU_Prima(false);
        graph1.Kruskal_GPU(false);
        graph1.GPU_Floid_Warshell(false);
        graph1.GPU_Dijkstra(false, 0);

        Graph graph2 = Graph();
        graph2.Generate_Graph(400, 90, false);
        graph2.GPU_Prima(false);
        graph2.Kruskal_GPU(false);
        graph2.GPU_Floid_Warshell(false);
        graph2.GPU_Dijkstra(false, 0);

        Graph graph3 = Graph();
        graph3.Generate_Graph(600, 90, false);
        graph3.GPU_Prima(false);
        graph3.Kruskal_GPU(false);
        graph3.GPU_Floid_Warshell(false);
        graph3.GPU_Dijkstra(false, 0);

        Graph graph4 = Graph();
        graph4.Generate_Graph(800, 90, false);
        graph4.GPU_Prima(false);
        graph4.Kruskal_GPU(false);
        graph4.GPU_Floid_Warshell(false);
        graph4.GPU_Dijkstra(false, 0);

        Graph graph5 = Graph();
        graph5.Generate_Graph(1000, 90, false);
        graph5.GPU_Prima(false);
        graph5.Kruskal_GPU(false);
        graph5.GPU_Floid_Warshell(false);
        graph5.GPU_Dijkstra(false, 0);

        Graph graph6 = Graph();
        graph6.Generate_Graph(1200, 90, false);
        //graph6.GPU_Prima(false);
        graph6.Kruskal_GPU(false);
        graph6.GPU_Floid_Warshell(false);
        graph6.GPU_Dijkstra(false, 0);

        Graph graph7 = Graph();
        graph7.Generate_Graph(1400, 90, false);
        //graph7.GPU_Prima(false);
        graph7.Kruskal_GPU(false);
        graph7.GPU_Floid_Warshell(false);
        graph7.GPU_Dijkstra(false, 0);

        Graph graph8 = Graph();
        graph8.Generate_Graph(1600, 90, false);
        //graph8.GPU_Prima(false);
        graph8.Kruskal_GPU(false);
        graph8.GPU_Floid_Warshell(false);
        graph8.GPU_Dijkstra(false, 0);
        
        Graph graph9 = Graph();
        graph9.Generate_Graph(1800, 90, false);
        //graph9.GPU_Prima(false);
        graph9.Kruskal_GPU(false);
        graph9.GPU_Floid_Warshell(false);
        graph9.GPU_Dijkstra(false, 0);
        
        Graph graph10 = Graph();
        graph10.Generate_Graph(2000, 90, false);
        //graph10.GPU_Prima(false);
        graph10.Kruskal_GPU(false);
        graph10.GPU_Floid_Warshell(false);
        graph10.GPU_Dijkstra(false, 0);
        
        
 */
        return 1;
    }

    

//
// Created by One on 24.05.2021.
//

#ifndef UNTITLED12_CLASS_H
#define UNTITLED12_CLASS_H
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
typedef long long unsigned int l_int_t;
typedef vector<pair<int,pair<l_int_t,l_int_t>>> l_vec_int_t;
typedef pair<vector<int>,l_vec_int_t> matrix_edges_t;
typedef pair<vector<int>,matrix_edges_t> nodes_matrix_edges_t;

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


void new_work(l_int_t len, int ptr,int n){
    //Print_NOT(n);
    double start;
    double end;

    cout<< "New Graph len: "<< len<<endl;
    start = omp_get_wtime();
    matrix_edges_t new_mx= New_Matrix_Edges(len, 80);
    vector<int> mx=new_mx.first;
    l_vec_int_t ed=new_mx.second;
    vector<int> nd= New_Nodes(len);
    vector<vector<int>> new_ed;
    for (l_int_t i=0;i<ed.size();i++){
        pair<l_int_t,l_int_t> ff=ed[i].second;
        new_ed.push_back({ed[i].first,(int)ff.first, (int)ff.second});
    }
    /*
    for (int i=0;i<len;i++){
        for (int j=0;j<len;j++) {
            cout<<mx[i*len+j]<<" ";
        }
        cout<<endl;
    }
     */

    cout << "Generation : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);
    cout<<endl;
    start = omp_get_wtime();
    //vector<int> FW = Floid_Warshell (mx,(l_int_t) len);

    cout << "Floid_Warshell : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);

/*
    for (int i=0;i<len;i++){
        for (int j=0;j<len;j++) {
            cout<<FW[i*len+j]<<" ";
        }
        cout<<endl;
    }
*/

    cout<<endl;
    start = omp_get_wtime();
    vector<int> OFW = OpenMP_Floid_Warshell (mx,(l_int_t) len,4);
    cout << "OpenMP_Floid_Warshell : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);

    /*
    for (int i=0;i<len;i++){
        for (int j=0;j<len;j++) {
            cout<<OFW[i*len+j]<<" ";
        }
        cout<<endl;
    }
     */

    cout<<endl;
    start = omp_get_wtime();
    //l_vec_int_t PR=Prima( mx,(l_int_t) len);
    cout << "Prima : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);
    /*
    for (l_int_t i=0;i<PR.size();i++){
        cout<< PR[i].first<<" "<< PR[i].second.first<<" "<< PR[i].second.second<<endl;
    }
     */
    cout<<endl;
    start = omp_get_wtime();
    //l_vec_int_t OPR=OpenMP_Prima( mx,(l_int_t) len,4);
    cout << "OpenMP_Prima : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);
    /*
    for (l_int_t i=0;i<PR.size();i++){
        cout<< PR[i].first<<" "<< PR[i].second.first<<" "<< PR[i].second.second<<endl;
    }
     */
    cout<<endl;
    start = omp_get_wtime();
    //l_vec_int_t KR=Kruskal( new_ed,nd);
    cout << "Kruskal : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);
/*
    for (l_int_t i=0;i<PR.size();i++){
        cout<< KR[i].first<<" "<< KR[i].second.first<<" "<< KR[i].second.second<<endl;
    }
*/
    cout<<endl;
    start = omp_get_wtime();
    l_vec_int_t OKR=OpenMP_Kruskal( new_ed,nd);
    cout << "OpenMP_Kruskal : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);
/*
    for (l_int_t i=0;i<PR.size();i++){
        cout<< OKR[i].first<<" "<< OKR[i].second.first<<" "<< OKR[i].second.second<<endl;
    }
*/

    cout<<endl;
    start = omp_get_wtime();
    //vector<int> DI=Dijkstra( mx,(l_int_t) len,0);
    cout << "Dijkstra : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);
    cout<<endl;
/*
    for (l_int_t i=0;i<(l_int_t) len;i++){

        cout<< DI[i]<<" ";

        cout<<endl;
    }
*/

    cout<<endl;
    start = omp_get_wtime();
    //vector<int> ODI= OpenMP_Dijkstra( mx,(l_int_t) len,4,0);
    cout << "OpenMP_Dijkstra : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);
    cout<<endl;
/*
    for (l_int_t i=0;i<(l_int_t) len;i++){

        cout<< ODI[i]<<" ";

        cout<<endl;
    }
    */
/*
    cout<<endl;
    start = omp_get_wtime();
    vector<vector<int>> NDI= New_Modified_Dijkstra( mx,(l_int_t) len,4);
    cout << "New_Modified_Dijkstra : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);
    cout<<endl;
    */
/*
    for (l_int_t i=0;i<(l_int_t) len;i++){
        for (l_int_t j=0;j<(l_int_t) len;j++){
            cout<< ODI[i][j]<<" ";
        }
        cout<<endl;
    }
    */



    }


int main(){
/*
    double start;
    double end;
    l_int_t len=10;
    cout<< "New Graph len: "<< len<<endl;
    start = omp_get_wtime();
    matrix_edges_t new_mx= New_Matrix_Edges(len, 90);
    vector<int> mx=new_mx.first;
    l_vec_int_t ed=new_mx.second;
    vector<int> nd= New_Nodes(len);
    vector<vector<int>> new_ed;
    for (l_int_t i=0;i<ed.size();i++){
        pair<l_int_t,l_int_t> ff=ed[i].second;
        new_ed.push_back({ed[i].first,(int)ff.first, (int)ff.second});
    }

    cout<<endl;
    start = omp_get_wtime();
    vector<vector<int>> DI=Modified_Dijkstra( mx,(l_int_t) len);
    cout << "Dijkstra : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);
    cout<<endl;
    for (l_int_t i=0;i<(l_int_t) len;i++){
        for (l_int_t j=0;j<(l_int_t) len;j++){
            cout<< DI[i][j]<<" ";
        }
        cout<<endl;
    }

    cout<<endl;
    start = omp_get_wtime();
    vector<vector<int>> ODI= OpenMP_Modified_Dijkstra( mx,(l_int_t) len,4);
    cout << "OpenMP_Dijkstra : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);
    cout<<endl;

    for (l_int_t i=0;i<(l_int_t) len;i++){
        for (l_int_t j=0;j<(l_int_t) len;j++){
            cout<< ODI[i][j]<<" ";
        }
        cout<<endl;
    }

    cout<<endl;
*/
/*
    Print_NOT(4);
    double start;
    double end;


    start = omp_get_wtime();
    cout<<0<<endl;
    nodes_matrix_edges_t new_mx= Import_Graph("MX.txt");
    cout<<9<<endl;
    vector<int> nd=new_mx.first;
    vector<int>mx=new_mx.second.first;
    l_vec_int_t ed=new_mx.second.second;
    cout<<1<<endl;
    vector<vector<int>> new_ed;
    for (l_int_t i=0;i<ed.size();i++){
        pair<l_int_t,l_int_t> ff=ed[i].second;
        new_ed.push_back({ed[i].first,(int)ff.first, (int)ff.second});
    }
    cout<<2<<endl;
    start = omp_get_wtime();
    vector<int> FW = Floid_Warshell (mx,(l_int_t) nd.size());

    cout << "Floid_Warshell : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);
    cout<<endl;
    start = omp_get_wtime();
    vector<int> OFW = OpenMP_Floid_Warshell (mx,(l_int_t) nd.size(),4);
    cout << "OpenMP_Floid_Warshell : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);
    cout<<endl;
    start = omp_get_wtime();
    l_vec_int_t KR=Kruskal( new_ed,nd);
    cout << "Kruskal : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);
/*
    for (l_int_t i=0;i<PR.size();i++){
        cout<< KR[i].first<<" "<< KR[i].second.first<<" "<< KR[i].second.second<<endl;
    }

    cout<<endl;
    start = omp_get_wtime();
    l_vec_int_t OKR=OpenMP_Kruskal( new_ed,nd);
    cout << "OpenMP_Kruskal : " << endl;
    end = omp_get_wtime();
    printf("Time = %.16g\n", end - start);

    for (l_int_t i=0;i<PR.size();i++){
        cout<< OKR[i].first<<" "<< OKR[i].second.first<<" "<< OKR[i].second.second<<endl;
    }
    */
    new_work(200,90,4);
/*
    new_work(400,90,4);
    new_work(600,90,4);
    new_work(800,90,4);
    new_work(1000,90,4);

    new_work(1200,90,4);
    new_work(1400,90,4);
    new_work(1600,90,4);
    new_work(1800,90,4);
    new_work(2000,90,4);
*/
    return 1;
}
#endif //UNTITLED12_CLASS_H

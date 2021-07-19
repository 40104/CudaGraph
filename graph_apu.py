#Импорт необходимых билиотек
import cpu_graph as cg
import Gpu_graph_thust as gg
from time import time
import networkx as nx

#Создание итогового класса
class Graph_algorythms:
    #Переменные хранения графа
    def __init__(self):
        self.nodes = []
        self.matrix = []
        self.edges = []

    #Декоратор измерения времени
    def measure_time(function):
        def timed(*args, **kwargs):
            begin = time()
            result = function(*args, **kwargs) 
            end = time() 
            
            print('Function '+ str(function)+' completed in {:5.3f} seconds'.format(end - begin)) 
            return result 
        return timed
        
    #Функция генерации графа
    @measure_time  
    def Generate_New_Graph(self,l,p,ptr=False):
        
        self.nodes=cg.New_Nodes(l)
        mid_rezult=cg.New_Matrix_Edges(l,p)
        self.matrix= mid_rezult[0]
        self.edges=mid_rezult[1]
        dd=[]
        for i in self.edges:
            dd.append([i[0],i[1][0],i[1][1]])
        self.edges=dd    
        if ptr == True:
            print(self.nodes)
            print(self.matrix)
            print(self.edges)
    #Алгоритм Флойда-Уоршелла CPU однопоток        
    @measure_time        
    def Floid_Warshell(self,ptr=False):
        mid=cg.Floid_Warshell(self.matrix,len(self.nodes))
        if ptr == True:
            print(mid)
    #Алгоритм Флойда-Уоршелла CPU многопоток        
    @measure_time        
    def OpenMP_Floid_Warshell(self,number_of_treads,ptr=False):
        mid=cg.OpenMP_Floid_Warshell(self.matrix,len(self.nodes),number_of_treads)
        if ptr == True:
            print(mid)
    #Алгоритм Флойда-Уоршелла GPU        
    @measure_time        
    def GPU_Floid_Warshell(self,ptr=False):
        mid=gg.Gpu_Floid_Warshell(self.matrix)
        if ptr == True:
            print(mid)
    #Алгоритм Прима СPU однопоток      
    @measure_time        
    def Prima(self,ptr=False):
        mid=cg.Prima(self.matrix,len(self.nodes))
        if ptr == True:
            print(mid)
    #Алгоритм Прима СPU многопоток        
    @measure_time
    def OpenMP_Prima(self,number_of_treads,ptr=False):
        mid=cg.OpenMP_Prima(self.matrix,len(self.nodes),number_of_treads)
        if ptr == True:
            print(mid)
    #Алгоритм Краскала СPU однопоток        
    @measure_time        
    def Kruskal(self,ptr=False):
        mid=cg.Kruskal(self.edges,self.nodes)
        if ptr == True:
            print(mid)
    #Алгоритм Краскала СPU многопоток        
    @measure_time        
    def OpenMP_Kruskal(self,ptr=False):
        mid=cg.OpenMP_Kruskal(self.edges,self.nodes)
        if ptr == True:
            print(mid)
    #Алгоритм Краскала GPU         
    @measure_time
    def Gpu_Kruskal(self,ptr=False):
        mid=gg.Gpu_Kruskal(self.edges,self.nodes)
        if ptr == True:
            print(mid)
    #Функция отрисовки графа        
    def Plt_Graph(self):
        G=nx.Graph()
        ge=[[i[1],i[2],i[0]] for i in self.edges]
        G.add_weighted_edges_from(ge)
        nx.draw(G,pos=nx.spectral_layout(G), nodecolor='r',edge_color='b')
    #Функция импорта графа
    @measure_time   
    def Import_Graph(self,name,ptr=False):
        mid_rezult=cg.Import_Graph(name)
        self.nodes=mid_rezult[0]
        self.matrix=mid_rezult[1][0]
        self.edges=mid_rezult[1][1]
        dd=[]
        for i in self.edges:
            dd.append([i[0],i[1][0],i[1][1]])
        self.edges=dd    
        if ptr == True:
            print(self.nodes)
            print(self.matrix)
            print(self.edges)
    #Функция получения колличества потоков
    def Number_Of_Treads(self):
        print(cg.Print_NOT())
    #Алгоритм Дейкстры СPU однопоток
    @measure_time
    def Dijkstra(self,s,ptr=False):
        mid=cg.Dijkstra(self.matrix,len(self.nodes),s)
        if ptr == True:
            print(mid)
    #Алгоритм Дейкстры СPU многопоток
    @measure_time        
    def OpenMP_Dijkstra(self,s,n,ptr):
        mid=cg.OpenMP_Dijkstra(self.matrix,len(self.nodes),n,s)
        if ptr == True:
            print(mid)
    #Алгоритм Флойда-Уоршелла Python    
    @measure_time         
    def Python_Floid_Warshell(self,ptr):
        matrix=self.matrix
        ln=len(self.nodes)
        for k in range(ln):
            for i in range(ln):
                for j in range(ln):
                    t=matrix[i*ln+k]+matrix[k*ln+j]
                    matrix[i*ln+j]=matrix[i*ln+j] if matrix[i*ln+j]<=t else t
        if ptr == True:
            print(matrix)
    #Алгоритм Прима Python        
    @measure_time        
    def Python_Prima(self,ptr):
        l=len(self.nodes)
        INF=99999
        visited=[False for i in range(l)]
        visited[0]=True
        tree=[]
        counter=1
        while counter<l:
            minn=INF
            i_min=INF
            j_min=INF
            for i in range(l):
                if visited[i]==True:
                    for j in range(l):
                        if visited[j]==False and self.matrix[i*l+j]!=INF and self.matrix[i*l+j]<minn:
                            minn=self.matrix[i*l+j]
                            i_min=i
                            j_min=j
            tree.append([minn,i_min,j_min])
            visited[j_min]=True
            counter+=1
        if ptr == True:
            print(tree)
            
    #Алгоритм Краскала Python        
    @measure_time
    def Python_Kraskal(self,ptr):
        l=len(self.nodes)
        c=len(self.edges)
        tree=[]
        edg=[]
        nod=[]
        Nodes=self.nodes
        for i in self.edges:
            edg.append(i[0])
            nod.append([i[1],i[2]])
        
        for i in range(c):
            for j in range(c):
                if edg[i]<edg[j]:
                    a=edg[i]
                    b=nod[i]
                    edg[i]=edg[j]
                    edg[j]=a
                    nod[i]=nod[j]
                    nod[j]=b
        
        for i in range(0,c):
            i_node=nod[i][0]
            j_node=nod[i][1]
            val=edg[i]
            if Nodes[i_node]!=Nodes[j_node]:
                tree.append([val,i_node,j_node])
                old=Nodes[j_node]
                new=Nodes[i_node] 
                for j in range(0,l): 
                    if Nodes[j]==old: 
                        
                        Nodes[j]=new
        if ptr == True: 
            print(tree)
    #Алгоритм Дейкстры Python        
    @measure_time
    def Python_Dijkstra(self,start,ptr):
        l=len(self.nodes)
        mx=self.matrix
        INF=99999
        visited=[False for i in range(l) ]
        pos=[INF for i in range(l)]
        
        pos[start]=0
        for i in range(l):
            
            minn=INF
            for j in range(l):
                if (visited[j]==False) and (pos[j]<minn):
                    minn=pos[j]
                    index_min=j
                    
            visited[index_min]=True 
           
            for j in range(l): 
                if (visited[j]==False) and (mx[index_min*l+j]>0) and (pos[index_min] !=INF) and (pos[index_min]+mx[index_min*l+j]< pos[j]): 
                    pos[j]=pos[index_min]+mx[index_min*l+j]
        if ptr == True: 
            print(pos)
        

           
        
        
                        
                
                    
                    
            
        
        
        
                    
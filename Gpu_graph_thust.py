import ThrustRTC as trtc
import math

def Gpu_Floid_Warshell(matrix):
    kernel_Floid_Warshell = trtc.Kernel(['arr', 'k','l'],
	'''
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i <= l) {
            for (size_t j = 0;j < l;j++){
                    int t=arr[i * l + k]+ arr[k * l + j];
                    arr[i * l + j]=(arr[i * l + j] <= t ? arr[i * l + j]  : t);
                    }
            }
	''')
    
    dvec_in = trtc.device_vector_from_list(matrix, 'int32_t')
    l=int(math.sqrt(len(matrix)))
    
    dv_l = trtc.DVInt32(l)
    block_size=256
    grid_size=math.ceil(l/block_size) 
                
    for k in range(0,l):
        dv_k = trtc.DVInt32(k)
        kernel_Floid_Warshell.launch( grid_size,block_size, [dvec_in, dv_k,dv_l])

    return(dvec_in.to_host())
 
   
def Gpu_Kruskal(edges,nodes):
    values=[]
    key_1=[]
    key_2=[]
    for i in edges:
        values.append(i[0])
        key_1.append(i[1])
        key_2.append(i[2])
        
    dvalues= trtc.device_vector_from_list(values, 'int32_t')    
    dkey_1= trtc.device_vector_from_list(key_1, 'int32_t')    
    dkey_2= trtc.device_vector_from_list(key_2, 'int32_t')
    it=[i for i in range(0,len(edges))]
    iter= trtc.device_vector_from_list(it, 'int32_t')
 
    trtc.Sort_By_Key( dvalues, iter)
    
    doutput_1 = trtc.device_vector('int32_t', dvalues.size())
    doutput_2 = trtc.device_vector('int32_t', dvalues.size())
    
    trtc.Gather(iter, dkey_1, doutput_1)
    trtc.Gather(iter, dkey_2, doutput_2)
    
    def add_tree(output_1,output_2,value):
        Tree=[]
        for i in range (0,len(edges)):
            i_node=output_1[i] 
            j_node=output_2[i] 
            val=value[i] 
            if nodes[i_node]!=nodes[j_node]:
                Tree.append([val,i_node,j_node]) 
                old=nodes[j_node] 
                new=nodes[i_node] 
                for j in range(0,len(nodes)):
                    if nodes[j]==old: 
                        nodes[j]=new
       
        return Tree

    
    return add_tree(doutput_1.to_host().tolist(),doutput_2.to_host().tolist(),dvalues.to_host().tolist())
    

    

    



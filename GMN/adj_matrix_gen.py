import os
import numpy as np
import glob
import concurrent.futures

ISA='x86'
#input_folder="dataset/gdl_pf"
input_folder = glob.glob(f'dataset/gdl_pf/{ISA}/**/**/*.gdl', recursive=True)
output_folder=f"dataset/graph_adj_matrix/{ISA}"
os.makedirs(output_folder,exist_ok=True)

def adj_gen(filename,output_folder):
    with open(filename) as f:
        text=f.read()
    node_num=0
    lines=text.split('\n')
    for i,line in enumerate(lines):
        if line.startswith("// node 0"):
            newlines=lines[i:-2] #extract the information that include the nodes and edges
            break
    for newline in newlines:
        if newline.startswith('//'):
            node_num+=1
    adj_matrix=np.zeros((node_num,node_num),dtype=int)
    for line in newlines:
        if line.startswith('edge:'):
            parts=line.split()
            source=int(parts[3].strip('"'))
            target=int(parts[5].strip('"'))
            if len(parts)>7:
                edge_label=parts[7].strip('"')
                if edge_label=="True":
                    adj_matrix[source][target]=2
                else:
                    adj_matrix[source][target]=-1  
            else: 
                adj_matrix[source][target]=1  
    name_parts=filename.split('/')
    lib=name_parts[-2].split('-')[0]
    bin=name_parts[-1].split('@')[0]
    func=name_parts[-1].split('@')[1].strip('.gdl')
    opt_level=name_parts[-2].split('-')[-1]
    version=name_parts[-2].split('-')[-2]
    if func.startswith('sub'):
        #print(f"Delete sub_XX function: {func}")
        return
    print(f"lib:{lib},bin:{bin},version:{version},func:{func},opt_level:{opt_level}")
    output_filename=ISA+'@'+lib+'@'+bin+'@'+version+'@'+func+'@'+opt_level+'.csv'
    #print(output_filename)
    output_path=os.path.join(output_folder,output_filename)                
    np.savetxt(output_path,adj_matrix,delimiter=',',fmt='%d')
        #return adj_matrix              

def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
                futures = {executor.submit(adj_gen, file_path, output_folder): file_path for file_path in input_folder}
                for future in concurrent.futures.as_completed(futures):
                    file_path = futures[future]
                    try:
                        future.result()
                        print(f"Processed {file_path} successfully.")
                    except Exception as exc:
                        print(f"File {file_path} generated an exception: {exc}")
if __name__ == '__main__':
    main()                             
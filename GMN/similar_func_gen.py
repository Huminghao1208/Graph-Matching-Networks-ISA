import os
import glob
import random
import shutil
import concurrent.futures
from collections import defaultdict
import itertools

ISA = 'x86'
input_folder = f'dataset/graph_adj_matrix_all/{ISA}'
output_folder = f"dataset/similar_func_pairs/{ISA}"
PAIRS_TO_GENERATE = 50000

def group_files(input_folder):
    file_groups = defaultdict(list)
    for file_path in glob.glob(f'{input_folder}/*.csv'):
        filename = os.path.basename(file_path)
        parts = filename.split('@')
        key = '@'.join(parts[:-1])  # Group by everything except optimization level
        file_groups[key].append(file_path)
    return file_groups

def generate_similar_pairs(file_groups):
    similar_pairs = []
    for group in file_groups.values():
        if len(group) > 1:
            similar_pairs.extend((a, b) for a in group for b in group if a < b)
    return similar_pairs

def generate_unsimilar_pairs(file_groups):
    all_files = [file for group in file_groups.values() for file in group]
    unsimilar_pairs = set()
    all_possible_pairs = list(itertools.combinations(all_files, 2))
    random.shuffle(all_possible_pairs)
    
    for a, b in all_possible_pairs:
        if a.split('@')[:-1] != b.split('@')[:-1]:
            unsimilar_pairs.add((min(a, b), max(a, b)))
        if len(unsimilar_pairs) == PAIRS_TO_GENERATE:
            break
    
    return list(unsimilar_pairs)

def copy_pair(pair, output_dir, index):
    a, b = pair
    a_name = f'{index}$a${os.path.basename(a)}'
    b_name = f'{index}$b${os.path.basename(b)}'
    shutil.copy(a, os.path.join(output_dir, a_name))
    shutil.copy(b, os.path.join(output_dir, b_name))

def func_pair_gen(input_folder, output_folder):
    os.makedirs(os.path.join(output_folder, 'pos'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'neg'), exist_ok=True)

    file_groups = group_files(input_folder)
    
    similar_pairs = generate_similar_pairs(file_groups)
    random.shuffle(similar_pairs)
    similar_pairs = similar_pairs[:min(PAIRS_TO_GENERATE, len(similar_pairs))]

    unsimilar_pairs = generate_unsimilar_pairs(file_groups)

    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
        similar_futures = [executor.submit(copy_pair, pair, os.path.join(output_folder, 'pos'), i) 
                           for i, pair in enumerate(similar_pairs)]
        unsimilar_futures = [executor.submit(copy_pair, pair, os.path.join(output_folder, 'neg'), i) 
                             for i, pair in enumerate(unsimilar_pairs)]
        
        concurrent.futures.wait(similar_futures + unsimilar_futures)

    print(f"Generated {len(similar_pairs)} similar pairs and {len(unsimilar_pairs)} unsimilar pairs.")

def main():
    func_pair_gen(input_folder, output_folder)

if __name__ == '__main__':
    main()
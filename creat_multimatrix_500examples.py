#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:16:43 2023

@author: hjx
"""
import numpy as np
import os
from tqdm import tqdm
from multiparameter_landscape import multiparameter_landscape
"""
save rivet
"""
import subprocess
import shlex
mulG_file = "/home/hjx/Mix_GENEO/multiG_500examples/bifiltration/"
mulD_file = "/home/hjx/Mix_GENEO/multiD_500examples/bifiltration/"
mixG_file = "/home/hjx/Mix_GENEO/mixG_500examples/bifiltration/"
rivet_executable = '/home/hjx/Downloads/rivet/rivet_console'
base_mulG = "/home/hjx/Mix_GENEO/multiG_500examples/rivet_output/"
base_mulD = "/home/hjx/Mix_GENEO/multiD_500examples/rivet_output/"
base_mixG = "/home/hjx/Mix_GENEO/mixG_500examples/rivet_output/"
mulG_files = os.listdir(mulG_file)
mulG_files.sort(key=lambda x:int(x[9:].split('.')[0]))
mulD_files = os.listdir(mulD_file)
mulD_files.sort(key=lambda x:int(x[9:].split('.')[0]))
mixG_files = os.listdir(mixG_file)
mixG_files.sort(key=lambda x:int(x[9:].split('.')[0]))

def rivet_name(base,id, homology):
    output_name = base+("rivet_output_h%d/"%(homology)) +id + ("_H%d.rivet" % (homology))
    return output_name

l = [0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051]
def compute_file(input_name,base, output_name=None, homology=0):
    if not output_name:
        input_name_i = os.path.basename(input_name)
        (input_name_id, suffix) = os.path.splitext(input_name_i)
        output_name = rivet_name(base, input_name_id, homology)
        #print(output_name)
    cmd = "%s %s %s -H %d --xbins 10 --ybins 10 -f msgpack" % \
          (rivet_executable, input_name, output_name, homology)
    subprocess.check_output(shlex.split(cmd))
    return output_name

for f1 in tqdm(mulG_files):
    f = mulG_file + f1
    #print(f)
    output_name = compute_file(f, base_mulG, homology=1)
    output_name = compute_file(f, base_mulG, homology=0)

for f2 in tqdm(mulD_files):
    f = mulD_file + f2
    #print(f)
    output_name = compute_file(f, base_mulD, homology=1)
    output_name = compute_file(f, base_mulD, homology=0)

for f3 in tqdm(mixG_files):
    f = mixG_file + f3
    #print(f)
    output_name = compute_file(f, base_mixG, homology=1)
    output_name = compute_file(f, base_mixG, homology=0)

"""
save mutilandscape matrix
"""
def get_multiland(input_name, output_path, homology):
    with open(input_name, 'rb') as f:
        computed_data = f.read()
    multi_landscape = multiparameter_landscape(
        computed_data, maxind=3, grid_step_size=10, bounds=[[0, 0], [255, 255]]
    )

    output_name_i = os.path.basename(input_name)
    (output_name_id, suffix) = os.path.splitext(output_name_i)
    np.save(output_path+output_name_id+".npy", multi_landscape.landscape_matrix)

multiland_path_mulG = "/home/hjx/Mix_GENEO/multiG_500examples/multiland_output/"
multiland_path_mulD = "/home/hjx/Mix_GENEO/multiD_500examples/multiland_output/"
multiland_path_mixG = "/home/hjx/Mix_GENEO/mixG_500examples/multiland_output/"

def save_multiland(rivet_p, multiland_output_p, homology):
    rivet_input_p = rivet_p+"rivet_output_h%d/" % homology
    rivet_files = os.listdir(rivet_input_p)
    rivet_files.sort(key=lambda x: int(x[9:].split('_')[0]))
    multiland_output_pa = multiland_output_p+"multiland_output_h%d/" % homology
    l1 = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    for w in l1:
        for i in tqdm(rivet_files[w:w+500]):
            rivet_input_files = rivet_input_p + i
            get_multiland(rivet_input_files, multiland_output_pa, homology)

rivet_path = [base_mulG, base_mulD, base_mixG]
multi_path = [multiland_path_mulG, multiland_path_mulD, multiland_path_mixG]
for r,m in zip(rivet_path,multi_path):
    save_multiland(r, m, 0)
    save_multiland(r, m, 1)

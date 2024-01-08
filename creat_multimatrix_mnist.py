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
# file_train = "/home/hjx/Mix_GENEO/mulG/x_train/bifiltration/"
# file_test = "/home/hjx/Mix_GENEO/mulG/x_test/bifiltration/"
# file_train = "/home/hjx/Mix_GENEO/mulD/x_train/bifiltration/"
# file_test = "/home/hjx/Mix_GENEO/mulD/x_test/bifiltration/"
file_train = "/home/hjx/Mix_GENEO/mixG/x_train/bifiltration/"
file_test = "/home/hjx/Mix_GENEO/mixG/x_test/bifiltration/"
rivet_executable = '/home/hjx/Downloads/rivet/rivet_console'
# base_test = "/home/hjx/Mix_GENEO/test_mnist_multiGENEO/rivet_output/"
# base_train = "/home/hjx/Mix_GENEO/train_mnist_multiGENEO/rivet_output/"
# base_test = "/home/hjx/Mix_GENEO/test_mnist_multiDGENEO/rivet_output/"
# base_train = "/home/hjx/Mix_GENEO/train_mnist_multiDGENEO/rivet_output/"
base_test = "/home/hjx/Mix_GENEO/test_mnist_mixGENEO/rivet_output/"
base_train = "/home/hjx/Mix_GENEO/train_mnist_mixGENEO/rivet_output/"

def rivet_name(base,id, homology):
    output_name = base+("rivet_output_h%d/"%(homology)) +id + ("_H%d.rivet" % (homology))
    return output_name

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
"""
buid_test_rivet
"""
files_test = os.listdir(file_test)
files_test.sort(key=lambda x:int(x[9:].split('.')[0]))
print(len(files_test))
for f1 in tqdm(files_test):
    f = file_test + f1
    output_name = compute_file(f, base_test, homology=1)

"""
buid_test_rivet
"""
files_test = os.listdir(file_test)
files_test.sort(key=lambda x:int(x[9:].split('.')[0]))
print(len(files_test))
for f1 in tqdm(files_test):
    f = file_test + f1
    output_name = compute_file(f, base_test, homology=0)

"""
build_train_rivet
"""
files_train = os.listdir(file_train)
files_train.sort(key=lambda x:int(x[9:].split('.')[0]))
print(len(files_train))
for f1 in tqdm(files_train):
    f = file_train + f1
    output_name = compute_file(f, base_train, homology=0)

"""
build_train_rivet
"""
files_train = os.listdir(file_train)
files_train.sort(key=lambda x:int(x[9:].split('.')[0]))
print(len(files_train))
for f1 in tqdm(files_train):
    f = file_train + f1
    output_name = compute_file(f, base_train, homology=1)

"""
save mutilandscape matrix
"""
def get_multiland(input_name, output_path, homology):
    with open(input_name, 'rb') as f:
        computed_data = f.read()
    multi_landscape = multiparameter_landscape(
        computed_data, maxind=3, grid_step_size=10, bounds=[[0, 0], [255, 255]]
    )
    #print(multi_landscape.landscape_matrix)
    output_name_i = os.path.basename(input_name)
    (output_name_id, suffix) = os.path.splitext(output_name_i)
    np.save(output_path+output_name_id+".npy", multi_landscape.landscape_matrix)

def save_multiland(rivet_p, multiland_output_p, homology):
    rivet_input_p = rivet_p+"rivet_output_h%d/" % homology
    rivet_files = os.listdir(rivet_input_p)
    rivet_files.sort(key=lambda x: int(x[9:].split('_')[0]))
    multiland_output_pa = multiland_output_p+"multiland_output_h%d/" % homology
    for i in tqdm(rivet_files):
        rivet_input_files = rivet_input_p + i
        get_multiland(rivet_input_files, multiland_output_pa, homology)
"""
build_train_multiland
"""
# train_rivet_path = "/home/hjx/Mix_GENEO/train_mnist_multiGENEO/rivet_output/"
# train_multiland_output_path = "/home/hjx/Mix_GENEO/train_mnist_multiGENEO/multiland_output/"
# train_rivet_path = "/home/hjx/Mix_GENEO/train_mnist_multiDGENEO/rivet_output/"
# train_multiland_output_path = "/home/hjx/Mix_GENEO/train_mnist_multiDGENEO/multiland_output/"
train_rivet_path = "/home/hjx/Mix_GENEO/train_mnist_mixGENEO/rivet_output/"
train_multiland_output_path = "/home/hjx/Mix_GENEO/train_mnist_mixGENEO/multiland_output/"
save_multiland(train_rivet_path, train_multiland_output_path, 0)
save_multiland(train_rivet_path, train_multiland_output_path, 1)
"""
build_test_multiland
"""
# test_rivet_path = "/home/hjx/Mix_GENEO/test_mnist_multiGENEO/rivet_output/"
# test_multiland_output_path = "/home/hjx/Mix_GENEO/test_mnist_multiGENEO/multiland_output/"
# test_rivet_path = "/home/hjx/Mix_GENEO/test_mnist_multiDGENEO/rivet_output/"
# test_multiland_output_path = "/home/hjx/Mix_GENEO/test_mnist_multiDGENEO/multiland_output/"
test_rivet_path = "/home/hjx/Mix_GENEO/test_mnist_mixGENEO/rivet_output/"
test_multiland_output_path = "/home/hjx/Mix_GENEO/test_mnist_mixGENEO/multiland_output/"
save_multiland(test_rivet_path, test_multiland_output_path, 0)
save_multiland(test_rivet_path, test_multiland_output_path, 1)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from get_convolve_img import turn_to_img
import scipy as sp
import tensorflow as tf
import platform
if platform.system() == "Darwin":
    matplotlib.use("TkAgg")
from tqdm import tqdm
import create_multicomplex_simplify as cms

def convolve_img_with_kernel(kernel,img, params = None):
    return (sp.ndimage.convolve(img, kernel) if params is None
            else sp.ndimage.convolve(img, kernel, **params))

data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()
ker_0 = np.loadtxt('/home/hjx/Mix_GENEO/ker1/img1_kernel_8.txt', delimiter=',')
print(ker_0)
ker_1 = np.loadtxt('/home/hjx/Mix_GENEO/ker2/img2_kernel_4.txt', delimiter=',')
ker_2 = np.loadtxt('/home/hjx/Mix_GENEO/ker1/img1_kernel_4.txt', delimiter=',')

ker_3 = np.loadtxt('/home/hjx/Mix_GENEO/ker2/img2_kernel_3.txt', delimiter=',')
ker_4 = np.loadtxt('/home/hjx/Mix_GENEO/ker2/img2_kernel_4.txt', delimiter=',')
img_3 = convolve_img_with_kernel(ker_3,x_train[0])
img_4 = convolve_img_with_kernel(ker_4,x_train[0])
img_3_4 = img_4-img_3

def get_bifiltration(x,path):
    L = len(x)
    for j in tqdm(range(L)):
        """
        mulG
        """
        # img_0 = convolve_img_with_kernel(ker_0, x[j])
        # img_0 = turn_to_img(img_0, len(img_0))
        # n = len(img_0)
        # img_1 = x[j]
        # img_0 = turn_to_img(img_0, len(img_0))
        # img_1 = turn_to_img(img_1, len(img_1))
        """
        mulD
        """
        # img_3 = convolve_img_with_kernel(ker_3, x[j])
        # img_4 = convolve_img_with_kernel(ker_4, x[j])
        # img_0 = img_4 - img_3
        # n = len(img_0)
        # img_0 = turn_to_img(img_0, len(img_0))
        # img_1_1 = convolve_img_with_kernel(ker_1, x[j])
        # img_1_2 = convolve_img_with_kernel(ker_2, x[j])
        # img_1 = img_1_2 - img_1_1
        # img_1 = turn_to_img(img_1, len(img_1))
        """
        mixG
        """
        img_0 = convolve_img_with_kernel(ker_0, x[j])
        img_0 = turn_to_img(img_0, len(img_0))
        n = len(img_0)
        img_3 = convolve_img_with_kernel(ker_3, x[j])
        img_4 = convolve_img_with_kernel(ker_4, x[j])
        img_1 = img_4 - img_3
        img_1 = turn_to_img(img_1, len(img_1))
        vertex = cms.get_points(img_0, img_1)
        edge = cms.get_edges(vertex, n)
        face = cms.get_faces(vertex, n)
        f_1_2 = open(path+"x_ctrain_{}.txt".format(j), 'w')
        f_1_2.write("--datatype bifiltration\n--xlabel x-label\n--ylabel y-label\n\n# data starts here\n")
        # f_1_2.write("1 ; 0 0 \n2 ; 1 0\n1 2 ; 2 0")
        for i in range(len(vertex)):
            f_1_2.write('{} ; {} {}\n'.format(vertex[i].num, vertex[i].t_x, vertex[i].t_y))
        for j in range(len(edge)):
            f_1_2.write('{} {} ; {} {}\n'.format(edge[j].vertex_1.num, edge[j].vertex_2.num, edge[j].e_x, edge[j].e_y))
        for k in range(len(face)):
            f_1_2.write(
                '{} {} {} ; {} {}\n'.format(face[k].vertex_1.num, face[k].vertex_2.num, face[k].vertex_3.num,
                                            face[k].f_x,
                                            face[k].f_y))
        f_1_2.close()
"""
mulG
"""
# path_test = '/home/hjx/Mix_GENEO/mulG/x_test/bifiltration/'
# path_train = '/home/hjx/Mix_GENEO/mulG/x_train/bifiltration/'
# get_bifiltration(x_test, path_test)
# get_bifiltration(x_train, path_train)
"""
mulD
"""
# path_test = '/home/hjx/Mix_GENEO/mulD/x_test/bifiltration/'
# path_train = '/home/hjx/Mix_GENEO/mulD/x_train/bifiltration/'
# get_bifiltration(x_test, path_test)
# get_bifiltration(x_train, path_train)
"""
mixG
"""
path_test = '/home/hjx/Mix_GENEO/mixG/x_test/bifiltration/'
path_train = '/home/hjx/Mix_GENEO/mixG/x_train/bifiltration/'
get_bifiltration(x_test, path_test)
get_bifiltration(x_train, path_train)
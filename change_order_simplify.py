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
x_ctrain = []
ker_0 = np.loadtxt('/home/hjx/Mix_GENEO/ker1/img1_kernel_8.txt', delimiter=',')
print(ker_0)
ker_1 = np.loadtxt('/home/hjx/Mix_GENEO/ker2/img2_kernel_4.txt', delimiter=',')
ker_2 = np.loadtxt('/home/hjx/Mix_GENEO/ker1/img1_kernel_4.txt', delimiter=',')

ker_3 = np.loadtxt('/home/hjx/Mix_GENEO/ker2/img2_kernel_3.txt', delimiter=',')
ker_4 = np.loadtxt('/home/hjx/Mix_GENEO/ker2/img2_kernel_4.txt', delimiter=',')
img_3 = convolve_img_with_kernel(ker_3,x_train[0])
img_4 = convolve_img_with_kernel(ker_4,x_train[0])
img_3_4 = img_4-img_3

for i in range(10):

    for j in range(len(y_train)):
        if y_train[j] == i:
            x_ctrain.append(x_train[j])

def get_bifiltration(x_ctrain,path):
    l = [0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051]
    for w in l:
        print("现在是大概",w)
        for j in tqdm(range(500)):
            """
            mulG
            """
            # img_0 = convolve_img_with_kernel(ker_0, x_ctrain[j+w])
            # img_0 = turn_to_img(img_0, len(img_0))
            # n = len(img_0)
            # img_1 = x_ctrain[j+w]
            # img_0 = turn_to_img(img_0, len(img_0))
            # img_1 = turn_to_img(img_1, len(img_1))
            """
            mulD
            """
            # img_3 = convolve_img_with_kernel(ker_3, x_ctrain[j+w])
            # img_4 = convolve_img_with_kernel(ker_4, x_ctrain[j+w])
            # img_0 = img_4 - img_3
            # n = len(img_0)
            # img_0 = turn_to_img(img_0, len(img_0))
            # img_1_1 = convolve_img_with_kernel(ker_1, x_ctrain[j+w])
            # img_1_2 = convolve_img_with_kernel(ker_2, x_ctrain[j+w])
            # img_1 = img_1_2 - img_1_1
            # img_1 = turn_to_img(img_1, len(img_1))
            """
            mixG
            """
            img_0 = convolve_img_with_kernel(ker_0, x_ctrain[j+w])
            img_0 = turn_to_img(img_0, len(img_0))
            n = len(img_0)
            img_3 = convolve_img_with_kernel(ker_3, x_ctrain[j+w])
            img_4 = convolve_img_with_kernel(ker_4, x_ctrain[j+w])
            img_1 = img_4 - img_3
            img_1 = turn_to_img(img_1, len(img_1))
            vertex = cms.get_points(img_0, img_1)
            edge = cms.get_edges(vertex, n)
            face = cms.get_faces(vertex, n)


            f_1_2 = open(path+"x_ctrain_{}.txt".format(j + w), 'w')
            f_1_2.write("--datatype bifiltration\n--xlabel x-label\n--ylabel y-label\n\n# data starts here\n")
            # f_1_2.write("1 ; 0 0 \n2 ; 1 0\n1 2 ; 2 0")
            for i in range(len(vertex)):
                f_1_2.write('{} ; {} {}\n'.format(vertex[i].num, vertex[i].t_x, vertex[i].t_y))
            for j in range(len(edge)):
                f_1_2.write('{} {} ; {} {}\n'.format(edge[j].vertex_1.num, edge[j].vertex_2.num, edge[j].e_x, edge[j].e_y))
            for k in range(len(face)):
                f_1_2.write(
                    '{} {} {} ; {} {}\n'.format(face[k].vertex_1.num, face[k].vertex_2.num, face[k].vertex_3.num, face[k].f_x,
                                                face[k].f_y))
            f_1_2.close()


mulG_file = "/home/hjx/Mix_GENEO/multiG_500examples/bifiltration/"
mulD_file = "/home/hjx/Mix_GENEO/multiD_500examples/bifiltration/"
mixG_file = "/home/hjx/Mix_GENEO/mixG_500examples/bifiltration/"
get_bifiltration(x_ctrain, mulG_file)
get_bifiltration(x_ctrain, mulD_file)
get_bifiltration(x_ctrain, mixG_file)
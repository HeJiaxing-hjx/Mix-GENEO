from create_multiimg import GIENEO
import numpy as np
import tensorflow as tf
import matplotlib
import platform
if platform.system() == "Darwin":
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()
img_0 = x_train[0]

def get_convovle_img(img):
    g1 = GIENEO(size=5, sigma=5, centers=5)
    g1.generate()
    img1 = g1.convolve(img)
    g2 = GIENEO(size=5, sigma=5, centers=5)
    g2.generate()
    img2 = g2.convolve(img)

    array1 = np.array(img1)

    m = array1.shape[0]
    #print(m)
    array2 = np.array(img2)
    img1 = turn_to_img(array1, m)
    #print("img1", img1)
    img2 = turn_to_img(array2, m)
    #print("img2", img2)
    fig, axs = plt.subplots(1, 2)
    axs = axs.ravel()
    axs[0].imshow(img1, cmap="gray")
    axs[1].imshow(img2, cmap="gray")
    plt.show()
    return img1,img2, g1.kernel, g2.kernel


def turn_to_img(array, n):
    ymax = 255
    ymin = 0
    xmax = max(map(max, array))
    xmin = min(map(min, array))

    for i in range(n):
        for j in range(n):
            array[i][j] = round(((ymax - ymin) * (array[i][j] - xmin) / (xmax - xmin)) + ymin)#round(number[,ndigits]), return：返回一个数字，默认情况下，返回值类型应和参数number类型相同
    return array

# img1, img2, ker1, ker2 = get_convovle_img(img_0)
# f_1 = open('D://jisuan//multi_parameter//ctest2//ker1//img1_kernel.txt','w')
# np.savetxt('D:/jisuan/multi_parameter/ctest2/ker1/img1_kernel_10.txt', ker1, delimiter=',')
# f_1.close()
# f_2 = open('D://jisuan//multi_parameter//ctest2//ker2//img2_kernel.txt','w')
# np.savetxt('D:/jisuan/multi_parameter/ctest2/ker2/img2_kernel_10.txt', ker2, delimiter=',')
# f_2.close()
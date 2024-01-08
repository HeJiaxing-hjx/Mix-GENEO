import matplotlib
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import platform
if platform.system() == "Darwin":
    matplotlib.use("TkAgg")
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf

class GIENEO:
    def __init__(self, size, sigma = None, centers = None, signal_dim = None):
        self.size = size

        self.centers = centers
        print("self.centers", self.centers)
        self.components = len(self.centers)
        self.sigma = sigma
        print("self.sigma", self.sigma)
        self.g = self.generate_1d()
        print("self.g")
        if signal_dim is not None:
            self.generate(dim=signal_dim)



    @property
    def centers(self):
        print(1)
        print("self._centers",self._centers)
        return self._centers

    @centers.setter
    def centers(self, new_centers):
        print("new_centers", new_centers)
        if isinstance(new_centers, int):
            self._centers = self.uniform_initializer(new_centers,
                                                    target_range = (-1,1))
            print("self._centers", self._centers)
            print(3)
        else:
            new_centers = np.asarray(new_centers)
            assert new_centers.ndim == 1, "The array of centers should have dimension 1"
            self._centers = self.scale(new_centers)

    def update_parameters(self, coefficients, centers, sigma):
        """Allows to update the parameters and generate the kernel
        """
        self.centers = centers
        self.sigma = sigma
        self.g = self.generate_1d(coefficients = coefficients)
        self.generate(dim =self.dim)


    @property
    def sigma(self):
        print("05")
        return self._sigma



    @sigma.setter
    def sigma(self, new_sigma):
        if isinstance(new_sigma, float):
            self._sigma = [new_sigma for c in range(self.components)]
            print("00")
        elif isinstance(new_sigma, int):
            self._sigma = self.uniform_initializer(new_sigma,
                                                   target_range = (0,.1))
            print("01")
        else:
            self._sigma = new_sigma
            print("02")
        assert len(self.sigma) == len(self.centers), "Provide as many values for sigma as centers"

    @staticmethod
    def uniform_initializer(num_coeffs, target_range=(-1,1)):
        r = target_range
        print("target_range", r)
        return (r[1] - r[0]) * np.random.random_sample(num_coeffs) + r[0]



    def generate_1d(self, coefficients=None):
        if coefficients is None:
            self.coefficients = self.uniform_initializer(self.components)
            print("self.coefficients", self.coefficients)
        else:
            self.coefficients = coefficients
        self.coefficients = self.coeffs_normalization(regularize=True)
        x = np.linspace(-1, 1, self.size)
        sym_gaussians = np.asarray([coeff * self.get_sym_gaussian(x, center, s)
                                    for coeff, center, s
                                    in zip(self.coefficients, self.centers, self.sigma)])
        sym_gaussians = np.sum(sym_gaussians, axis=0)
        print(sym_gaussians.min())
        sym_gaussians += abs(sym_gaussians.min())
        return sym_gaussians#27

    @staticmethod
    def coeffs_normalization_(coeff, center):
        return coeff / center if center != 0 else coeff

    def coeffs_normalization(self, regularize=True):
        """Generating the filters by revolving 1d arrays around the origin,
        it is possible to normalize the coefficients to compoensate for the
        difference in area spanned by each Gaussian as a function of its
        distance from the origin"""
        coeffs = [self.coeffs_normalization_(co, ce)
                  for co, ce in zip(self.coefficients, self.centers)]
        print("coeffs", coeffs)
        if regularize:
            return coeffs / np.linalg.norm(np.asarray(coeffs)) ** 2
        else:
            return coeffs



    def generate(self, dim = 2):
        """Generate an operator as a normalized linear combination of symmetric
        Gaussian
        """
        self.dim = dim
        if not hasattr(self, 'g'):
            self.g = self.generate_1d()
        x = np.linspace(-.5, .5, self.size)
        print("len(x)",len(x))
        grid = itertools.product(x, repeat = dim)
        indices = itertools.product(range(self.size), repeat = dim)

        self.kernel = self.init_kernel_tensor(self.size, dim)
        int_func = interp1d(x, self.g)

        for index, values in zip(indices, grid):
            self.kernel[index] = int_func(sum([v**2 for v in values]))
        #print(self.kernel)

    @staticmethod
    def gaussian(eval, center, sigma):
        return np.exp(( -(eval - center)**2) / (2 * sigma**2))

    def get_sym_gaussian(self, x, c, s):
        return self.gaussian(x, c, s) + self.gaussian(-x, c, s)

    @staticmethod
    def init_kernel_tensor(size, dim):
        dimensions = [size for i in range(dim)]
        print("dimensions", dimensions)
        return np.zeros(dimensions)

    def convolve(self, image, params = None):
        """Convolve the kernel with the image by using scipy.ndimage.convolve
        with the parameters specified in params
        """
        #?
        return (sp.ndimage.convolve(image, self.kernel) if params is None
                else sp.ndimage.convolve(image, self.kernel, **params))

    def visual_image(self, image, axes = None, convolve_params = None):
        if axes is None:
            fig, axs = plt.subplots(1,3)
            axs = axs.ravel()
        convolved_img = self.convolve(image, params=convolve_params)
        print(self.kernel)
        print("image", image)
        print("convolved_img", convolved_img)
        print(self.kernel.size)
        print(image.size)#784
        print(convolved_img.size)#784
        axs[0].imshow(self.kernel, cmap="gray")
        axs[1].imshow(image, cmap="gray")
        axs[2].imshow(convolved_img, cmap="gray")
        plt.show()







if __name__ == "__main__":
    plt.ion()
    g = GIENEO(size=3, sigma=5, centers=5)
    g.generate()
    print(g.centers)
    plt.ioff()

    data = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = data.load_data()
    g.visual_image(x_train[1000])
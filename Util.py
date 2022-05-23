from PIL import Image
import numpy as np
from scipy.ndimage import uniform_filter, binary_opening    # Mean filter, bwareaopen
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.titlesize"] = 14
matplotlib.rcParams["figure.titleweight"] = 'bold'
matplotlib.rcParams["axes.titlesize"] = 12


class ImageFusion:
    def __init__(self, im_1, im_2, is_path_1=True, is_path_2=True):
        self.ImageToFuse1 = ImageToFuse(im_1, is_path_1)
        self.ImageToFuse2 = ImageToFuse(im_2, is_path_2)

        if self.ImageToFuse1.Im.shape != self.ImageToFuse1.Im.shape:
            print("Images must have same dimensions!")
            return

        self.IDM = None
        self.IDMbw_area = None
        self.IDMbw = None
        self.FDM = None

        self.IF = None

    def plot_image(self, var_str, title="", show=False):
        im_to_plot = getattr(self, var_str)
        if len(im_to_plot.shape) == 3:
            cmap = plt.rcParams["image.cmap"]
        else:
            cmap = 'gray'
        plt.imshow(im_to_plot, cmap=cmap)
        plt.title(title)
        plt.axis('off')

        if show:
            plt.show()

    def plot_all(self):
        self.ImageToFuse1.plot_all()
        self.ImageToFuse2.plot_all()

        plt.figure().suptitle("Decision Maps")
        plt.subplot(1, 2, 1)
        self.plot_image("IDMbw", "Initial Decision Map (bw)")
        plt.subplot(1, 2, 2)
        self.plot_image("FDM", "Final Decision Map")

        plt.figure().suptitle("Final Fused Image")
        plt.subplot(1, 3, 1)
        self.ImageToFuse1.plot_image("Im", "Original Image 1")
        plt.subplot(1, 3, 2)
        self.ImageToFuse2.plot_image("Im", "Original Image 2")
        plt.subplot(1, 3, 3)
        self.plot_image("IF", "Fused Image")

        plt.show()

    def image_fusion_run(self, size, r, eps, bw_size):
        self.mean_filter(size)
        self.rough_foucs_map()
        self.accurate_foucs_map(r, eps)
        self.initial_decision_map()
        self.initial_decision_map_bw_opening(bw_size)
        self.final_decision_map(r, eps)
        self.image_fusion()

    def mean_filter(self, size):
        """
        Go to ImageToFuse.mean_filter for more details
        """
        M1 = self.ImageToFuse1.mean_filter(size)
        M2 = self.ImageToFuse2.mean_filter(size)
        return M1, M2

    def rough_foucs_map(self):
        """
        Go to ImageToFuse.rough_foucs_map for more details
        """
        RFM1 = self.ImageToFuse1.rough_foucs_map()
        RFM2 = self.ImageToFuse2.rough_foucs_map()
        return RFM1, RFM2

    def accurate_foucs_map(self, r, eps):
        """
        Go to ImageToFuse.accurate_foucs_map for more details
        """
        AFM1 = self.ImageToFuse1.accurate_foucs_map(r, eps)
        AFM2 = self.ImageToFuse2.accurate_foucs_map(r, eps)
        return AFM1, AFM2

    def initial_decision_map(self):
        """
        Creates initial descision map by taking max of both accurate descision maps

        :return: Initial Descision Map
        """
        self.IDM = self.ImageToFuse1.AFM > self.ImageToFuse2.AFM
        return self.IDM

    def initial_decision_map_bw_opening(self, size):
        """
        Improve initial decision map with binary opening -
        removing areas smaller than size from the boolean map

        :param size:    Binary Opening (MATLAB bwareaopen) area size
        :return:        Improved Initial Descision Map
        """
        self.IDMbw_area = size
        sq = np.ones((self.IDMbw_area, self.IDMbw_area), dtype=bool)
        temp_1 = binary_opening(self.IDM, sq)
        temp_2 = 1 - temp_1
        temp_3 = binary_opening(temp_2, sq)
        self.IDMbw = 1 - temp_3
        return self.IDMbw

    def final_decision_map(self, r, eps):
        """
        Creates final desicion map with initial fused image

        :param r:   Guided filter R
        :param eps: Guided filter Epsilon
        :return:    Final Descision map
        """
        temp_1 = self.IDMbw * self.ImageToFuse1.ImGray + (1-self.IDMbw) * self.ImageToFuse2.ImGray
        self.FDM = guided_filter(temp_1, self.IDMbw, r, eps)
        return self.FDM

    def image_fusion(self):
        """
        Creates fused image, using the FDM map on 3 R-G-B layers

        :return: Fused Image
        """
        FDM = self.FDM
        if len(self.ImageToFuse1.Im.shape) == 3:
            FDM = np.repeat(self.FDM[:, :, np.newaxis], 3, axis=2)
        self.IF = FDM*self.ImageToFuse1.Im + (1-FDM)*self.ImageToFuse2.Im
        # Normalize IF
        self.IF = (self.IF - self.IF.min())/(self.IF.max() - self.IF.min())
        return self.IF


class ImageToFuse:
    def __init__(self, im, is_path=True):
        """
        Class init function
        images are loaded as "double" images for guided filter mean/var easy calcs
        other vars are set to None for PEP8 standart init

        :param im:      Image to Fuse
        :param is_path: Is im path (True) or image (False)
        """
        if is_path:
            self.Im = np.asarray(Image.open(im), dtype=np.float64)/255
            self.ImGray = np.asarray(Image.open(im).convert('L'), dtype=np.float64) / 255
        else:
            self.Im = im
            im = (im*255).astype(np.uint8)
            self.ImGray = np.asarray(Image.fromarray(im).convert('L'), dtype=np.float64)/255

        self.mean_filter_size = None
        self.M = None
        self.RFM = None
        self.AFM = None

    def plot_image(self, var_str, title="", show=False):
        im_to_plot = getattr(self, var_str)

        if len(im_to_plot.shape) == 3:
            cmap = plt.rcParams["image.cmap"]
        else:
            cmap = 'gray'

        plt.imshow(im_to_plot, cmap=cmap)
        plt.title(title)
        plt.axis('off')

        if show:
            plt.show()

    def plot_all(self):
        plt.figure().suptitle("Image to Fuse")
        plt.subplot(2, 2, 1)
        self.plot_image("Im", "Original Image")
        plt.subplot(2, 2, 2)
        self.plot_image("ImGray", "Grayscale Image")
        plt.subplot(2, 3, 4)
        self.plot_image("M", "Avg filtered Image")
        plt.subplot(2, 3, 5)
        self.plot_image("RFM", "Rough focus map")
        plt.subplot(2, 3, 6)
        self.plot_image("AFM", "Accurate focus map")
        plt.show(block=False)

    def mean_filter(self, size):
        """
        Mean filter calculation

        :param size:    Median Filter Kernel Size
        :return:        Image after median filter
        """
        self.mean_filter_size = size
        self.M = uniform_filter(self.ImGray, size=self.mean_filter_size)
        return self.M

    def rough_foucs_map(self):
        """
        Create rough focus map by subtracting median filtered image from original image

        :return: Rough Focus Map
        """

        self.RFM = np.abs(self.ImGray-self.M)
        return self.RFM

    def accurate_foucs_map(self, r, eps):
        """
                Create accurate focus map by using guided filter

        :param r:
        :param eps:
        :return:
        """
        """
        """
        self.AFM = guided_filter(self.ImGray, self.RFM, r, eps)
        return self.AFM


def box(img, r):
    """ O(1) box filter
        img - >= 2d image
        r   - radius of box filter
    """
    (rows, cols) = img.shape[:2]
    imDst = np.zeros_like(img)

    tile = [1] * img.ndim
    tile[0] = r
    imCum = np.cumsum(img, 0)
    imDst[0:r+1, :, ...] = imCum[r:2*r+1, :, ...]
    imDst[r+1:rows-r, :, ...] = imCum[2*r+1:rows, :, ...] - imCum[0:rows-2*r-1, :, ...]
    imDst[rows-r:rows, :, ...] = np.tile(imCum[rows-1:rows, :, ...], tile) - imCum[rows-2*r-1:rows-r-1, :, ...]

    tile = [1] * img.ndim
    tile[1] = r
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r+1, ...] = imCum[:, r:2*r+1, ...]
    imDst[:, r+1:cols-r, ...] = imCum[:, 2*r+1 : cols, ...] - imCum[:, 0 : cols-2*r-1, ...]
    imDst[:, cols-r: cols, ...] = np.tile(imCum[:, cols-1:cols, ...], tile) - imCum[:, cols-2*r-1 : cols-r-1, ...]

    return imDst


def guided_filter(I, p, r, eps):
    """ grayscale (fast) guided filter
        I - guide image (1 channel)
        p - filter input (1 channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
    """

    (rows, cols) = I.shape

    N = box(np.ones([rows, cols]), r)

    meanI = box(I, r) / N
    meanP = box(p, r) / N

    corrI = box(I * I, r) / N
    corrIp = box(I * p, r) / N

    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = box(a, r) / N
    meanB = box(b, r) / N

    q = meanA * I + meanB
    return q

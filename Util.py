from PIL import Image
import numpy as np
from scipy.ndimage import uniform_filter, binary_opening    # Mean filter, bwareaopen
import ctypes                                               # DPI Fixes
import matplotlib.pyplot as plt
import matplotlib

# Matplotlib default title size
matplotlib.rcParams["figure.titlesize"] = 16
matplotlib.rcParams["figure.titleweight"] = 'bold'
matplotlib.rcParams["axes.titlesize"] = 14

# Fix DPI issues when using "full screen" figures
ctypes.windll.shcore.SetProcessDpiAwareness(0)


class ImageFusion:
    def __init__(self, im_1, im_2, is_path_1=True, is_path_2=True):
        """
        Class init function

        im_1 (path/ndarray):    Image1 to Fuse
        im_2 (path/ndarray):    Image2 to Fuse
        is_path_1 (bool):       Is im1 a path or an image
        is_path_2 (bool):       Is im2 a path or an image
        """
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
        """
        Plot single image

        var_str (string):   Name of image to plot (get var with getattr)
        title (string):     Plot title
        show (bool):        Plot.show()
        """
        im_to_plot = getattr(self, var_str)
        if len(im_to_plot.shape) == 3:
            cmap = plt.rcParams["image.cmap"]
        else:
            cmap = 'gray'
        plt.imshow(im_to_plot, cmap=cmap)
        plt.title(title)
        plt.yticks([])
        plt.xticks([])

        if show:
            plt.tight_layout()
            plt.get_current_fig_manager().window.state('zoomed')
            plt.show()

    def plot_all(self, block=True):
        """
        Plot all images

        block (bool): Wait for figure to be closed by user before moving on
        """
        self.ImageToFuse1.plot_all(block=False)
        self.ImageToFuse2.plot_all(block=True)

        plt.figure().suptitle("Decision Maps")
        plt.subplot(1, 2, 1)
        self.plot_image("IDMbw", "Initial Decision Map (bw)")
        plt.subplot(1, 2, 2)
        self.plot_image("FDM", "Final Decision Map")
        plt.tight_layout()
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()

        plt.figure().suptitle("Final Fused Image")
        plt.subplot(1, 3, 1)
        self.ImageToFuse1.plot_image("Im", "Original Image 1")
        plt.subplot(1, 3, 2)
        self.ImageToFuse2.plot_image("Im", "Original Image 2")
        plt.subplot(1, 3, 3)
        self.plot_image("IF", "Fused Image")

        plt.tight_layout()
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show(block=block)

    def plot_compare(self, image_cmp):
        """
        Plot comparison images of 2 images to fuse

        image_cmp (string): Name of images to plot (get var with getattr)
        """
        plt.figure()
        plt.get_current_fig_manager().window.state('zoomed')
        plt.subplot(1, 2, 1)
        self.ImageToFuse1.plot_image(image_cmp, "Image 1")
        plt.subplot(1, 2, 2)
        self.ImageToFuse2.plot_image(image_cmp, "Image 2")
        plt.tight_layout()
        plt.show()

    def export_image(self, var_str, filename):
        """
        Export JPG image

        var_str (string):   Name of image to export (get var with getattr)
        filename (string):  Output file name
        """
        im_to_export = getattr(self, var_str)
        im_to_export = (im_to_export*255).astype(np.uint8)
        im_to_export = Image.fromarray(im_to_export)
        im_to_export.save(filename + ".jpg")

    def image_fusion_run(self, size, r, eps, bw_size):
        """
        Run all methods

        Go to methods for more details
        """
        self.mean_filter(size)
        self.rough_focus_map()
        self.accurate_focus_map(r, eps)
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

    def rough_focus_map(self):
        """
        Go to ImageToFuse.rough_focus_map for more details
        """
        RFM1 = self.ImageToFuse1.rough_focus_map()
        RFM2 = self.ImageToFuse2.rough_focus_map()
        return RFM1, RFM2

    def accurate_focus_map(self, r, eps):
        """
        Go to ImageToFuse.accurate_focus_map for more details
        """
        AFM1 = self.ImageToFuse1.accurate_focus_map(r, eps)
        AFM2 = self.ImageToFuse2.accurate_focus_map(r, eps)
        return AFM1, AFM2

    def initial_decision_map(self):
        """
        Creates initial decision map by taking max of both accurate decision maps

        return (ndarray): Initial Decision Map
        """
        self.IDM = self.ImageToFuse1.AFM > self.ImageToFuse2.AFM
        return self.IDM

    def initial_decision_map_bw_opening(self, size):
        """
        Improve initial decision map with binary opening -
        removing areas smaller than {size} from the boolean map

        size (int):         Binary Opening (MATLAB bwareaopen) area size
        return (ndarray):   Improved Initial Decision Map
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
        Creates final decision map with initial fused image

        Go to guided_filter for more details
        return (ndarray): Final Decision map
        """
        temp_1 = self.IDMbw * self.ImageToFuse1.ImGray + (1-self.IDMbw) * self.ImageToFuse2.ImGray
        self.FDM = guided_filter(temp_1, self.IDMbw, r, eps)
        return self.FDM

    def image_fusion(self):
        """
        Creates fused image, using the FDM map on 3 R-G-B layers

        return (ndarray): Fused Image
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
        other vars are set to None for PEP8 standard init

        im (ndarray/path):  Image to Fuse
        is_path (bool):     Is im a path or an image
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
        """
        Plot single image

        var_str (string):   Name of image to plot (get var with getattr)
        title (string):     Plot title
        show (bool):        Plot.show()
        """
        im_to_plot = getattr(self, var_str)
        if len(im_to_plot.shape) == 3:
            cmap = plt.rcParams["image.cmap"]
        else:
            cmap = 'gray'
        plt.imshow(im_to_plot, cmap=cmap)
        plt.title(title)
        plt.yticks([])
        plt.xticks([])

        if show:
            plt.tight_layout()
            plt.get_current_fig_manager().window.state('zoomed')
            plt.show()

    def plot_all(self, block=True):
        """
        Plot all images

        block (bool): Wait for figure to be closed by user before moving on
        """
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

        plt.get_current_fig_manager().window.state('zoomed')
        plt.show(block=block)

    def export_image(self, var_str, filename):
        """
        Export JPG image

        var_str (string):   Name of image to export (get var with getattr)
        filename (string):  Output file name
        """
        im_to_export = getattr(self, var_str)
        im_to_export = (im_to_export*255).astype(np.uint8)
        im_to_export = Image.fromarray(im_to_export)
        im_to_export.save(filename + ".jpg")

    def mean_filter(self, size):
        """
        Mean filter calculation

        size (int):         Median Filter Kernel Size
        return (ndarray):   Filtered Image
        """
        self.mean_filter_size = size
        self.M = uniform_filter(self.ImGray, size=self.mean_filter_size)
        return self.M

    def rough_focus_map(self):
        """
        Create rough focus map by subtracting median filtered image from original image

        return (ndarray): Rough Focus Map
        """
        self.RFM = np.abs(self.ImGray-self.M)
        return self.RFM

    def accurate_focus_map(self, r, eps):
        """
        Create accurate focus map by using guided filter

        Go to guided_filter for more details
        return (ndarray): Rough Focus Map
        """
        self.AFM = guided_filter(self.ImGray, self.RFM, r, eps)
        return self.AFM


def box(img, r):
    """
    O(1) box filter

    img (ndarray):      2D image
    r (int):            Radius of box filter
    return (ndarray):   Box filter
    """
    (rows, cols) = img.shape[:2]
    imDst = np.zeros_like(img)

    tile = [1] * img.ndim
    tile[0] = r
    imCum = np.cumsum(img, 0)
    imDst[0:r+1, :] = imCum[r:2*r+1, :]
    imDst[r+1:rows-r, :] = imCum[2*r+1:rows, :] - imCum[0:rows-2*r-1, :]
    imDst[rows-r:rows, :] = np.tile(imCum[rows-1:rows, :], tile) - imCum[rows-2*r-1:rows-r-1, :]

    tile = [1] * img.ndim
    tile[1] = r
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r+1] = imCum[:, r:2*r+1]
    imDst[:, r+1:cols-r] = imCum[:, 2*r+1: cols] - imCum[:, 0: cols-2*r-1]
    imDst[:, cols-r: cols] = np.tile(imCum[:, cols-1:cols], tile) - imCum[:, cols-2*r-1: cols-r-1]

    return imDst


def guided_filter(i, p, r, eps):
    """
    Grayscale (fast) guided filter

    -----
    https://arxiv.org/pdf/1505.00996.pdf
    Fast Guided Filter, Kaiming He Jian Sun, Microsoft
    -----

    i (ndarray):        Guide Image (1 channel)
    p (ndarray):        Filter Input (1 channel)
    r (int):            Window Radius
    eps (float):        Regularization (roughly, allowable variance of non-edge noise)
    return (ndarray):   Guided Filter Image
    """
    (rows, cols) = i.shape
    n_box = box(np.ones([rows, cols]), r)

    mean_i = box(i, r) / n_box
    mean_p = box(p, r) / n_box

    corr_i = box(i * i, r) / n_box
    corr_ip = box(i * p, r) / n_box

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = box(a, r) / n_box
    mean_b = box(b, r) / n_box

    q = mean_a * i + mean_b
    return q

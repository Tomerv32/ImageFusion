from Util import *
import time


def example_full(path_1, path_2, is_path_1=True, is_path_2=True):
    fuse_obj = ImageFusion(path_1, path_2, is_path_1, is_path_2)
    fuse_obj.plot_compare("Im")

    fuse_obj.mean_filter(mean_filter_size)
    fuse_obj.plot_compare("M")

    fuse_obj.rough_focus_map()
    fuse_obj.plot_compare("RFM")

    fuse_obj.accurate_focus_map(gf_r, gf_eps)
    fuse_obj.plot_compare("AFM")

    fuse_obj.initial_decision_map()
    fuse_obj.plot_image("IDM", "Initial Decision Map", show=True)

    fuse_obj.initial_decision_map_bw_opening(bw_open_size)
    fuse_obj.plot_image("IDMbw", "Initial Decision Map (bw)", show=True)

    fuse_obj.final_decision_map(gf_r, gf_eps)
    fuse_obj.plot_image("FDM", "Final Decision Map", show=True)

    fuse_obj.image_fusion()
    fuse_obj.plot_image("IF", "Fused Image", show=True)

    # fuse_obj.export_image("IF", "IF")
    # fuse_obj.ImageToFuse1.export_image("Im", "Image1")
    # fuse_obj.ImageToFuse2.export_image("Im", "Image2")


def example_two(path_1, path_2, is_path_1=True, is_path_2=True):
    fuse_obj = ImageFusion(path_1, path_2, is_path_1, is_path_2)
    fuse_obj.image_fusion_run(mean_filter_size, gf_r, gf_eps, bw_open_size)
    fuse_obj.plot_all()


def example_multiple(imgs_path):
    import os
    file_list = os.listdir(imgs_path)

    curr_im = imgs_path + file_list[0]
    is_path = True

    for im in file_list[1:]:
        fuse_obj = ImageFusion(curr_im, imgs_path + im, is_path)
        fuse_obj.image_fusion_run(mean_filter_size, gf_r, gf_eps, bw_open_size)
        curr_im = fuse_obj.IF
        is_path = False

    fuse_obj.plot_image("IF", "Fused Image", show=True)


def main():     # Examples
    example_full("img/lytro-03-A.jpg", "img/lytro-03-B.jpg")
    time.sleep(time_delay)
    example_two("img/03desk1.bmp", "img/03desk2.bmp")
    time.sleep(time_delay)
    example_multiple(r"./img/Bug/")
    time.sleep(time_delay)
    example_two("img/test_1.jpg", "img/test_2.jpg")


if __name__ == '__main__':
    # Delay between examples
    time_delay = 3

    # Filters Params
    mean_filter_size = 7    # Mean Filter Kernel
    gf_r = 5                # Guided Filter Radius
    gf_eps = 0.3            # Guided Filter Epsilon
    bw_open_size = 20       # BW Opening Area Size

    main()

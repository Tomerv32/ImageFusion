from Util import *
import os


def example(path_1, path_2):
    fuse_obj = ImageFusion(path_1, path_2)
    fuse_obj.image_fusion_run(mean_filter_size, gf_r, gf_eps, bw_open_size)
    fuse_obj.plot_image("IF", show=True)


if __name__ == '__main__':
    # Filteres Params
    mean_filter_size = 7    # mean_filter
    gf_r = 5                # Guided Filter radius
    gf_eps = 0.3            # Guided Filter epsilon
    bw_open_size = 20       # BW opening

    # TODO:
    # Documentation should be great! verify it :)
    # Go over box and guided_filter functions

    # Step by step
    FusionImage = ImageFusion("img/lytro-03-A.jpg", "img/lytro-03-B.jpg")
    FusionImage.mean_filter(mean_filter_size)
    FusionImage.rough_foucs_map()
    FusionImage.accurate_foucs_map(gf_r, gf_eps)
    FusionImage.initial_decision_map()
    FusionImage.initial_decision_map_bw_opening(bw_open_size)
    FusionImage.final_decision_map(gf_r, gf_eps)
    FusionImage.image_fusion()
    # FusionImage.plot_all()

    # Example 2
    example("img/lytro-05-A.jpg", "img/lytro-05-B.jpg")

    # Example 3
    example("img/03desk1.bmp", "img/03desk2.bmp")

    # Example 4
    img_path = r"./img/Bug/"
    file_list = os.listdir(img_path)

    curr_im = img_path + file_list[0]
    is_path = True

    for im in file_list[1:]:
        FusionImage = ImageFusion(curr_im, img_path + im, is_path)
        FusionImage.image_fusion_run(mean_filter_size, gf_r, gf_eps, bw_open_size)
        curr_im = FusionImage.IF
        is_path = False

    FusionImage.plot_image("IF", show=True)

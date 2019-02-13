import sys
import cv2
import os
import glob
import argparse
import numpy as np
from multiprocess import pool

IMG_EXT = [".jpg", ".png", ".bmp", ".jpeg", ".tiff", ".tif"]
PARALLEL_IMG_SIZE = 100
CROP_MIN_NUM = 9
RANDOM_SEED = 65535
THREAD_NUM = 10

def crop_imgs(img_lst, crop_candidate_lst, crop_num, target_path, disp=False):
    np.random.seed(RANDOM_SEED)
    total_img_num = 0
    if disp:
        print("Start Processing: ", end='', flush=True)
    for img_path in img_lst:
        img = cv2.imread(img_path)
        if disp:
            print(". ", end='', flush=True)
        for crop_size in crop_candidate_lst:
            try:
                crop_width, crop_high = list(map(int, crop_size.split("*")))
                file_name = os.path.basename(img_path)
                # img shape [high, width, channel]
                if crop_width > img.shape[1] or crop_high > img.shape[0]:
                    # donot crop, just save it
                    if not os.path.exists(os.path.join(target_path,
                                                       "{}_noncrop{}".format(os.path.splitext(file_name)[0],
                                                                             os.path.splitext(file_name)[1]))):
                        cv2.imwrite(os.path.join(target_path,
                                                 "{}_noncrop{}".format(os.path.splitext(file_name)[0],
                                                                       os.path.splitext(file_name)[1])),
                                    img)
                        total_img_num += 1
                else:
                    cropped_num = 0
                    start_x = 0
                    start_y = 0
                    crop_all_img = False
                    row_num = 0
                    col_num = 0
                    while cropped_num < crop_num or not crop_all_img:
                        if start_x + crop_width <= img.shape[1] and start_y + crop_high <= img.shape[0]:
                            cropped = img[start_y:start_y+crop_high, start_x:start_x+crop_width]
                            start_x += crop_width
                            col_num += 1
                            new_file_name = os.path.join(target_path,
                                                 "{}_crop{:04d}_{:04d}_{:04d}_{:03d}_{:03d}{}".format(
                                                     os.path.splitext(file_name)[0],
                                                     crop_width,
                                                     crop_high,
                                                     cropped_num,
                                                     row_num,
                                                     col_num,
                                                     os.path.splitext(file_name)[1]))
                            if start_x + crop_width >= img.shape[1]:
                                start_x = 0
                                start_y += crop_high
                                row_num += 1
                                col_num = 0
                        else:
                            # cannot from left to right and up to down crop now, and also donot meet the min requriement,
                            # so we crop random
                            crop_all_img = True
                            x = np.random.randint(0, max(img.shape[1] - crop_width, 1))
                            y = np.random.randint(0, max(img.shape[0] - crop_high, 1))
                            cropped = img[y:y + crop_high, x:x + crop_width]
                            col_num = int(x/crop_width)
                            row_num = int(y/crop_high)
                            new_file_name = os.path.join(target_path,
                                                         "{}_crop{:04d}_{:04d}_{:04d}_{:03d}_{:03d}_rand{}".format(
                                                             os.path.splitext(file_name)[0],
                                                             crop_width,
                                                             crop_high,
                                                             cropped_num,
                                                             row_num,
                                                             col_num,
                                                             os.path.splitext(file_name)[1]))

                        cv2.imwrite(new_file_name, cropped)
                        cropped_num += 1
                    total_img_num += cropped_num

            except Exception as e:
                print("Wrong format crop candidate configuration: {} image shape {} [{}]".format(crop_size, img.shape, str(e)), flush=True)
                continue
    if disp:
        print(". ", flush=True)
    return total_img_num

def main(args):
    if not os.path.exists(args.srcImg):
        print("Path {} is not existed".format(args.srcImg))
        return 1
    if not os.path.exists(args.targetPath):
        os.mkdir(args.targetPath)
    if os.path.isfile(args.srcImg):
        img_list = [args.srcImg]
    else:
        img_list = glob.glob(os.path.join(args.srcImg, "*.*"))
        img_list = list(filter(lambda s: os.path.splitext(s)[1] in IMG_EXT, img_list))
    print("There are total {} images to be processed".format(len(img_list)))

    crop_candidate_lst = args.cropSizeList
    if len(img_list) > PARALLEL_IMG_SIZE:
        work_pool = pool.Pool(processes=THREAD_NUM)
        param_lst = []
        for cnt in range(int(np.ceil(len(img_list)/PARALLEL_IMG_SIZE))):
            param_lst.append((img_list[cnt*PARALLEL_IMG_SIZE:min(len(img_list), (cnt+1)*PARALLEL_IMG_SIZE)],
                              crop_candidate_lst,
                              CROP_MIN_NUM,
                              args.targetPath,
                              True if cnt == 0 else False))
        rst_obj = work_pool.starmap_async(crop_imgs, param_lst)
        while not rst_obj.ready():
            sys.stdout.flush()
            rst_obj.wait(5)
        total = sum(list(rst_obj.get()))
        work_pool.close()
    else:
        total = crop_imgs(img_list, crop_candidate_lst, CROP_MIN_NUM, args.targetPath)
    print("Total generated {} imgs in {}".format(total, args.targetPath))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cropSizeList',
        type=str,
        nargs='+',
        required=True,
        help='To be cropped img size which can be a list like 800*800, 500*500, 200*200, w*h',
    )
    parser.add_argument(
        '--srcImg',
        type=str,
        required=True,
        help='Image path which can be a single image or a data path. Supported type: jpg png bmp jpeg tiff',
    )
    parser.add_argument(
        '--targetPath',
        type=str,
        required=True,
        help='To be stored path',
    )

    args_flag, unparsed = parser.parse_known_args()
    main(args_flag)
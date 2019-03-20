import cv2
import numpy as np
import imutils
from collections import namedtuple
import copy
import json

Resolution = namedtuple("Resolution", ("high", "width"))
front_imgs = ["t1.png", "t2.png", "t3.png", "t4.png"]
background_imgs = ["bg.png"]
sample_size = Resolution(3024, 4032)  # high, width
sample_num_per_img = 4
max_buffer = 5
max_angle = 5


def find_available_pos(img, filled_lst, background_size):
    global  max_buffer
    current_buffer = np.random.choice(range(max_buffer))
    start_x = int(sample_size.width * 0.02)
    start_y = int(sample_size.high * 0.02)
    # filled_lst shape [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]
    if len(filled_lst) == 0:
        filled_lst.append([start_x, start_y, start_x + np.shape(img)[1], start_y + np.shape(img)[0]])
        return start_x, start_y, filled_lst
    else:
        count = 0
        last_y = 0
        while count < 100:
            count += 1
            current_y = start_y
            candidate_lst = np.array(filled_lst)[np.where((np.array(filled_lst)[:, 1] <= current_y) & (np.array(filled_lst)[:, 1] > last_y))[0]]
            if len(candidate_lst) == 0:
                #print("{} {} {}".format(filled_lst, current_y, last_y))
                suggest_x = start_x
            else:
                suggest_x = candidate_lst[np.argsort(candidate_lst, axis=0)[::-1][:, 2][0]][2] + current_buffer
            suggest_y = current_y
            if suggest_x + np.shape(img)[1] <= background_size[1] and \
                    suggest_y + np.shape(img)[0] <= background_size[0]:
                filled_lst.append([suggest_x, suggest_y, suggest_x + np.shape(img)[1], suggest_y + np.shape(img)[0]])
                return suggest_x, suggest_y, filled_lst
            else:
                if len(candidate_lst) > 0:
                    current_y = candidate_lst[np.argsort(candidate_lst, axis=0)[:, 3][0]][3]
                candidate_lst = np.array(filled_lst)[np.where(np.array(filled_lst)[:, 1] >= current_y)[0]]
                if len(candidate_lst) == 0:
                    tmp_corrdinate = filled_lst[np.argsort(filled_lst, axis=0)[:, 3][0]]
                    suggest_x = tmp_corrdinate[0]
                    suggest_y = tmp_corrdinate[3] + current_buffer
                    if suggest_y < start_y: # shall not happen
                        continue
                    if suggest_x + np.shape(img)[1] <= background_size[1] and \
                            suggest_y + np.shape(img)[0] <= background_size[0]:
                        filled_lst.append(
                            [suggest_x, suggest_y, suggest_x + np.shape(img)[1], suggest_y + np.shape(img)[0]])
                        return suggest_x, suggest_y, filled_lst
                    else:
                        last_y = start_y
                        start_y = tmp_corrdinate[3] + current_buffer
                else:
                    last_y = start_y
                    start_y = candidate_lst[np.argsort(candidate_lst, axis=0)[:, 1][0]][1]
        if count == 100:
            print("Cannot find appropriate position. Algorithm may have bug {}".format(filled_lst))
            return None, None, filled_lst


def rotate_image(mat, background, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    new_width = int(np.shape(mat)[1] * abs(np.cos(np.radians(angle))) + np.shape(mat)[0] * abs(np.sin(np.radians(angle))))
    new_high = int(np.shape(mat)[1] * abs(np.sin(np.radians(angle))) + np.shape(mat)[0] * abs(np.cos(np.radians(angle))))
    background = cv2.resize(background, (int(new_width*1.2), int(new_high*1.2)), interpolation = cv2.INTER_AREA)
    #print("{}: {} vs {} ({} {})".format(angle, np.shape(mat), np.shape(background), new_width, new_high))
    height, width = mat.shape[:2] # image shape has 3 dimensions
    bg_height, bg_width = background.shape[:2]  # image shape has 3 dimensions
    bg_image_center = (bg_width / 2, bg_height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    fill_y = max(int(bg_image_center[1]-int(height/2)), 0)
    fill_x = max(int(bg_image_center[0]-int(width/2)), 0)

    background[fill_y:fill_y+height, fill_x:fill_x+width] = mat
    background = imutils.rotate(background, angle)

    (cX, cY) = (bg_width / 2, bg_height / 2)
    # rotate our image by 45 degrees
    M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
    background = cv2.warpAffine(background, M, (bg_width, bg_height))
    #print("{} {} {} {}".format(fill_x, int(np.shape(mat)[0] * abs(np.sin(np.radians(angle)))/2),
    #                                fill_y, int(np.shape(mat)[1] * abs(np.sin(np.radians(angle)))/2)))
    cropped_x = max(fill_x - int(np.shape(mat)[0] * abs(np.sin(np.radians(angle)))/2), 0)
    cropped_y = max(fill_y - int(np.shape(mat)[1] * abs(np.sin(np.radians(angle)))/2), 0)
    cropped = background[cropped_y:cropped_y+new_high, cropped_x:cropped_x+new_width]
    #print(np.shape(cropped))
    #cv2.imshow("Image", cropped)
    #cv2.imshow("background", background)
    #cv2.waitKey(0)
    return cropped

#def update_available_pair()

# My assumption is that all front imgs shall be original size and it means that the size shall reflect the true size in real world
sample_lst = np.random.choice(front_imgs, sample_num_per_img)
print("sample_lst: {}".format(sample_lst))
images = []
shape_lst = []
horizon_lst = []
vertical_lst = []
total_area = 0

for img_name in sample_lst:
    images.append(cv2.imread(img_name))
    shape_lst.append([np.shape(images[-1])[0], np.shape(images[-1])[1]])
    total_area += np.shape(images[-1])[0] * np.shape(images[-1])[1]
    if np.shape(images[-1])[0] > np.shape(images[-1])[1]:
        vertical_lst.append(len(images) - 1)
    else:
        horizon_lst.append(len(images) -1)
shape_lst = np.array(shape_lst)
ratio = min(sample_size.high/float(sum(np.argsort(shape_lst[:, 0])[::-1][:2])),
            sample_size.width/float(sum(np.argsort(shape_lst[:, 1])[::-1][:2])), 1,
            float(sample_size.width*sample_size.high)/float(total_area))

print("Change ratio: {}".format(ratio))
background_img_name = np.random.choice(background_imgs, 1)[0]
background_img_original = cv2.imread(background_img_name)
background_img = cv2.resize(background_img_original, (sample_size.width, sample_size.high))

filled_lst = []
json_file = {}
# start from max area of horizon
sorted_horizon_lst = []
for idx in horizon_lst:
    sorted_horizon_lst.append(np.shape(images[idx])[1])
sorted_horizon_lst = np.argsort(sorted_horizon_lst)[::-1]
for image_idx in sorted_horizon_lst:
    img = images[horizon_lst[image_idx]]
    print("horizon: {}".format(sample_lst[horizon_lst[image_idx]]))
    # resize the image
    img = imutils.resize(img, width=int(np.shape(img)[1]*ratio))
    # rotate the image
    degree = np.random.choice(range(-max_angle, max_angle))

    img = rotate_image(img, background_img_original.copy(), degree)
    available_x, available_y, filled_lst = find_available_pos(img, filled_lst, np.shape(background_img))
    if available_x is not None:
        background_img[available_y:available_y + np.shape(img)[0], available_x:available_x + np.shape(img)[1]] = img
        json_file[sample_lst[horizon_lst[image_idx]]] = {
            "ratio": float(ratio),
            "degree": int(degree),
            "xmin": int(available_x),
            "ymin": int(available_y),
            "xmax": int(available_x + np.shape(img)[1]),
            "ymax": int(available_y + np.shape(img)[0])
        }
        cv2.imshow("horizon", background_img)
        cv2.waitKey(0)
    else:
        continue

sorted_vertical_lst = []
for idx in vertical_lst:
    sorted_vertical_lst.append(np.shape(images[idx])[0])
sorted_vertical_lst = np.argsort(sorted_vertical_lst)[::-1]
for image_idx in sorted_vertical_lst:
    img = images[vertical_lst[image_idx]]
    print("vertical: {} {}".format(sample_lst[vertical_lst[image_idx]], np.shape(img)))
    # resize the image
    img = imutils.resize(img, width=int(np.shape(img)[1] * ratio))
    # rotate the image
    degree = np.random.choice(range(-max_angle, max_angle))

    img = rotate_image(img, background_img_original.copy(), degree)
    available_x, available_y, filled_lst = find_available_pos(img, filled_lst, np.shape(background_img))
    if available_x is not None:
        background_img[available_y:available_y + np.shape(img)[0], available_x:available_x + np.shape(img)[1]] = img
        json_file[sample_lst[vertical_lst[image_idx]]] = {
            "ratio": float(ratio),
            "degree": int(degree),
            "xmin": int(available_x),
            "ymin": int(available_y),
            "xmax": int(available_x + np.shape(img)[1]),
            "ymax": int(available_y + np.shape(img)[0])
        }
        cv2.imshow("vertical", background_img)
        cv2.waitKey(0)
    else:
        continue

cv2.imwrite("output.png", background_img)
with open("output.json", "w") as f:
    json.dump(json_file, f, indent=2)



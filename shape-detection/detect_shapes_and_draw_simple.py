import cv2
import numpy as np
import imutils


def draw_contours(img, cnts):  # conts = contours
    img = np.copy(img)
    img = cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    return img


def draw_min_rect_circle(img, cnts):  # conts = contours
    img = np.copy(img)

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)

        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue

        min_rect = cv2.minAreaRect(cnt)  # min_area_rectangle
        min_rect = np.int0(cv2.boxPoints(min_rect))
        cv2.drawContours(img, [min_rect], 0, (0, 255, 0), 2)  # green
        """
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center, radius = (int(x), int(y)), int(radius)  # center and radius of minimum enclosing circle
        img = cv2.circle(img, center, radius, (0, 0, 255), 2)  # red
        """
    return img


def draw_approx_hull_polygon(img, cnts, get_hulls=False):
    img = np.copy(img)
    #img = np.zeros(img.shape, dtype=np.uint8)

    #cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)  # blue
    min_side_len = img.shape[0] / 32  # 多边形边长的最小值 the minimum side length of polygon
    min_poly_len = img.shape[0] / 16  # 多边形周长的最小值 the minimum round length of polygon
    min_side_num = 3  # 多边形边数的最小值
    approxs = [cv2.approxPolyDP(cnt, min_side_len, True) for cnt in cnts]  # 以最小边长为限制画出多边形
    approxs = [approx for approx in approxs if cv2.arcLength(approx, True) > min_poly_len]  # 筛选出周长大于 min_poly_len 的多边形
    approxs = [approx for approx in approxs if len(approx) > min_side_num]  # 筛选出边长数大于 min_side_num 的多边形
    # Above codes are written separately for the convenience of presentation.
    #cv2.polylines(img, approxs, True, (0, 255, 0), 2)  # green

    hulls = [cv2.convexHull(cnt) for cnt in cnts]

    cv2.polylines(img, hulls, True, (0, 0, 255), 2)  # red


    # for cnt in cnts:
    #     cv2.drawContours(img, [cnt, ], -1, (255, 0, 0), 2)  # blue
    #
    #     epsilon = 0.02 * cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
    #     cv2.polylines(img, [approx, ], True, (0, 255, 0), 2)  # green
    #
    #     hull = cv2.convexHull(cnt)
    #     cv2.polylines(img, [hull, ], True, (0, 0, 255), 2)  # red
    if get_hulls:
        return img, hulls
    else:
        return img


def generate_background_lst(xmin, ymin, xmax, ymax, img):
    background_lst = []
    for row in range(np.shape(img)[0]):
        for column in range(np.shape(img)[1]):
            if row < ymin or row > ymax or column < xmin or column > xmax:
                if len(background_lst) > 0:
                    if not np.equal(background_lst, img[row, column]).all(axis=1).any():
                        background_lst.append(img[row, column])
                else:
                    background_lst.append(img[row, column])
    print(np.shape(background_lst))
    return background_lst


def reset_image(background_lst, img):
    img = np.copy(img)
    for background in background_lst:
        rowlst, collst = np.where(np.equal(img, background).all(axis=2))
        img[rowlst, collst] = [231, 12, 12]
    return img


def reset_image_by_location(xmin, ymin, xmax, ymax, img):
    img = np.copy(img)
    img[:ymin, :] = [231, 12, 12]
    img[ymax:, :] = [231, 12, 12]
    img[:, :xmin] = [231, 12, 12]
    img[:, xmax:] = [231, 12, 12]
    return img

def run():
    image = cv2.imread('output.png')  # a black objects on white image is better
    image = imutils.resize(image, width=640)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
    """
    cv2.imshow("adaptiveThreshold", thresh)
    #cv2.waitKey(0)
    lower = np.mean(gray)*0.66
    upper = np.mean(gray)*1.33
    print("mean: {} {}".format(lower, upper))
    thresh = cv2.Canny(gray, lower, upper, L2gradient=True)
    cv2.imshow("Canny_mean", thresh)

    lower = np.median(gray) * 0.66
    upper = np.median(gray) * 1.33
    print("median: {} {}".format(lower, upper))
    thresh = cv2.Canny(gray, lower, upper, L2gradient=True)
    cv2.imshow("Canny_median", thresh)

    ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    lower = ret * 0.28
    upper = ret * 0.48
    print("THRESH_OTSU: {} {}".format(lower, upper))
    thresh = cv2.Canny(gray, lower, upper, L2gradient=True)
    cv2.imshow("THRESH_OTSU_gray", thresh)

    cv2.waitKey(0)
    exit(1)
    #"""

    ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    lower = ret * 0.5
    upper = ret
    print("THRESH_OTSU: {} {}".format(lower, upper))
    thresh = cv2.Canny(gray, lower, upper, L2gradient=True)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new_contours = []

    for cnt in contours:
        first_loop = []
        for ccnt in cnt:
            if ccnt[0][0] < 50 or ccnt[0][1] < 50 or \
                    np.shape(image)[0] - ccnt[0][1] < 50 or \
                    np.shape(image)[1] - ccnt[0][0] < 50:
                continue
            else:
                first_loop.append(ccnt)
        if len(first_loop) > 0:
            new_contours.append(np.array(first_loop))

    imgs = [
        image, thresh,
        draw_min_rect_circle(image, new_contours),
        draw_approx_hull_polygon(image, new_contours),
    ]

    for idx, img in enumerate(imgs):
        # cv2.imwrite("%s.jpg" % id(img), img)
        cv2.imshow("contours_{}".format(idx), img)
    cv2.waitKey(0)

    new_contours = []
    for cnt in contours:
        for ccnt in cnt:
            if ccnt[0][0] < 50 or ccnt[0][1] < 50 or \
                    np.shape(image)[0] - ccnt[0][1] < 50 or \
                    np.shape(image)[1] - ccnt[0][0] < 50:
                continue
            else:
                new_contours.append(ccnt)
    new_contours = [np.array(new_contours)]

    new_img, hulls = draw_approx_hull_polygon(image, new_contours, get_hulls=True)
    print(np.shape(hulls))
    min_x = np.min(hulls[0][:, 0, 0])
    max_x = np.max(hulls[0][:, 0, 0])
    min_y = np.min(hulls[0][:, 0, 1])
    max_y = np.max(hulls[0][:, 0, 1])
    print(np.shape(new_img))
    #background_lst = generate_background_lst(min_x, min_y, max_x, max_y, image)
    #new_image = reset_image(background_lst, image)
    new_image = reset_image_by_location(min_x, min_y, max_x, max_y, image)


    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
    cv2.imshow("new_image", thresh)
    cv2.waitKey(0)

    lines = cv2.HoughLines(thresh, rho=1, theta=np.pi / 180, threshold=80,
                           min_theta=0)  # , max_theta=40) np.pi / 180
    # lines = cv2.HoughLines(changed_img, 1, np.pi / 180, 150, None, 0, 0)
    print(len(lines))
    painted_lines = []
    count = 0
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            same_line = False
            for existed_line in painted_lines:
                if rho - 20 < existed_line[0] < rho + 20 and theta - 5 < existed_line[1] < theta + 5:
                    same_line = True
                    break
            if same_line:
                painted_lines.append([rho, theta])
                continue
            painted_lines.append([rho, theta])
            count += 1
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(image, pt1, pt2, (0, 0, 255), 3)

    cv2.imshow("lines", image)
    cv2.waitKey(0)
    print(count)
    print(painted_lines)


if __name__ == '__main__':
    run()
pass
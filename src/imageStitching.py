import copy
import os

import cv2
import numpy as np


def readImages(path):
    renameFiles(path)
    imgs = []
    for i in range(3):
        image_temp = cv2.imread(path + str(i + 1) + ".jpg")
        imgs.append(image_temp)
    num = len(imgs)
    aspect_ratio = imgs[0].shape[0] / imgs[0].shape[1]
    scale = 1280 / (num * imgs[0].shape[1])
    w = int(imgs[0].shape[1] * scale)
    h = int(aspect_ratio * w)
    images = [cv2.resize(i, (w, h)) for i in imgs]
    return images


def stitchImgSet(img_set):
    gr = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img_set]
    pts = [cv2.goodFeaturesToTrack(i, 10000, 0.001, 6) for i in gr]
    pts = [np.int0(i) for i in pts]

    imgpts = copy.deepcopy(img_set)
    for i in range(len(imgpts)):
        img = imgpts[i]
        goodpts = pts[i]
        for j in goodpts:
            x, y = j.ravel()
            cv2.circle(img, (x, y), 3, 255, -1)

    NMSimgs = copy.deepcopy(img_set)

    best_features = []
    for i in range(len(NMSimgs)):
        best_features.append(NMS(pts[i], 1100))

    for i in range(len(best_features)):
        for j in best_features[i]:
            cv2.circle(NMSimgs[i], (j[0], j[1]), 3, 255, -1)

    D = []
    for i in range(len(gr)):
        D.append(get_pts(gr[i], best_features[i], 40))

    set1 = []
    for i in range(len(D) - 1):
        set1.append(match_pts(D[i], D[i + 1]))

    best1, best2, inliers = RANSAC_2(best_features[0], best_features[1], set1[0])
    if inliers >= 6:
        drawMatch(best1, img_set[0], best2, img_set[1])
        H = EstimateHomography(best1, best2)
        Imgstitch = warpAndStitch(img_set[0], best1, img_set[1], best2, H)
    else:
        H = EstimateHomography(best1, best2)
        Imgstitch = img_set[0]
    return Imgstitch, H


def NMS(feat, pts):
    best = []
    score = []
    for i in range(len(feat)):
        if i == 0:
            best.append([feat[i][0][0], feat[i][0][1]])
            continue
        dist = feat[:i] - feat[i][0]
        dist = dist ** 2
        dist = np.sum(dist, axis=2)
        min_dist = min(dist)
        index = np.where(min_dist == dist)
        if [feat[index[0][0]][0][0], feat[index[0][0]][0][1]] in best:
            continue
        best.append([int(feat[index[0][0]][0][0]), int(feat[index[0][0]][0][1])])
        score.append(min_dist[0])
    score = np.array(score)
    index = np.argsort(score)
    best = np.array(best)
    best = best[index]
    return best[:pts]


def get_pts(image, points, window_size):
    good_pts = []
    for i in points:
        diff_lowery = 0
        diff_lowerx = 0
        lowery = i[1] - int(window_size / 2)
        if lowery < 0:
            lowery = i[1] + abs(lowery) - int(window_size / 2)
            diff_lowery = abs(i[1] - int(window_size / 2))

        uppery = i[1] + int(window_size / 2) + diff_lowery
        if uppery > image.shape[0]:
            diff = uppery - image.shape[0]
            uppery = i[1] - diff + int(window_size / 2)
            lowery = lowery - diff

        lowerx = i[0] - int(window_size / 2)
        if lowerx < 0:
            lowerx = i[0] + abs(lowerx) - int(window_size / 2)
            diff_lowerx = abs(i[0] - int(window_size / 2))

        upperx = i[0] + int(window_size / 2) + diff_lowerx
        if upperx > image.shape[1]:
            diff = upperx - image.shape[1]
            upperx = i[0] - diff + int(window_size / 2)
            lowerx = lowerx - diff

        patch = image[lowery:uppery, lowerx:upperx]
        blur = cv2.GaussianBlur(patch, (5, 5), 0)
        good_pt = cv2.resize(blur, (8, 8))
        good_pt = np.reshape(good_pt, (good_pt.shape[0] * good_pt.shape[1]))
        mean = np.sum(good_pt) / good_pt.shape[0]
        std = ((1 / good_pt.shape[0]) * (np.sum((good_pt - mean) ** 2))) ** (1 / 2)
        good_pt = (good_pt - (mean)) / std
        good_pts.append(good_pt)
    return good_pts


def match_pts(D1, D2):
    set1 = []
    D1 = np.array(D1, dtype=np.int64)
    D2 = np.array(D2, dtype=np.int64)
    for i in range(len(D1)):
        ssd = D2 - np.reshape(D1[i], (1, D1.shape[1]))
        ssd = ssd ** 2
        ssd = np.sum(ssd, axis=1)
        minSSD = np.min(ssd)
        if minSSD > 15:
            continue
        index = np.where(ssd == minSSD)
        set1.append([i, index[0][0]])
    return set1


def ssd(match_right, nw_right):
    ssd = match_right - nw_right
    ssd = ssd ** 2
    ssd = np.sum(ssd, axis=1)
    inlier = np.where(ssd < 2)
    return inlier


def check_match(pts):
    check = 1
    for i in range(len(pts)):
        point = np.reshape(pts[i], (1, pts.shape[1]))
        ssd = pts - point
        ssd = ssd ** 2
        ssd = np.sum(ssd, axis=1)
        ind = np.where(ssd < 2)
        if len(ind[0]) > 1:
            check = 0
            break
    return check


def RANSAC_2(points1, points2, pairs):
    N = 15000
    best_score = None
    pairs = np.array(pairs)
    ind = np.arange(pairs.shape[0])
    matched_left = points1[pairs[:, 0]]
    matched_right = points2[pairs[:, 1]]
    counter = 0
    while True:
        count = 0
        while N > count:
            random_ind = np.random.choice(ind, size=4, replace=False)
            left = matched_left[random_ind]
            right = matched_right[random_ind]
            H = cv2.getPerspectiveTransform(np.float32(left), np.float32(right))
            tmp_left = np.concatenate((matched_left, np.ones((matched_left.shape[0], 1))), axis=1)
            new_right = np.dot(H, tmp_left.T)
            new_right[-1, :] = new_right[-1, :] + 0.0001
            new_right = new_right / (new_right[-1, :])
            new_right = new_right.T
            check = check_match(new_right[:, :2])
            if check == 0:
                count += 0.5
                continue
            inliers = ssd(matched_right, new_right[:, :2])
            num_inliers = len(inliers[0])
            if best_score == None or best_score < num_inliers:
                best_score = num_inliers
                best_points1 = matched_left[inliers]
                best_points2 = matched_right[inliers]
            count += 1

        counter += 1
        if best_score > 6 or counter > 5:
            break
    return best_points1, best_points2, best_score


def drawMatch(points1, img1, points2, img2):
    max_height = np.max([img1.shape[0], img2.shape[0]])
    tmp_img1 = np.zeros((max_height, img1.shape[1], img1.shape[2]), dtype=np.uint8)
    tmp_img2 = np.zeros((max_height, img2.shape[1], img1.shape[2]), dtype=np.uint8)
    tmp_img1[:img1.shape[0], :img1.shape[1]] = img1
    tmp_img2[:img2.shape[0], :img2.shape[1]] = img2
    joint_imgs = np.concatenate((tmp_img1, tmp_img2), axis=1)

    for i in range(len(points1)):
        start = (points1[i, 0], points1[i, 1])
        end = (points2[i, 0] + img1.shape[1], points2[i, 1])
        cv2.line(joint_imgs, start, end, (0, 255, 255), 1)
    return


def EstimateHomography(src, dst):
    A = np.zeros((2 * len(src), 9))
    i = 0
    for a in range(len(A)):
        if a % 2 == 0:
            A[a, :] = [src[i][0], src[i][1], 1, 0, 0, 0, -(dst[i][0] * src[i][0]), -(dst[i][0] * src[i][1]), -dst[i][0]]
        else:
            A[a, :] = [0, 0, 0, src[i][0], src[i][1], 1, -(dst[i][1] * src[i][0]), -(dst[i][1] * src[i][1]), -dst[i][1]]
            i += 1

    U, sigma, V = np.linalg.svd(A)
    Vt = V.T
    h = Vt[:, 8] / Vt[8][8]
    H = np.reshape(h, (3, 3))
    return H


def getNewDimensions(img2, H):
    point = np.array([[0, 0, 1], [0, img2.shape[0], 1], [img2.shape[1], img2.shape[0], 1], [img2.shape[1], 0, 1]])
    border = np.dot(H, point.T)
    border = border / border[-1]
    col_min = np.min(border[0, :])
    col_max = np.max(border[0, :])
    row_min = np.min(border[1, :])
    row_max = np.max(border[1, :])
    if col_min < 0:
        new_width = round(col_max - col_min)
    else:
        new_width = round(col_max - col_min)

    if row_min < 0:
        new_height = round(row_max - row_min)
    else:
        new_height = round(row_max - row_min)
    shift = np.array([[1, 0, -col_min], [0, 1, -row_min], [0, 0, 1]])
    H = np.dot(shift, H)
    return new_height, new_width, H


def EstimateTranslation(points1, points2, H):
    onet = np.ones((points2.shape[0], 1))
    point = np.concatenate((points2, onet), axis=1)
    point = point.T
    transformed_point = np.dot(H, point)
    transformed_point = transformed_point / transformed_point[-1]
    points1 = points1.T
    translations = points1 - transformed_point[:2]
    translations = translations.T

    translations_sum = np.sum(translations, axis=0)
    translations_mean = translations_sum / translations.shape[0]
    translation = np.array([[translations_mean[0]], [translations_mean[1]]])
    return translation


def stitch(stitch_this, stitch_to):
    to_shape = stitch_to.shape
    this_shape = stitch_this.shape
    shape = np.array([to_shape, this_shape])
    stitched_img = np.zeros((np.max(shape[:, 0]), np.max(shape[:, 1]), 3), dtype=np.uint8)
    stitched_img[:to_shape[0], :to_shape[1]] = stitch_to
    ind = np.where(stitch_this > 0)
    stitched_img[ind[:2]] = stitch_this[ind[:2]]
    return stitched_img


def warpAndStitch(img, points1, img2, points2, H):
    H = np.linalg.inv(H)
    new_height, new_width, H = getNewDimensions(img2, H)
    stitch_this = cv2.warpPerspective(img2, H, (new_width, new_height))

    translation = EstimateTranslation(points1, points2, H)

    if translation[0, 0] < 0 and translation[1, 0] < 0:
        M = np.float32([[1, 0, abs(round(translation[0, 0]))], [0, 1, abs(round(translation[1, 0]))]])
        translated_shape = (abs(round(translation[0, 0])) + img.shape[1], abs(round(translation[1, 0])) + img.shape[0])
        img = cv2.warpAffine(img, M, translated_shape)
        stitched_img = stitch(img, stitch_this)
    elif translation[0, 0] < 0 and translation[1, 0] > 0:
        M = np.float32([[1, 0, abs(round(translation[0, 0]))], [0, 1, 0]])
        translated_shape = (abs(round(translation[0, 0])) + img.shape[1], img.shape[0])
        img = cv2.warpAffine(img, M, translated_shape)

        M = np.float32([[1, 0, 0], [0, 1, abs(round(translation[1, 0]))]])
        translated_shape = (stitch_this.shape[1], abs(round(translation[1, 0])) + stitch_this.shape[0])
        stitch_this = cv2.warpAffine(stitch_this, M, translated_shape)
        stitched_img = stitch(img, stitch_this)
    elif translation[0, 0] > 0 and translation[1, 0] < 0:
        M = np.float32([[1, 0, abs(round(translation[0, 0]))], [0, 1, 0]])
        translated_shape = (abs(round(translation[0, 0])) + stitch_this.shape[1], stitch_this.shape[0])
        stitch_this = cv2.warpAffine(stitch_this, M, translated_shape)

        M = np.float32([[1, 0, 0], [0, 1, abs(round(translation[1, 0]))]])
        translated_shape = (img.shape[1], abs(round(translation[1, 0])) + img.shape[0])
        img = cv2.warpAffine(img, M, translated_shape)
        stitched_img = stitch(img, stitch_this)
    else:
        M = np.float32([[1, 0, round(translation[0, 0])], [0, 1, round(translation[1, 0])]])
        translated_shape = (
            abs(round(translation[0, 0])) + stitch_this.shape[1], abs(round(translation[1, 0])) + stitch_this.shape[0])
        stitch_this = cv2.warpAffine(stitch_this, M, translated_shape)
        stitched_img = stitch(img, stitch_this)
    return stitched_img


def ImageStitch(path):
    images = readImages(path)
    base_idx = int((len(images) / 2))
    tmp = [images[base_idx], images[base_idx - 1]]
    all_H = []
    count = 1
    for i in range(1, len(images)):
        Imgstitch, H = stitchImgSet(tmp)
        all_H.append(H)
        if i % 2 == 0:
            tmp = [Imgstitch, images[base_idx - count]]
        else:
            tmp = [Imgstitch, images[base_idx + count]]
            count += 1
    return Imgstitch, all_H


# def getFileNames(path):
def getListofFiles(dir):
    inputs = []

    for path in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, path)):
            inputs.append(path)
    return inputs


def renameFiles(dir):
    inputs = getListofFiles(dir)
    goal = ['1.jpg', '2.jpg', '3.jpg']
    flag = False
    if(all(x in inputs for x in goal)):
        flag = True
    if not flag:
        if len(inputs) != 3:
            print("Code works for 3 input images, taking first 3 images to stitch")
        for i in range(3):
            old = dir + inputs[i]
            new = dir + str(i + 1) + ".jpg"
            os.rename(old, new)


hill = "../input/hill/"
tv = "../input/tv/"

out = "../output/"

choice = input("Select from the following:\n"
               "1. Hill\n"
               "2. TV\n"
               "3. Custom\n")

try:
    int(choice)
except ValueError:
    print("Incorrect selection! Please choose a number")
    exit()

if int(choice) == 1:
    path = hill
    save = "hill"
elif int(choice) == 2:
    path = tv
    save = "tv"
elif int(choice) == 3:
    path = input("Enter your custom local path: ")
    save = "custom"

else:
    print("Incorrect selection! Please choose from 1 to 3")
    exit()

if path == "":
  path = "../input/custom/"

if os.path.exists(path):
    stitched, H = ImageStitch(path)
    # for i in range(len(H)):
    #     print("Homography matrix " + str(i + 1) + ":\n", H[i])
    cv2.imshow("Panorama", stitched)
    cv2.imwrite(out + "stitched_" + save + ".jpg", stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Your path: " + "'" + path + "'" + "does not exist. Please re-enter")


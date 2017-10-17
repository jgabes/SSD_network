import os
import pickle
from math import sqrt, floor

import cv2
import numpy as np
from matplotlib import pyplot as plt


class BBox:
    def __init__(self, col_min, row_min, col_max, row_max, img_col, img_row):
        self.r_min = max(0, min(1, row_min * 1.0 / img_row))
        self.c_min = max(0, min(1, col_min * 1.0 / img_col))
        self.r_max = max(0, min(1, row_max * 1.0 / img_row))
        self.c_max = max(0, min(1, col_max * 1.0 / img_col))

    def get_box(self, img_row, img_col):
        a = max(0, min(img_row, int(round(self.r_min * img_row))))
        b = max(0, min(img_col, int(round(self.c_min * img_col))))
        c = max(0, min(img_row, int(round(self.r_max * img_row))))
        d = max(0, min(img_col, int(round(self.c_max * img_col))))

        return ((b, a), (d, c))


feature_scales = [8, 4, 2, 1]
m = len(feature_scales)  # number of scales
box_ratios = [3, 2, 1, 1.1, 1 / 2, 1 / 3, ]
# box_ratios = [1]

smin = 0.2
smax = 0.9
k = range(1, m + 1)
# k = [1, 2, 3, 4] ##(0,m]
scale_factors = [smin + (smax - smin) / (m - 1) * (k_i - 1) for k_i in k]
scale_factor_diff = scale_factors[1] - scale_factors[0]
im_shape = 300


# scale = [i for i in reversed(scale)]

# calculation of anchor box width and height as reported by paper
def int_cast_tuple(a):
    if type(a[0]) is tuple:  # list or tuples of tuples
        return tuple(map(tuple, np.asanyarray(a, dtype=int)))
    else:
        return (int(round(a[0])), int(round(a[1])))


def clip_tuple(tup, my_min, my_max):
    return (max(my_min, min(my_max, tup[0])),
            max(my_min, min(my_max, tup[1])))


def tuple_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def tuple_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def bbox_center(a):
    return ((a[0][0] + a[1][0]) / 2, (a[0][1] + a[1][1]) / 2)


def aboxDims(my_scale, ratio, fm):
    if ratio == 1.1:
        my_scale = my_scale + scale_factor_diff
        ratio = 1
    w = my_scale * fm * np.sqrt(ratio)
    h = my_scale * fm / np.sqrt(ratio)
    return w, h


# Given a square image, where will your anchor box centers be
def centers(im_size, fm):
    wid = im_size / fm
    cs = []
    for row in [wid / 2 + k * wid for k in range(fm)]:
        for col in [wid / 2 + k * wid for k in range(fm)]:
            cs.append((row, col))
    return cs


# Input a location, output a box
def featureBox(col, row, fm, scale, ratio):
    w, h = aboxDims(scale, ratio, fm)
    up_l = (int(round(col - w / 2)), int(round(row - h / 2)))
    low_r = (int(round(col + w / 2)), int(round(row + h / 2)))

    return up_l, low_r


# lil graphing function because matplotlib can't do symbols
def drawBox(im, pt, l):
    cv2.line(im, (pt[0] - l, pt[1]), (pt[0] + l, pt[1]), (0, 255, 0))
    cv2.line(im, (pt[0], pt[1] - l), (pt[0], pt[1] + l), (0, 255, 0))


# Given two bounding boxes (not the class : p), give the Jacard Distance
def jacardDist(bb1, bb2):
    overlap_pixels = []
    up_l = bb1[0]
    low_r = bb1[1]
    jac_map = {}
    for i in range(up_l[0], low_r[0]):
        for j in range(up_l[1], low_r[1]):
            jac_map[(i, j)] = 1;
    size_box_1 = abs(up_l[0] - low_r[0]) * abs(up_l[1] - low_r[1])

    up_l = bb2[0]
    low_r = bb2[1]
    overlap_count = 0
    for i in range(up_l[0], low_r[0]):
        for j in range(up_l[1], low_r[1]):
            if (jac_map.__contains__((i, j))):
                overlap_pixels.append((j, i))
                overlap_count += 1
    size_box_2 = abs(up_l[0] - low_r[0]) * abs(up_l[1] - low_r[1])

    AUB = size_box_1 + size_box_2 - overlap_count
    AINTB = overlap_count
    jac_ov = AINTB / AUB
    return jac_ov, overlap_pixels, size_box_1, size_box_2


# Visualise Jacrd distance
def visualizeJacardDistance():
    im = np.zeros(shape=[100, 200, 3], dtype=np.uint8)
    box1 = ((10, 50), (150, 60))
    box2 = ((50, 25), (75, 80))
    cv2.rectangle(im, box1[0], box1[1], (255, 0, 0))
    cv2.rectangle(im, box2[0], box2[1], (0, 0, 255))
    jac_ov, overlap_pixels, sb1, sb2 = jacardDist(box1, box2)
    for pixel in overlap_pixels:
        im[pixel[0], pixel[1], 1] = 255
    f3 = plt.figure(3)
    plt.imshow(im)
    my_title = 'Jacard Overlap = {0:.2f}'.format(jac_ov)
    plt.title(my_title)
    plt.pause(0.05)
    pass


# visualizeJacardDistance()


def loadShowImages():
    this_scale = 2
    # p = pickle.load(open('236.p', 'rb'))
    p = pickle.load(open('241.p', 'rb'))
    im = p['old_pic']
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    logoss = p['logos']
    smiless = p['smiles']
    for logos in logoss:
        bbl = logos.get_box(im.shape[0], im.shape[1])
        cv2.rectangle(im, (bbl[0], bbl[1]), (bbl[2], bbl[3]), (0, 255, 0))
    for smiles in smiless:
        sml = smiles.get_box(im.shape[0], im.shape[1])
        cv2.rectangle(im, (sml[0], sml[1]), (sml[2], sml[3]), (255, 0, 0))
    f9 = plt.figure(9)
    plt.imshow(im)
    plt.pause(0.05)
    im = im[0:608, 0:608]

    c_r, c_c = centers(im.shape[0], this_scale)
    for r in c_r:
        for c in c_c:
            cv2.circle(im, (int(c), int(r)), 5, (255, 0, 0))

    # cv2.imshow('center of feature maps', im), cv2.waitKey(1)
    f5 = plt.figure(5)
    plt.imshow(im)
    plt.pause(0.05)

    for r in box_ratios:
        up_l, low_r = featureBox(c_c[0], c_r[0], fm=im.shape[0] / this_scale, scale=scale_factors[this_scale], ratio=r)
        cv2.rectangle(im, tuple(np.round(up_l).astype(np.int)),
                      tuple(np.round(low_r).astype(np.int)), (255, 0, 0))
    this_scale = 4
    c_r, c_c = centers(im.shape[0], this_scale)
    for r in c_r:
        for c in c_c:
            drawBox(im, (int(c), int(r)), 5)
    for r in box_ratios:
        up_l, low_r = featureBox(c_c[3], c_r[2], fm=im.shape[0] / this_scale, scale=scale_factors[this_scale - 1], ratio=r)
        cv2.rectangle(im, tuple(np.round(up_l).astype(np.int)),
                      tuple(np.round(low_r).astype(np.int)), (0, 255, 0))
    # cv2.imshow('bbox', im), cv2.waitKey(0)
    f6 = plt.figure(6)
    plt.imshow(im)
    plt.title('different base boxes at different feature map scales')
    plt.pause(0.05)
    pass


# loadShowImages()


def nonMaximalSupression(boxes):
    overlapThresh = 0.3
    if len(boxes) == 0:
        return []
    # x1, y1, x2, y2
    coords = np.zeros([len(boxes), 4])
    area = np.zeros([len(boxes)])
    for box, i in zip(boxes, range(len(boxes))):
        coords[i, 0] = box[0][0]
        coords[i, 1] = box[0][1]
        coords[i, 2] = box[1][0]
        coords[i, 3] = box[1][1]
        area[i] = abs(box[0][0] - box[1][0]) * abs(box[0][1] - box[1][1])

    index_list = np.argsort(coords[:, 3])
    keep = []
    while len(index_list) > 0:
        l = len(index_list) - 1
        i = index_list[l]
        keep.append(index_list[l])
        suppress = [l]
        for index in range(0, l):
            j = index_list[index]
            top = max(coords[i, 0], coords[j, 0])
            lft = max(coords[i, 1], coords[j, 1])
            bot = min(coords[i, 2], coords[j, 2])
            rht = min(coords[i, 3], coords[j, 3])
            wid = max(0, rht - lft + 1)
            hgt = max(0, bot - top + 1)

            overlap = float(wid * hgt) / area[j]
            if overlap > overlapThresh:
                suppress.append(index)
        index_list = np.delete(index_list, suppress)
    return [boxes[k] for k in keep]


def visualizeNonMaximalSupression():
    boxes = []
    im = np.zeros([500, 500, 3], np.uint8)
    boxes.append(((100, 200), (300, 400)))
    boxes.append(((110, 210), (310, 410)))
    boxes.append(((150, 250), (250, 350)))
    boxes.append(((300, 10), (400, 110)))

    for box in boxes:
        cv2.rectangle(im, box[0], box[1], (255, 0, 0))
    f1 = plt.figure(1)
    plt.imshow(im)
    plt.title('Original Boxes')
    plt.pause(0.05)

    out_boxes = nonMaximalSupression(boxes)
    for box in out_boxes:
        cv2.rectangle(im, box[0], box[1], (0, 255, 0))

    f2 = plt.figure(2)
    plt.imshow(im)
    plt.title('Non Maximal Supression')
    plt.pause(0.05)
    pass


# visualizeNonMaximalSupression()

def make_all_boxes():
    scale_groups = []
    offset = 0.5
    all_boxes = {}
    all_box_pixels = {}
    for feature_scale, scale_factor in zip(feature_scales, scale_factors):
        dtype = np.float32
        boxes = []
        # cent_rows, cent_cols = centers(im_shape, this_scale)
        # for cent_row in cent_rows:
        #     for cent_col in cent_cols:
        cent_row = im_shape // 2
        cent_col = im_shape // 2
        for r in box_ratios:
            box = int_cast_tuple(featureBox(cent_col, cent_row, fm=im_shape / feature_scale, scale=scale_factor, ratio=r))
            boxes.append(box)
            all_boxes[feature_scale, r] = box
            up_l = box[0]
            low_r = box[1]
            box_extent = {}
            for i in range(up_l[0], low_r[0]):
                for j in range(up_l[1], low_r[1]):
                    box_extent[(i, j)] = 1;
            all_box_pixels[feature_scale, r] = box_extent
        plt.figure(5)
      #  img = np.zeros([300, 300])
      #  for box in boxes:
      #      cv2.rectangle(img, box[0], box[1], (255,255,255))
      #      my_title = "scale: {}".format(feature_scale)
      #  plt.imshow(img)
      #  plt.title(my_title)
      #  plt.pause(0.1)
        scale_groups.append(boxes)
    return scale_groups, all_boxes, all_box_pixels


scale_groups, all_boxes, all_boxes_pixels = make_all_boxes()


def make_all_centers():
    all_centers = {}
    for scale in feature_scales:
        my_centers = centers(im_shape, scale)
        all_centers[scale] = my_centers
    return all_centers


all_centers = make_all_centers()


def shift_box(bb1, bb2):
    diff = tuple_sub(bb2[0], bb1[0])
    return (bb1[0], tuple_sub(bb2[1], diff))


def realtime_jacc(bb1):
    jacc = {}
    for feature_scale in feature_scales:
        for ratio in box_ratios:
            bb2 = all_boxes[feature_scale, ratio]
            up_l = bb2[0]
            low_r = bb2[1]
            size_box_1 = abs(up_l[0] - low_r[0]) * abs(up_l[1] - low_r[1])

            jac_map = all_boxes_pixels[feature_scale, ratio]
            bb1_s = shift_box(bb2, bb1)

            #    pass
            if boxes_overlap(bb2, bb1_s):
                overlap_pixels = []
                up_l = bb1_s[0]
                low_r = bb1_s[1]
                overlap_count = 0
                for i in range(up_l[0], low_r[0]):
                    for j in range(up_l[1], low_r[1]):
                        if (jac_map.__contains__((i, j))):
                            overlap_pixels.append((j, i))
                            overlap_count += 1
                size_box_2 = abs(up_l[0] - low_r[0]) * abs(up_l[1] - low_r[1])

                AUB = size_box_1 + size_box_2 - overlap_count
                AINTB = overlap_count
                jac_ov = AINTB / AUB
                if jac_ov > 0.5:
                    off = tuple_sub(bb1_s[1], bb1[1])
                    jacc[feature_scale, ratio] = (jac_ov, bb2, off)

    return jacc


# bb1=((left_x, top_y), (right_x, bottom_y))
def boxes_overlap(bb1, bb2):
    if bb1[0][0] > bb2[1][0] or bb2[0][0] > bb1[1][0]: return False
    if bb1[0][1] > bb2[1][1] or bb2[0][1] > bb1[1][1]:
        return False
    else:
        return True


def tst_overlap():
    im = np.zeros(shape=[100, 200, 3], dtype=np.uint8)
    box1 = ((10, 50), (50, 60))
    box2 = ((0, 25), (10, 49))
    cv2.rectangle(im, box1[0], box1[1], (255, 0, 0))
    cv2.rectangle(im, box2[0], box2[1], (0, 0, 255))
    does_overlap = boxes_overlap(box1, box2)
    print(does_overlap)
    f3 = plt.figure(3)
    plt.imshow(im)
    my_title = does_overlap
    plt.title(my_title)
    plt.pause(0.05)
    pass


def tuple_dist(a, b):
    if type(a[0]) is tuple:
        a0 = 0
        a1 = 0
        for tup in a:
            a0 += tup[0] / len(a)
            a1 += tup[1] / len(a)
        a = (a0, a1)
    if type(b[0]) is tuple:
        b0 = 0
        b1 = 0
        for tup in b:
            b0 += tup[0] / len(b)
            b1 += tup[1] / len(b)
        b = (b0, b1)
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def find_closest_center(centers, bbox):
    best_center = None
    closest_dist = sqrt(im_shape ** 2 + im_shape ** 2)
    center_number = None
    for i, center in enumerate(centers):
        dist = tuple_dist(center, bbox)
        if dist < closest_dist: closest_dist = dist; best_center = center; center_number = i
    return best_center, center_number


def center_box(center, box):
    offset = tuple_sub((im_shape // 2, im_shape // 2), center)
    return (tuple_sub(box[0], offset), tuple_sub(box[1], offset))


def find_offset(box_a, box_b):
    a, b = tuple_sub(box_b[0], box_a[0])
    c, d = tuple_sub(box_b[1], box_a[1])
    # (left, top, right, bottom)
    return (a, b, c, d)


here = os.path.abspath('.')


def make_label(box, label, obj_class, im):
    jacc = realtime_jacc(box1)
    for key in jacc.keys():
        jd, box2, off = jacc[key]  # key is (feature scale, scale factor, ratio)
        scale = key[0]
        r = key[1]
        centers = all_centers[scale]
        closest, num = find_closest_center(centers, box1)

        jacc_box = center_box(closest, box2)
        jacc_box = int_cast_tuple(jacc_box)
        offsets = find_offset(jacc_box, box1)
        boxnum = box_ratios.index(r)
        label[scale][floor(num / scale), num % scale, boxnum * 5:(boxnum + 1) * 5] = \
            [1, offsets[0], offsets[1], offsets[2], offsets[3]]
    # cv2.rectangle(im, jacc_box[0], jacc_box[1], (255, 255, 255))
    # plt.imshow(im)
    # plt.pause(0.01)


def read_label(label, im):
    plt.figure(2)

    for scale in feature_scales:
        lab = label[scale]
        for i, n in enumerate(range(0, 30, 5)):
            copy_img = im.copy()
            single_ratio = lab[:, :, n]
            where = np.argwhere(single_ratio)
            for loc in where:
                scale_center = all_centers[scale][loc[1] + scale * loc[0]]
                ratio = box_ratios[i]
                box = all_boxes[scale,ratio]
                box = int_cast_tuple(center_box(scale_center, box))
                # (left, top, right, bottom)
                c, t, l, b, r = lab[loc[0], loc[1], n:n + 5]
                #cv2.rectangle(copy_img, box[0], box[1], (255, 0, 0))
                #plt.imshow(copy_img)
                #plt.pause(0.01)

                #box2 = int_cast_tuple((tuple_add(box[0], (t, l)), tuple_add(box[1], (b, r))))
                #cv2.rectangle(copy_img, box2[0], box2[1], (0, 0, 255))
                #plt.imshow(copy_img)
                #plt.pause(0.01)
                #pass


if __name__ == '__main__':
    file_names = [file for file in os.listdir('.') if file.endswith('p')]
    for file in file_names:
        p = pickle.load(open(file, 'rb'))
        im = p['old_pic']
        im = cv2.resize(im, (300, 300))
        # box1 = ((10, 10), (100, 100))
        # box1 = ((2, 39), (36, 73))
        logoss = p['logos']
        smiless = p['smiles']
        label = {}
        k = len(box_ratios)
        label[8] = np.zeros(shape=[8, 8, 5 * k])
        label[4] = np.zeros(shape=[4, 4, 5 * k])
        label[2] = np.zeros(shape=[2, 2, 5 * k])
        label[1] = np.zeros(shape=[1, 1, 5 * k])
        #plt.figure(1)
        #for logo in logoss:
        #    box1 = logo.get_box(300, 300)
        #    cv2.rectangle(im, box1[0], box1[1], (255, 255, 255))
        #    plt.imshow(im)
        #    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #    im_save = im.copy()

            # loadShowImages()
            # for boxes in scale_groups:
        #    make_label(box=box1, label=label, obj_class=1, im=im_save)

        for smile in smiless:
            box1 = smile.get_box(300, 300)
            #cv2.rectangle(im, box1[0], box1[1], (255, 255, 255))
            #plt.imshow(im)
            #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            #im_save = im.copy()

            # loadShowImages()
            # for boxes in scale_groups:
            make_label(box=box1, label=label, obj_class=1, im=im)

            # im = im_save.copy()
        read_label(label, im)
        p['label']=label
        pickle.dump(p, open(p['name'][:-4]+'.p', "wb"))
        print(p['name'][:-4]+'.p')

        pass

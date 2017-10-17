import os
import xml.etree.ElementTree as ET
import cv2
import pickle
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

        return ((b,a),(d,c))


here = os.path.abspath('.')

xml_names = [name for name in os.listdir(here) if name.endswith(".xml")]
bad_list = []
first = True
h_list = []

for xml_name in xml_names:
    pic_info = {}
    tree = ET.parse(xml_name)
    root = tree.getroot()
    pic_info['name'] = root.find('filename').text
    print(xml_name, ' ', pic_info['name'])
    pic_info['cols'] = int(root.find('size').find('width').text)
    pic_info['rows'] = int(root.find('size').find('height').text)
    pic_info['depth'] = int(root.find('size').find('depth').text)
    im = cv2.imread(pic_info['name'])
    if pic_info['rows'] == 0 or pic_info['cols'] == 0:
        if first:
            first = False
            bad_list.append(pic_info['name'])
        else:
            print(pic_info['name'])
            im = cv2.imread(pic_info['name'])
            pic_info['cols']=im.shape[0]
            pic_info['rows']=im.shape[1]

    if pic_info['rows'] != 0 or pic_info['cols'] != 0:
        objs = [x for x in root if x.tag == 'object']
        logos = [obj for obj in objs if obj.find('name').text == 'logo']
        smiles = [obj for obj in objs if obj.find('name').text == 'smile']
        pic_info['num_boxes'] = len(objs)
        pic_info['num_smiles'] = len(smiles)
        pic_info['num_logo'] = len(logos)
        s_list = []
        l_list = []
        for smile in smiles:
            box = smile.find('bndbox')
            s = BBox(int(box.find('xmin').text),  #min column
                     int(box.find('ymin').text),  #min row
                     int(box.find('xmax').text),  #max column
                     int(box.find('ymax').text),  #max row
                     pic_info['cols'],
                     pic_info['rows'])
            #new_img = im.copy()
            #bbox = s.get_box(img_row=pic_info['rows'], img_col=pic_info['cols'])
            #new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
            #cv2.rectangle(new_img,bbox[0], bbox[1],(255,255,255),  thickness=3)
            #plt.imshow(new_img)
            #plt.pause(0.1)
            #pass
            if s.c_min >1 or s.c_max> 1 or s.r_min > 1 or s.r_max > 1:
                print('you fucked up there are bad values')
            s_list.append(s)
            h_list.append(s.r_max - s.r_min)

        for logo in logos:
            box = logo.find('bndbox')
            l = BBox(int(box.find('xmin').text),  #min column
                     int(box.find('ymin').text),  #min row
                     int(box.find('xmax').text),  #max column
                     int(box.find('ymax').text),  #max row
                     pic_info['cols'],
                     pic_info['rows'])
            if l.c_min >1 or l.c_max> 1 or l.r_min > 1 or l.r_max > 1:
                print('you fucked up there are bad values')
            l_list.append(l)
            h_list.append(l.r_max - l.r_min)
        pic_info['smiles'] = s_list
        pic_info['logos'] = l_list


        if not (im is not None):
            pass
        im_r = im.shape[0]
        im_c = im.shape[1]
        if im_r>im_c:
            ratio = 300.0/im_c
            im_r = cv2.resize(im, (300 , int(ratio * im_r)))
        else:
            ratio = 300.0/im_r
            im_r = cv2.resize(im, (int(ratio*im_c), 300) )

        #cv2.imshow('new', im_r)
        #cv2.imshow('old', im)
        #plt.figure(1)


        pic_info['old_pic'] = im
        pic_info['new_pic'] = im_r
        pickle.dump(pic_info, open(pic_info['name'][:-4]+'.p', "wb"))
#from matplotlib import pyplot as plt
#import numpy as np
#import math

#data = h_list
#bins = np.linspace(math.ceil(min(data)),
#                   math.floor(max(data)),
#                   20) # fixed number of bins

#plt.xlim([min(data)-5, max(data)+5])

#plt.hist(data, bins=np.fliplr(bins), alpha=0.5)
#plt.title('Random Gaussian data (fixed number of bins)')
#plt.xlabel('variable X (20 evenly spaced bins)')
#plt.ylabel('count')

#plt.show()
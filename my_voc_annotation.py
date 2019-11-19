import xml.etree.ElementTree as ET
from os import getcwd
import os

classes = ["kangaroo", "raccoon"]

#ROOT = '/opt/sdb/workspace/data/'
ROOT = '/opt/sdb/workspace/keras/object/keras-yolo3/datasets/'

NAME = 'voc-kangaroo-raccoon'

def get_image_ids(dir_path):
    img_ids = []
    if not os.path.exists(dir_path):
        return []
    for root, dirs, files in os.walk(dir_path):
        for img in files:
            if img[-3:].lower() == 'jpg':
                img_ids.append(img.split('.')[0])
    return img_ids


def convert_annotation(NAME, image_id, list_file):
    in_file = open(ROOT + 'VOCdevkit/%s/Annotations/%s.xml'%(NAME, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

dir_path = ROOT + 'VOCdevkit/{}/JPEGImages'.format(NAME)
image_ids = get_image_ids(dir_path)

if os.path.exists('my_train.txt'):
    os.remove('my_train.txt')

list_file = open('my_train.txt', 'w')
for image_id in image_ids:
    list_file.write(ROOT + 'VOCdevkit/%s/JPEGImages/%s.jpg'%(NAME, image_id))
    convert_annotation(NAME, image_id, list_file)
    list_file.write('\n')
list_file.close()


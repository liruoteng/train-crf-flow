#!/usr/python/bin
import flowlib as fl
import sys
import shutil
import numpy as np
import os
import lmdb
from scipy.misc import imresize


UNKNOWN_FLOW_THRESH = 1e9

# 1. Read in label files
def main():

    path_src = 'labels/'
    data_path = 'data/'
    image_ext = '.ppm'
    label_ext = '.flo'
    train_imgs = ['0000000-img0.ppm', '0000000-img1.ppm']
    label_file = '0000000-gt.flo'
    preprocess_mode = 'pad'
    im_sz = [384, 512]

    # Train
    # Labels
    gt = fl.read_flow(data_path + label_file)
    print('label')
    path_dst = 'train_labels_lmdb'
    convert2lmdb(gt[:, :, 0], path_dst, preprocess_mode, im_sz, 'label')


def convert2lmdb(img_src, path_dst, preprocess_mode, im_sz, data_mode):
    sys.path.append('/home/liruoteng/Documents/MATLAB/caffe/python')
    import caffe
    if os.path.isdir(path_dst):
        shutil.rmtree(path_dst)
        print('DB ' + path_dst + ' already exists.\n Delete' + path_dst + '.')

    db = lmdb.open(path_dst, map_size=int(1e12))

    with db.begin(write=True) as in_txn:
        img = img_src
        if data_mode == 'label':
            img = preprocess_label(img, preprocess_mode, im_sz, data_mode)
        elif data_mode == 'image':
            img = preprocess_image(img, preprocess_mode, im_sz, data_mode)

        # TODO : USE FLOW U AND V IN FUTURE, NOW ONLY FLOW MAP U(HORIZONTAL)
        img_dat = caffe.io.array_to_datum(img)
        in_txn.put('label_file', img_dat.SerializeToString())


def preprocess_label(img, preprocess_mode, im_sz, data_mode):
    img = np.floor(img).astype(type('float', (float,), {}))
    img = preprocess_data(img, preprocess_mode, im_sz, data_mode)
    img = np.expand_dims(img, axis=0)
    return img


def preprocess_image(img, preprocess_mode, im_sz, data_mode):
    return img


def preprocess_data(img, preprocess_mode, im_sz, data_mode):
    if preprocess_mode == 'pad':

        if data_mode == 'image':
            img = np.pad(img, ((0, im_sz[0] - img.shape[0]), (0, im_sz[1] - img.shape[1]), (0, 0)), 'constant',
                         constant_values=(0))
        elif data_mode == 'label':
            img = np.pad(img, ((0, im_sz[0] - img.shape[0]), (0, im_sz[1] - img.shape[1])), 'constant',
                         constant_values=(0))
        else:
            print('Invalid data mode.')

    elif preprocess_mode == 'res':
        img = imresize(img, (im_sz[0], im_sz[1]), interp='bilinear')
    else:
        print('Invalid preprocess mode.')

    return img


# caller

main()


'''
def flow2label(flow):

    u = flow[:, :, 0]
    v = flow[:, :, 1]
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idx_unknown = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idx_unknown] = 0
    v[idx_unknown] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))
    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))
    classes_u = np.floor(u)

    return classes_u

'''
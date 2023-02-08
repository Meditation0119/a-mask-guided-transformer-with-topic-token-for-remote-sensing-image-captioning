from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
import skimage.io
import cv2
from torchvision import transforms as trn

preprocess = trn.Compose([
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #trn.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def main(params):

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    N = len(imgs)

    seed(123)  # make reproducible

    dir_fc = params['output_dir'] + '_fc'
    dir_att = params['output_dir'] + '_att'

    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    if not os.path.isdir(dir_att):
        os.mkdir(dir_att)

    for i, img in enumerate(imgs):
        # load the image
        #I = skimage.io.imread(os.path.join(params['images_root'], img['file_path']))
        I=cv2.imread(os.path.join(params['images_root'], img['file_path']))

        # handle grayscale input images
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)
        I=cv2.resize(I,(224,224))
        #I = I.astype('float32')
        #I = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
        I = preprocess(I)
        # image_show_or = I.cpu().numpy().transpose(1, 2, 0)
        # cv2.imwrite('origin.jpg', image_show_or)
        # write to pkl
        tmp_fc, tmp_att = np.zeros([512]), I
        np.save(os.path.join(dir_fc, str(img['id'])), tmp_fc)
        np.savez_compressed(os.path.join(dir_att, str(img['id'])), feat=tmp_att.data.cpu().float().numpy())

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N))
    print('wrote ', params['output_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default=r'\data/RSICD/rsicd_talk.json',
                        help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default=r'data/OSdataset256/', help='output h5 file')

    # options
    parser.add_argument('--images_root', default=r'data/OSdataset256/',
                        help='root location in which images are stored, to be prepended to file_path in input json')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch
from models.VitModel import VisionTransformer


# Input arguments and options
parser = argparse.ArgumentParser()
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Input paths
parser.add_argument('--model', type=str, default='save/rsicd/model-best.pth',
                help='path to model to evaluate')
parser.add_argument('--cnn_weights', type=str, default='best_RSICD.pth',
                help='path to cnn model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='save/rsicd/infos_-best.pkl',
                help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=1,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=1,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1, #self-cider =0
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=3, #self-cider =1
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--max_length', type=int, default=16,
                help='Maximum length during sampling')
parser.add_argument('--length_penalty', type=str, default='',
                help='wu_X or avg_X, X is the alpha')
parser.add_argument('--group_size', type=int, default=1,
                help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--diversity_lambda', type=float, default=0.5,
                help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--decoding_constraint', type=int, default=0,
                help='If 1, not allowing same word in a row')
parser.add_argument('--block_trigrams', type=int, default=0,
                help='block repeated trigram.')
parser.add_argument('--remove_bad_endings', type=int, default=0,
                help='Remove bad endings')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default="",
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='data/RSICD_images/',
                help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='data/RSICD/_fc',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='data/RSICD/_att',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_box_dir', type=str, default='data/cocotalk_box',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='data/rsicd_label.h5',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='data/rsicd_talk.json',
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='val',
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='',
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='', 
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--verbose_beam', type=int, default=1, 
                help='if we need to print out all beam search beams.')
parser.add_argument('--verbose_loss', type=int, default=0, 
                help='if we need to calculate loss.')

opt = parser.parse_args()

# Load infos
with open(opt.infos_path,'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    opt.input_box_dir = getattr(infos['opt'], 'input_box_dir', '')
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id
ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval", "block_trigrams"]

for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            print(vars(infos['opt'])['input_json'])
            print(vars(infos['opt'])['input_label_h5'])
            #assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping
torch.cuda.set_device(0)

# Setup the model
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model),strict=False)
model.cuda()
model.eval()
#cnn_model = utils.build_cnn(opt)  #Resnet
vit_model=VisionTransformer(image_size=(224, 224),num_classes=30)

try:
    vit_model.load_state_dict(torch.load(opt.cnn_weights),strict=False)
except:
    vit_model.load_state_dict(torch.load('best_RSICD.pth')['state_dict'], strict=False)

vit_model = vit_model.cuda()

vit_model.eval()

crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw(opt)
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']


# Set sample options
loss, split_predictions, lang_stats = eval_utils.eval_split(vit_model,model, crit, loader,
    vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/rsicd_result.json', 'w'))

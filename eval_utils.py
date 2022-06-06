from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils

bad_endings = ['a', 'an', 'the', 'in', 'for', 'at', 'of', 'with',
               'before', 'after', 'on', 'upon', 'near', 'to', 'is', 'are', 'am']
bad_endings += ['the']


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/rsicd_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join(
        'eval_results/', '.cache_' + model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    # serialize to temporary json file. Sigh, COCO API...
    json.dump(preds_filt, open(cache_path, 'w'))

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    out['bad_count_rate'] = sum([count_bad(_['caption'])
                                 for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join(
        'eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


def eval_split(vit_model, model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get(
        'num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    # Use this nasty way to make other code clean since it's a global configuration
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)

    # Make sure in the evaluation mode
    vit_model.eval()
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        tmp = [data['fc_feats'], data['att_feats'], data['labels'],
               data['masks'], data['att_masks']]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp

        with torch.no_grad():
            att_feats = att_feats.view(att_feats.size(0), 3, 224, 224)
            att_feats = vit_model(att_feats)
            fc_feats = att_feats[:, 0]
            # fc_feats = att_feats.mean(3).mean(2)
            # att_feats = torch.nn.functional.adaptive_avg_pool2d(
            #     att_feats, [7, 7]).permute(0, 2, 3, 1)
            # att_feats = att_feats.permute(0, 2, 3, 1)
            # att_feats = att_feats.view(att_feats.size(0), 49, -1)

            att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), 5,) + att_feats.size(
            )[1:])).contiguous().view((att_feats.size(0) * 5), -1, att_feats.size()[-1])
            fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), 5,) + fc_feats.size(
            )[1:])).contiguous().view(*((fc_feats.size(0) * 5,) + fc_feats.size()[1:]))
            # att_masks = (att_feats.data > -np.inf).long()
            # # att_masks[:, 0] += 1
            #
            # # seq_mask = seq_mask.unsqueeze(-2)
            #
            # random_mask = torch.rand((att_masks.size(0), att_masks.size(1))).cuda()
            # # random_eye=torch.eye(att_masks.size(1)).unsqueeze(0).cuda()
            # # random_eye=random_eye.repeat(seq_mask.size(0),1,1)
            # # random_mask=random_mask+random_eye
            # random_mask = (random_mask > 0.3).long().unsqueeze(-1)
            # random_mask = random_mask.repeat(1, 1, 768)
            # att_masks = att_masks * random_mask

            _att_feats = att_feats
            _fc_feats = fc_feats

        #if data.get('labels', None) is not None and verbose_loss:
        if data.get('labels', None) is not None :
            with torch.no_grad():
                seq=model(fc_feats, att_feats, labels,att_masks)
                loss = crit(seq, labels[:, 1:], masks[:, 1:]).item()
                # seq=torch.argmax(seq,2)
                # print(seq.shape)
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [_fc_feats[np.arange(loader.batch_size) * loader.seq_per_img],
               _att_feats[np.arange(
                   loader.batch_size) * loader.seq_per_img],
               att_masks[np.arange(loader.batch_size) * loader.seq_per_img] if att_masks is not None else None]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        #print(fc_feats)
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(fc_feats, att_feats, att_masks,
                        opt=eval_kwargs, mode='sample')[0].data

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _[
                      'seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)

        sents = utils.decode_sequence(loader.get_vocab(), seq)

        # for k, sent in enumerate(sents):
        #     entry = {'image_id': data['infos'][k]['id'], 'caption': sent,'image_path': os.path.join(data['infos'][k]['file_path'])}
        #     #entry = {'image_id': data['infos'][0]['id'], 'caption': sent}
        #     if eval_kwargs.get('dump_path', 0) == 1:
        #         entry['file_name'] = data['infos'][k]['file_path']
        #     predictions.append(entry)
        #     if eval_kwargs.get('dump_images', 0) == 1:
        #         # dump the raw image to vis/ folder
        #         cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + \
        #             '" vis/imgs/img' + \
        #             str(len(predictions)) + '.jpg'  # bit gross
        #         print(cmd)
        #         os.system(cmd)
        #
        #     if verbose:
        #         print('image %s: %s' % (entry['image_id'], entry['caption']))
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent,'image_path':os.path.join(data['infos'][k]['file_path'])}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 0:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join( data['infos'][k]['file_path']) + \
                    '" vis/imgs/img' + \
                    str(len(predictions)) + '.jpg'
                # bit gross

                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' % (entry['image_id'], entry['caption']))
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %
                  (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(
            dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    vit_model.train()
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats

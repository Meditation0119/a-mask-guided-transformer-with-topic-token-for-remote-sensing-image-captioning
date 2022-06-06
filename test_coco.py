import sys
import os

sys.path.append("coco-caption")

annFile = 'coco-caption/annotations/captions_val2014.json'

from pycocotools.coco import COCO

from pycocoevalcap.eval import COCOEvalCap

if True:
    coco = COCO(annFile)
    valids = coco.getImgIds()

    preds = [1, 2, 3, 4, 5, 6]
    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    # serialize to temporary json file. Sigh, COCO API...
    json.dump(preds_filt, open(cache_path, 'w'))

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
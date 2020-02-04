import os
import sys
import cv2
import yaml
import time
import logging
import argparse
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
from attrdict import AttrDict
from collections import OrderedDict
# import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from twonet import Dual_net
from inference_crop import Crop_image
from data_complx_channel1_scale import Provider



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='standard.yaml', help='path to config file')
    parser.add_argument('-st', '--start', type=int, default=1000)
    parser.add_argument('-en', '--end', type=int, default=None)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))
    
    valid = Provider('valid', cfg)
    valid_provider = valid

    stride = 1000
    PAD = cfg.TRAIN.pad
    thresd = cfg.TRAIN.thresd
    model_name = cfg.TEST.model_name
    model_path = cfg.TRAIN.save_path
    model_path = os.path.join(model_path, model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Dual_net(input_channels=cfg.TRAIN.input_nc).to(device)
    f_valid_txt = open(os.path.join(model_path, 'valid.txt'), 'a')
    for iters in range(args.start, args.end, stride):
        # Load model
        ckpt_name = 'model-'+str(iters)+'.ckpt'
        print('Load ' + ckpt_name, end=' ')
        ckpt_path = os.path.join(model_path, ckpt_name)
        checkpoint = torch.load(ckpt_path)
        new_state_dict = OrderedDict()
        state_dict = checkpoint['model_weights']
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            # name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        valid_score = []
        for k in range(valid_provider.data.num):
            raw, label = valid_provider.data.gen(k)
            crop_img = Crop_image(raw,crop_size=cfg.TRAIN.crop_size,overlap=cfg.TRAIN.overlap)
            for i in range(crop_img.num):
                for j in range(crop_img.num):
                    raw_crop = crop_img.gen(i, j)
                    #########
                    #inference
                    if crop_img.dim == 3:
                        raw_crop_ = raw_crop[np.newaxis, :, :, :]
                    else:
                        raw_crop_ = raw_crop[np.newaxis, np.newaxis, :, :]
                    inputs = torch.Tensor(raw_crop_).to(device)
                    inputs = F.pad(inputs, (PAD, PAD, PAD, PAD))
                    
                    with torch.no_grad():
                        pred = model(inputs)
                    pred = pred[2]
                    pred = F.pad(pred, (-PAD, -PAD, -PAD, -PAD))
                    pred = F.softmax(pred, dim=1)
                    pred = torch.argmax(pred, dim=1).squeeze(0)
                    pred = pred.data.cpu().numpy()
                    # pred = np.squeeze(pred)
                    # pred = pred[1]
                    #########
                    crop_img.save(i, j, pred)
            results = crop_img.result()
            results[results<=thresd] = 0
            results[results>thresd] = 1
            temp_label = label.flatten()
            temp_result = results.flatten()
            f1_common = f1_score(1 - temp_label, 1 - temp_result)
            valid_score.append(f1_common)
        avg_f1 = sum(valid_score) / len(valid_score)
        print('step %d, f1 = %.6f' % (iters, avg_f1))
        f_valid_txt.write('step %d, f1 = %.6f' % (iters, avg_f1))
        f_valid_txt.write('\n')
        f_valid_txt.flush()
        sys.stdout.flush()

    print('***Done***')
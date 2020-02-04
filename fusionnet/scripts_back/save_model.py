import os
import cv2
import torch
import yaml
import time
import argparse
import numpy as np
from PIL import Image
# from twonet import Dual_net
from model_instanceBN_sigmoid import FusionNet
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from attrdict import AttrDict
# from inference_crop import Crop_image
from inference_crop_batch import Crop_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='standard.yaml', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='test', help='path to config file')
    parser.add_argument('-t', '--tta', action='store_false', default=True)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp

    # if cfg.TRAIN.track == 'simple':
    #     # from data_simple import Provider
    #     # from data_simple_channel1 import Provider
    #     from data_simple_channel1_scale import Provider
    # else:
    #     # from data_complx_channel1 import Provider
    #     from data_complx_channel1_scale import Provider

    if cfg.TRAIN.track == 'simple':
        if cfg.TEST.mode == 'train':
            test_path = '../train_result/simple'
            base_path = '../../data/simple/train_raw'
        elif cfg.TEST.mode == 'valid':
            test_path = '../valid_result/simple'
            base_path = '../../data/simple/valid_raw'
        elif cfg.TEST.mode == 'test':
            test_path = '../test_result/simple'
            base_path = '../../data/simple/test_raw'
        else:
            raise AttributeError('No this test mode!')
    else:
        test_path = '../test_result/complex'
        base_path = '../../data/complex/' + cfg.TEST.crop_way + '/test_raw'
    model_name = cfg.TEST.model_name
    save_path = os.path.join(test_path, model_name + '_aug_batch', 'result')
    model_path = cfg.TRAIN.save_path
    model_path = os.path.join(model_path, model_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    thresd = cfg.TEST.thresd

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = Dual_net(input_channels=cfg.TRAIN.input_nc).to(device)
    model = FusionNet(input_nc=cfg.TRAIN.input_nc,output_nc=cfg.TRAIN.output_nc,ngf=cfg.TRAIN.ngf).to(device)
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)

    ckpt = 'model.ckpt'
    ckpt_path = os.path.join(model_path, ckpt)
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        iters = checkpoint['current_iter']
        avg_f1_tmp = checkpoint['valid_result']
        if cuda_count > 1:
            model.load_state_dict(checkpoint['model_weights'])
        else:
            new_state_dict = OrderedDict()
            state_dict = checkpoint['model_weights']
            for k, v in state_dict.items():
                name = k[7:] # remove module.
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
    else:
        raise AttributeError('No checkpoint found at %s' % model_path)

    #### save model
    ckpt2 = 'model_simple.ckpt'
    ckpt_path2 = os.path.join(model_path, ckpt2)
    states = {'current_iter': iters, 'valid_result': avg_f1_tmp,
            'model_weights': model.state_dict()}
    torch.save(states, ckpt_path2)

    print('***Done***')
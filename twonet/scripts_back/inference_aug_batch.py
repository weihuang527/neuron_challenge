import os
import cv2
import torch
import yaml
import time
import argparse
import numpy as np
from PIL import Image
from twonet import Dual_net
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

    if cfg.TRAIN.track == 'simple':
        # from data_simple import Provider
        # from data_simple_channel1 import Provider
        from data_simple_channel1_scale import Provider
    else:
        # from data_complx_channel1 import Provider
        from data_complx_channel1_scale import Provider

    if args.mode == 'train':
        pass
    else:
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
        model = Dual_net(input_channels=cfg.TRAIN.input_nc).to(device)
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

        # checkpoint = torch.load(ckpt_path)
        # new_state_dict = OrderedDict()
        # state_dict = checkpoint['model_weights']
        # for k, v in state_dict.items():
        #     name = k[7:] # remove module.
        #     # name = k
        #     new_state_dict[name] = v
        # model.load_state_dict(new_state_dict)

        PAD = cfg.TRAIN.pad
        img_list = os.listdir(base_path)

        debug = False
        for f_img in img_list:
            print('Inference: ' + f_img, end=' ')
            raw = np.asarray(Image.open(os.path.join(base_path, f_img)).convert('L'))
            if cfg.TRAIN.track == 'complex':
                if raw.shape[0] == 9959 or raw.shape[0] == 9958:
                    raw_ = np.zeros((10240,10240), dtype=np.uint8)
                    raw_[141:141+9959, 141:141+9958] = raw
                    raw = raw_
                    del raw_
            # raw = np.asarray(Image.open(os.path.join(base_path, f_img)))
            start = time.time()
            raw = raw.astype(np.float32) / 255.0
            results_aug = np.zeros_like(raw, dtype=np.float32)
            raw_aug = []
            raw_flipud = np.flipud(raw)
            if args.tta:
                raw_aug.append(raw)
                raw_aug.append(np.rot90(raw, 1))
                raw_aug.append(np.rot90(raw, 2))
                raw_aug.append(np.rot90(raw, 3))
                raw_aug.append(raw_flipud)
                raw_aug.append(np.rot90(raw_flipud, 1))
                raw_aug.append(np.rot90(raw_flipud, 2))
                raw_aug.append(np.rot90(raw_flipud, 3))
            else:
                raw_aug.append(raw)
            raw_aug = np.array(raw_aug)
            crop_img = Crop_image(raw_aug,crop_size=cfg.TRAIN.crop_size,overlap=cfg.TRAIN.overlap)
            for i in range(crop_img.num):
                for j in range(crop_img.num):
                    raw_crop = crop_img.gen(i, j)
                    #########
                    #inference
                    if crop_img.dim == 3:
                        raw_crop_ = raw_crop[:, np.newaxis, :, :].copy()
                    else:
                        raw_crop_ = raw_crop[np.newaxis, np.newaxis, :, :].copy()
                    inputs = torch.Tensor(raw_crop_).to(device)
                    inputs = F.pad(inputs, (PAD, PAD, PAD, PAD))
                    with torch.no_grad():
                        pred = model(inputs)
                    pred = pred[2]
                    pred = F.pad(pred, (-PAD, -PAD, -PAD, -PAD))
                    pred = F.softmax(pred, dim=1)
                    if cfg.TEST.if_binary:
                        pred = torch.argmax(pred, dim=1).squeeze(1)
                    else:
                        pred = pred[:, 1].squeeze(1)
                    pred = pred.data.cpu().numpy()
                    # pred = np.squeeze(pred)
                    #########
                    crop_img.save(i, j, pred)
            results = crop_img.result()
            if thresd is None:
                results[results<=0] = 0
                results[results>1] = 1
            else:
                results[results<=thresd] = 0
                results[results>thresd] = 1

            inference_results = []
            if args.tta:
                inference_results.append(results[0])
                inference_results.append(np.rot90(results[1], 3))
                inference_results.append(np.rot90(results[2], 2))
                inference_results.append(np.rot90(results[3], 1))
                inference_results.append(np.flipud(results[4]))
                inference_results.append(np.flipud(np.rot90(results[5], 3)))
                inference_results.append(np.flipud(np.rot90(results[6], 2)))
                inference_results.append(np.flipud(np.rot90(results[7], 1)))
            else:
                inference_results.append(results[0])
            inference_results = np.array(inference_results)

            if debug:
                print('the shape of inference: ', inference_results.shape)
                for k in range(inference_results.shape[0]):
                    name = f_img[:-4] + '_' + str(k) + '.tiff'
                    img = (inference_results[k] * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_path, name), img)
            results_aug = np.sum(inference_results, axis=0) / inference_results.shape[0]
            results_aug = (results_aug * 255).astype(np.uint8)
            # if cfg.TRAIN.track == 'complex':
            #     results = results[141:141+9959, 141:141+9958]
            print('COST TIME: ', (time.time() - start))
            cv2.imwrite(os.path.join(save_path, f_img[:-3]+'tiff'), results_aug)

    print('***Done***')
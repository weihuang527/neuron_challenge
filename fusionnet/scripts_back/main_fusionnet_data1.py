from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import cv2
import logging
import argparse
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from model import FusionNet
from model_instanceBN import FusionNet
from metrics_scaleable_f1_conv import scaleable_f1
from inference_crop import Crop_image

def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
                level    = logging.INFO,
                format   = '%(message)s',
                datefmt  = '%m-%d %H:%M',
                filename = path,
                filemode = 'w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    if cfg.TRAIN.is_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer

def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    valid = Provider('valid', cfg)
    valid_provider = valid
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    model = FusionNet(input_nc=cfg.TRAIN.input_nc,output_nc=cfg.TRAIN.output_nc,ngf=cfg.TRAIN.ngf).to(device)
    # n_data = list(cfg.DATA.patch_size)
    # rand_data = torch.rand(1,1,n_data[0],n_data[1])
    # writer.add_graph(model, (rand_data,))

    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model

def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model.ckpt')

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d, loss = %.4f' % (checkpoint['current_iter'], checkpoint['valid_result']))
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0

def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr

def loop(cfg, train_provider, valid_provider, model, criterion, optimizer, iters, writer):
    PAD = cfg.TRAIN.pad
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    valid_score = []
    valid_score_tmp = []
    thresd = cfg.TRAIN.thresd
    most_f1 = 0
    most_iters = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    while iters <= cfg.TRAIN.total_iters:
        # train
        iters += 1
        t1 = time.time()
        input, target = train_provider.next()
        
        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        optimizer.zero_grad()
        input = F.pad(input, (PAD, PAD, PAD, PAD))
        pred = model(input)
        pred = F.pad(pred, (-PAD, -PAD, -PAD, -PAD))
        # target = torch.unsqueeze(target, dim=1)
        # if iters == 1:
        #     writer.add_graph(model, (input,))

        loss = criterion(pred, target)
        loss.backward()
        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()
        
        sum_loss += loss.item()
        sum_time += time.time() - t1
        
        # log train
        if iters % cfg.TRAIN.display_freq == 0:
            rcd_time.append(sum_time)
            logging.info('step %d, loss = %.4f (wt: *10, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss / cfg.TRAIN.display_freq * 10, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
            writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 10, iters)
            f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss / cfg.TRAIN.display_freq * 10))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0
        
        # valid
        if iters % cfg.TRAIN.valid_freq == 0:
            input = F.pad(input, (-PAD, -PAD, -PAD, -PAD))
            input0 = (np.squeeze(input[0].data.cpu().numpy()) * 255).astype(np.uint8)
            # target = (np.squeeze(target[0].data.cpu().numpy()) * 255).astype(np.uint8)
            target = np.squeeze(target[0].data.cpu().numpy())
            target[target>thresd] = 1; target[target<thresd] = 0
            target = (target * 255).astype(np.uint8)
            pred = np.squeeze(pred[0].data.cpu().numpy())
            pred[pred>thresd] = 1; pred[pred<thresd] = 0
            pred = (pred * 255).astype(np.uint8)
            input0 = input0[0]
            im_cat = np.concatenate([input0, pred, target], axis=1)
            Image.fromarray(im_cat).save(os.path.join(cfg.cache_path, '%06d.png' % iters))
        
        # save
        if iters % cfg.TRAIN.save_freq == 0:
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
                        pred = F.pad(pred, (-PAD, -PAD, -PAD, -PAD))
                        pred = pred.data.cpu().numpy()
                        pred = np.squeeze(pred)
                        #########
                        crop_img.save(i, j, pred)
                results = crop_img.result()
                results[results<=thresd] = 0
                results[results>thresd] = 1
                temp_label = label.flatten()
                temp_result = results.flatten()
                f1_common = f1_score(1 - temp_label, 1 - temp_result)
                f1 = 0
                results_img = (results * 255).astype(np.uint8)
                label_img = (label * 255).astype(np.uint8)
                im_cat_valid = np.concatenate([results_img, label_img], axis=1)
                if k == valid_provider.data.num - 1:
                    Image.fromarray(im_cat_valid).save(os.path.join(cfg.cache_path, 'valid_%06d.png' % iters))
                valid_score.append(f1)
                valid_score_tmp.append(f1_common)
            avg_f1_soft = sum(valid_score) / len(valid_score)
            avg_f1_tmp = sum(valid_score_tmp) / len(valid_score_tmp)
            logging.info('step %d, f1_soft = %.6f' % (iters, avg_f1_soft))
            logging.info('step %d, f1_common = %.6f' % (iters, avg_f1_tmp))
            writer.add_scalar('valid', avg_f1_tmp, iters)
            f_valid_txt.write('step %d, f1 = %.6f' % (iters, avg_f1_tmp))
            f_valid_txt.write('\n')
            f_valid_txt.flush()
            sys.stdout.flush()
            if avg_f1_tmp > most_f1:
                most_f1 = avg_f1_tmp
                most_iters = iters
                states = {'current_iter': iters, 'valid_result': avg_f1_tmp,
                        'model_weights': model.state_dict(), 'optimizer_weights': optimizer.state_dict()}
                torch.save(states, os.path.join(cfg.save_path, 'model.ckpt'))
                print('***************save modol, when f1 = %.6f and iters = %d.***************' % (most_f1, most_iters))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='standard.yaml', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
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
        from data_simple import Provider
    else:
        # from data_complx import Provider
        from data_complx_data1 import Provider

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model = build_model(cfg, writer)
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, F.mse_loss, optimizer, init_iters, writer)
        writer.close()
    else:
        if cfg.TRAIN.track == 'simple':
            test_path = '../test_result/simple'
            base_path = '../../data/simple/test_raw'
        else:
            test_path = '../test_result/complex'
            base_path = '../../data/complex/' + cfg.TEST.crop_way + '/test_raw'
        model_name = cfg.TEST.model_name
        save_path = os.path.join(test_path, model_name, 'result')
        model_path = cfg.TRAIN.save_path
        model_path = os.path.join(model_path, model_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        thresd = cfg.TEST.thresd
        model = FusionNet(input_nc=cfg.TRAIN.input_nc,output_nc=cfg.TRAIN.output_nc,ngf=cfg.TRAIN.ngf)
        ckpt = 'model.ckpt'
        ckpt_path = os.path.join(model_path, ckpt)
        checkpoint = torch.load(ckpt_path)

        new_state_dict = OrderedDict()
        state_dict = checkpoint['model_weights']
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            # name = k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        PAD = cfg.TRAIN.pad
        img_list = os.listdir(base_path)
        for f_img in img_list:
            print('Inference: ' + f_img, end=' ')
            # raw = np.asarray(Image.open(os.path.join(base_path, f_img)).convert('L'))
            raw = np.asarray(Image.open(os.path.join(base_path, f_img)))
            raw = raw.transpose(2, 0, 1)
            if cfg.TRAIN.track == 'complex':
                raw_ = np.zeros((3,10240,10240), dtype=np.uint8)
                raw_[:, 141:141+9959, 141:141+9958] = raw
                raw = raw_
                del raw_
            crop_img = Crop_image(raw,crop_size=cfg.TRAIN.crop_size,overlap=cfg.TRAIN.overlap)
            start = time.time()
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
                    # pred = pred[0]
                    pred = F.pad(pred, (-PAD, -PAD, -PAD, -PAD))
                    # pred = F.softmax(pred, dim=1)
                    # pred = torch.argmax(pred, dim=1).squeeze(0)
                    pred = pred.data.cpu().numpy()
                    pred = np.squeeze(pred)
                    #########
                    crop_img.save(i, j, pred)
            results = crop_img.result()
            results[results<=thresd] = 0
            results[results>thresd] = 1
            results = (results * 255).astype(np.uint8)
            if cfg.TRAIN.track == 'complex':
                results = results[141:141+9959, 141:141+9958]
            print('COST TIME: ', (time.time() - start))
            cv2.imwrite(os.path.join(save_path, f_img[:-3]+'tiff'), results)
    print('***Done***')
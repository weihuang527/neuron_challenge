import numpy as np
import os
import cv2
import shutil
import argparse
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=int, default=1,
                        help='the mode for generating data. ' + 
                        '1 ==> fold-1 ' +
                        '2 ==> fold-2 ' +
                        '3 ==> all data ')
    args = parser.parse_args()
    mode = args.mode
    print('Crop training data in origin resolution under mode ' + str(mode))
    
    train_path = './data/U-RISC OPEN DATA COMPLEX/train'
    label_path = './data/U-RISC OPEN DATA COMPLEX/complex_train_label'
    valid_path = './data/U-RISC OPEN DATA COMPLEX/val'
    data_list = './data/data_list'
    if not os.path.exists(data_list):
        os.makedirs(data_list)
    
    crop_size = 1024
    over_lap = 512
    pad_size = 10240
    num = (pad_size - crop_size) // over_lap + 1

    file_list = os.listdir(train_path)
    file_list_png = []
    for f in file_list:
        if f[-4:] == '.png':
            file_list_png.append(f)
    
    if mode == 1:
        valid_dataset = ['0193_1_1565791505_30.png', '0198_1_1565791505_79.png', '0205_1_1565791505_15.png', '0236_1_1565791505_26.png']
        out_train_path = './data/complex/crop_x0_1024/train_raw'
        out_label_path = './data/complex/crop_x0_1024/train_label'
        out_valid_raw_path = './data/complex/crop_x0_1024/valid_raw'
        out_valid_label_path ='./data/complex/crop_x0_1024/valid_label'

        out_path = [out_train_path, out_label_path, out_valid_raw_path, out_valid_label_path]
        for ip in out_path:
            if not os.path.exists(ip):
                os.makedirs(ip)
        
        train_txt = './data/data_list/complex_crop_x0_1024_train_list.txt'
        valid_txt = './data/data_list/complex_crop_x0_1024_valid_list.txt'
        txt_raw_path = 'complex/crop_x0_1024/train_raw/'
        txt_label_path = 'complex/crop_x0_1024/train_label/'
        txt_valid_raw_path = 'complex/crop_x0_1024/valid_raw/'
        txt_valid_label_path = 'complex/crop_x0_1024/valid_label/'

        f_txt = open(train_txt, 'w')
        f_valid_txt = open(valid_txt, 'w')
        
        for f in file_list_png:
            if f not in valid_dataset:
                print('Processing ' + f)
                name = f[:-4]
                raw = cv2.imread(os.path.join(train_path, f))
                label = cv2.imread(os.path.join(label_path, name+'.tiff'), cv2.IMREAD_GRAYSCALE)
                h,w = label.shape
                raw_pad = np.zeros((pad_size,pad_size,3), dtype=np.uint8)
                label_pad = np.zeros((pad_size,pad_size), dtype=np.uint8)
                raw_pad[141:141+h, 141:141+w, :] = raw
                label_pad[141:141+h, 141:141+w] = label
                for i in range(num):
                    for j in range(num):
                        raw_crop = raw_pad[i*over_lap:i*over_lap+crop_size, j*over_lap:j*over_lap+crop_size, :]
                        label_crop = label_pad[i*over_lap:i*over_lap+crop_size, j*over_lap:j*over_lap+crop_size]
                        raw_crop_name = name+'_'+str(i).zfill(2)+'_'+str(j).zfill(2)+'.png'
                        label_crop_name = name+'_'+str(i).zfill(2)+'_'+str(j).zfill(2)+'.tiff'
                        cv2.imwrite(os.path.join(out_train_path, raw_crop_name), raw_crop)
                        cv2.imwrite(os.path.join(out_label_path, label_crop_name), label_crop)
                        f_txt.write(txt_raw_path+raw_crop_name+ ' ' +txt_label_path+label_crop_name)
                        f_txt.write('\n')
            else:
                print('Copy ' + f)
                shutil.copyfile(os.path.join(train_path, f), os.path.join(out_valid_raw_path, f))
                shutil.copyfile(os.path.join(label_path, f[:-4]+'.tiff'), os.path.join(out_valid_label_path, f[:-4]+'.tiff'))
                f_valid_txt.write(txt_valid_raw_path+f+ ' ' +txt_valid_label_path+f[:-4]+'.tiff')
                f_valid_txt.write('\n')
        f_txt.close()
        f_valid_txt.close()
    elif mode == 2:
        valid_dataset = ['0195_1_1565791505_56.png', '0199_1_1565791505_28.png', '0219_1_1565791505_12.png', '0263_1_1565791505_37.png']
        out_train_path = './data/complex/crop_x0_1024/train_raw_k2'
        out_label_path = './data/complex/crop_x0_1024/train_label_k2'
        out_valid_raw_path = './data/complex/crop_x0_1024/valid_raw_k2'
        out_valid_label_path ='./data/complex/crop_x0_1024/valid_label_k2'

        out_path = [out_train_path, out_label_path, out_valid_raw_path, out_valid_label_path]
        for ip in out_path:
            if not os.path.exists(ip):
                os.makedirs(ip)
        
        train_txt = './data/data_list/complex_crop_x0_1024_train_k2_list.txt'
        valid_txt = './data/data_list/complex_crop_x0_1024_valid_k2_list.txt'
        txt_raw_path = 'complex/crop_x0_1024/train_raw_k2/'
        txt_label_path = 'complex/crop_x0_1024/train_label_k2/'
        txt_valid_raw_path = 'complex/crop_x0_1024/valid_raw_k2/'
        txt_valid_label_path = 'complex/crop_x0_1024/valid_label_k2/'

        f_txt = open(train_txt, 'w')
        f_valid_txt = open(valid_txt, 'w')
        
        for f in file_list_png:
            if f not in valid_dataset:
                print('Processing ' + f)
                name = f[:-4]
                raw = cv2.imread(os.path.join(train_path, f))
                label = cv2.imread(os.path.join(label_path, name+'.tiff'), cv2.IMREAD_GRAYSCALE)
                h,w = label.shape
                raw_pad = np.zeros((pad_size,pad_size,3), dtype=np.uint8)
                label_pad = np.zeros((pad_size,pad_size), dtype=np.uint8)
                raw_pad[141:141+h, 141:141+w, :] = raw
                label_pad[141:141+h, 141:141+w] = label
                for i in range(num):
                    for j in range(num):
                        raw_crop = raw_pad[i*over_lap:i*over_lap+crop_size, j*over_lap:j*over_lap+crop_size, :]
                        label_crop = label_pad[i*over_lap:i*over_lap+crop_size, j*over_lap:j*over_lap+crop_size]
                        raw_crop_name = name+'_'+str(i).zfill(2)+'_'+str(j).zfill(2)+'.png'
                        label_crop_name = name+'_'+str(i).zfill(2)+'_'+str(j).zfill(2)+'.tiff'
                        cv2.imwrite(os.path.join(out_train_path, raw_crop_name), raw_crop)
                        cv2.imwrite(os.path.join(out_label_path, label_crop_name), label_crop)
                        f_txt.write(txt_raw_path+raw_crop_name+ ' ' +txt_label_path+label_crop_name)
                        f_txt.write('\n')
            else:
                print('Copy ' + f)
                shutil.copyfile(os.path.join(train_path, f), os.path.join(out_valid_raw_path, f))
                shutil.copyfile(os.path.join(label_path, f[:-4]+'.tiff'), os.path.join(out_valid_label_path, f[:-4]+'.tiff'))
                f_valid_txt.write(txt_valid_raw_path+f+ ' ' +txt_valid_label_path+f[:-4]+'.tiff')
                f_valid_txt.write('\n')
        f_txt.close()
        f_valid_txt.close()
    elif mode == 3:
        out_train_path = './data/complex/crop_x0_1024/train_raw_all'
        out_label_path = './data/complex/crop_x0_1024/train_label_all'

        out_path = [out_train_path, out_label_path]
        for ip in out_path:
            if not os.path.exists(ip):
                os.makedirs(ip)

        train_txt = './data/data_list/complex_crop_x0_1024_train_all_list.txt'
        txt_raw_path = 'complex/crop_x0_1024/train_raw_all/'
        txt_label_path = 'complex/crop_x0_1024/train_label_all/'

        f_txt = open(train_txt, 'w')

        for f in file_list_png:
            print('Processing ' + f)
            name = f[:-4]
            raw = cv2.imread(os.path.join(train_path, f))
            label = cv2.imread(os.path.join(label_path, name+'.tiff'), cv2.IMREAD_GRAYSCALE)
            h,w = label.shape
            raw_pad = np.zeros((pad_size,pad_size,3), dtype=np.uint8)
            label_pad = np.zeros((pad_size,pad_size), dtype=np.uint8)
            raw_pad[141:141+h, 141:141+w, :] = raw
            label_pad[141:141+h, 141:141+w] = label
            for i in range(num):
                for j in range(num):
                    raw_crop = raw_pad[i*over_lap:i*over_lap+crop_size, j*over_lap:j*over_lap+crop_size, :]
                    label_crop = label_pad[i*over_lap:i*over_lap+crop_size, j*over_lap:j*over_lap+crop_size]
                    raw_crop_name = name+'_'+str(i).zfill(2)+'_'+str(j).zfill(2)+'.png'
                    label_crop_name = name+'_'+str(i).zfill(2)+'_'+str(j).zfill(2)+'.tiff'
                    cv2.imwrite(os.path.join(out_train_path, raw_crop_name), raw_crop)
                    cv2.imwrite(os.path.join(out_label_path, label_crop_name), label_crop)
                    f_txt.write(txt_raw_path+raw_crop_name+ ' ' +txt_label_path+label_crop_name)
                    f_txt.write('\n')
        f_txt.close()
    else:
        raise AttributeError('No this mode!')
    print('Done')
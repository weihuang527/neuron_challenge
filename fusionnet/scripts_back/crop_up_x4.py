import numpy as np 
import os 
from PIL import Image 
import cv2 
import shutil

train_path = '../train_result/complex/2020-01-02--14-36-26_complex_fusionnet_512_padding0_MSE_lr0001_feature32_iBN_data0_x4/result'
valid_path = '../valid_result/complex/2020-01-02--14-36-26_complex_fusionnet_512_padding0_MSE_lr0001_feature32_iBN_data0_x4/result'

out_train_path = '../../data/complex/crop_x2_1024/train_x4pd'
out_valid_raw_path = '../../data/complex/crop_x2_1024/valid_x4pd'

out_path = [out_train_path, out_valid_raw_path]
for ip in out_path:
    if not os.path.exists(ip):
        os.makedirs(ip)

train_txt = '../../data/data_list/complex_crop_x2_1024_train_x4pd_list.txt'
valid_txt = '../../data/data_list/complex_crop_x2_1024_valid_x4pd_list.txt'
pre_train_txt = '../../data/data_list/complex_crop_x2_1024_train_list.txt'
pre_valid_txt = '../../data/data_list/complex_crop_x2_1024_valid_list.txt'

txt_raw_path = 'complex/crop_x2_1024/train_x4pd/'
txt_valid_raw_path = 'complex/crop_x2_1024/valid_x4pd/'

pre_f_txt = open(pre_train_txt, 'r')
pre_f_valid_txt = open(pre_valid_txt, 'r')
pre_train_list = [x[:-1] for x in pre_f_txt.readlines()]
pre_valid_list = [x[:-1] for x in pre_f_valid_txt.readlines()]

f_txt = open(train_txt, 'w')
f_valid_txt = open(valid_txt, 'w')

crop_size = 1024
over_lap = 512
pad_size = 10240
out_img_size = pad_size // 2
num = (out_img_size - crop_size) // over_lap + 1

for k in range(16):
    ids = k * num * num
    name = pre_train_list[ids].split(' ')[0]
    file_name = name[31:]
    file_name = file_name[:-10]
    print('Processing train ' + file_name)
    img = cv2.imread(os.path.join(train_path, file_name+'.tiff'), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (out_img_size, out_img_size), interpolation=cv2.INTER_LINEAR)
    for i in range(num):
        for j in range(num):
            img_crop = img[i*over_lap:i*over_lap+crop_size, j*over_lap:j*over_lap+crop_size]
            img_crop_name = file_name+'_'+str(i).zfill(2)+'_'+str(j).zfill(2)+'.tiff'
            cv2.imwrite(os.path.join(out_train_path, img_crop_name), img_crop)
            f_txt.write(pre_train_list[ids+num*i+j] + ' ' + txt_raw_path+img_crop_name)
            f_txt.write('\n')

for k in range(4):
    name = pre_valid_list[k].split(' ')[0]
    file_name = name[31:]
    file_name = file_name[:-4]
    print('Processing valid ' + file_name)
    img = cv2.imread(os.path.join(valid_path, file_name+'.tiff'), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (out_img_size, out_img_size), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(out_valid_raw_path, file_name+'.tiff'), img)
    f_valid_txt.write(pre_valid_list[k] + ' ' + txt_valid_raw_path+file_name+'.tiff')
    f_valid_txt.write('\n')

f_txt.close()
f_valid_txt.close()
print('Done')

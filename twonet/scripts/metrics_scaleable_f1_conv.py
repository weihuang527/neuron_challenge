import numpy as np
import os
import cv2
import time
import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self,ks=9):
        super(Conv,self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=ks, stride=1, padding=ks//2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # m.weight.data.normal_(1, 0)
                # m.bias.data.fill_(0)
    
    def forward(self, x):
        return self.conv1(x)

def scaleable_f1(pre_img, label_img, kernel_size=9, black_boundary=True):
    """
    pre_img: The image of predicited result from network. It must be binary, dtype=bool(0 or 1) or uint8(0 or 255)
    label_img: The corresponding to groundtruth, dtype=uint8
    kernel_size: The size of kernel, natural number
    black_boundary: The foreground of computed
    """
    # h, w = label_img.shape
    is_binary_pred = np.unique(pre_img)
    assert len(is_binary_pred) <= 2, 'The pred image is not binary.'
    label_img = label_img.copy()
    is_binary_label = np.unique(label_img)
    if len(is_binary_label) > 2:
        label_img[label_img <= 122] = 0
        label_img[label_img > 122] = 255

    if np.max(pre_img) == 255:
        pre_img = pre_img // 255
    
    if np.max(label_img) == 255:
        label_img = label_img // 255
    
    if black_boundary == True:
        pre_img = 1 - pre_img
        label_img = 1 - label_img
    
    num_pred = np.sum(pre_img)
    num_label = np.sum(label_img)
    if num_pred > num_label * 2:
        f1 = 0
    else:
        conv_opt = Conv(kernel_size)
        label_img = label_img[np.newaxis, np.newaxis,:,:]
        pre_img = pre_img[np.newaxis, np.newaxis,:,:]
        label_img = label_img.astype(np.float32)
        torch_label = torch.tensor(label_img)
        convd_label = conv_opt(torch_label)
        del torch_label
        convd_label = convd_label.data.numpy()
        convd_label[convd_label>=1] = 1
        tp = np.multiply(convd_label, pre_img)
        del convd_label
        TP_pred = np.sum(tp)

        pre_img = pre_img.astype(np.float32)
        torch_pred = torch.tensor(pre_img)
        convd_pred = conv_opt(torch_pred)
        del torch_pred
        convd_pred = convd_pred.data.numpy()
        convd_pred[convd_pred>=1] = 1
        tp = np.multiply(convd_pred, label_img)
        del convd_pred
        TP_label = np.sum(tp)

        TP_plus_FN = np.sum(label_img)
        TP_plus_FP = np.sum(pre_img)
        del label_img, pre_img

        if TP_plus_FP == 0:
            precision = 0
        else:
            precision = TP_pred / TP_plus_FP
        recall = TP_label / TP_plus_FN

        sum_pr = precision + recall
        if sum_pr == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / sum_pr

    return f1


if __name__ == "__main__":
    # pred_path = '../data/simple/valid_result/mse_0001_no_tanh_none_t09'
    pred_path = '../data/complex/crop/valid_result/plain_4gpu'
    # pred_path = '../data/simple/valid_result/mse_0001_no_tanh_none_t09_drop_half'
    # label_path = '../data/simple/valid_label'
    label_path = '../data/complex/crop/valid_label'
    file_list = os.listdir(pred_path)
    f1_list = []
    kernel_size = 9
    time1 = time.time()
    for f in file_list:
        pre_img = np.asarray(cv2.imread(os.path.join(pred_path, f), cv2.IMREAD_GRAYSCALE))
        label_img = np.asarray(cv2.imread(os.path.join(label_path, f), cv2.IMREAD_GRAYSCALE))
        # pre_img = np.ones_like(label_img)
        # pre_img[512,512] = 0
        f1 = scaleable_f1(pre_img, label_img, kernel_size=kernel_size)
        print('The F1 score of ' + f + ' is %.6f' % f1)
        f1_list.append(f1)
    time2 = time.time()
    avg = sum(f1_list) / len(f1_list)
    print('The F1 score of validation data is %.6f' % avg)
    print('COST TIME:',(time2-time1))

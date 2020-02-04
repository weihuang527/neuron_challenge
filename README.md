1. Docker image: renwu527/auto-emseg:v4.1

2. Prepare training dataset:
python crop_x0.py -m=1
python crop_x0.py -m=2
python crop_x2.py -m=1
python crop_x2.py -m=2
python crop_x4.py -m=1
python crop_x4.py -m=2

3. Train Fusionnet
cd ./fusionnet/scipts
python main_fusionnet.py -c=complex_fusionnet_512_padding0_MSE_lr0001_feature32_iBN_data0_x0
python main_fusionnet.py -c=complex_fusionnet_512_padding0_MSE_lr0001_feature32_iBN_data0_x2
python main_fusionnet.py -c=complex_fusionnet_512_padding0_MSE_lr0001_feature32_iBN_data0_x4
python main_fusionnet.py -c=complex_fusionnet_512_padding0_MSE_lr0001_k2_x0
python main_fusionnet.py -c=complex_fusionnet_512_padding0_MSE_lr0001_k2_x2
python main_fusionnet.py -c=complex_fusionnet_512_padding0_MSE_lr0001_k2_x4

4. Train Twonet
cd ./twonet/scipts
python main_twonet.py -c=complex_twonet2_512_padding0_scale_x0
python main_twonet.py -c=complex_twonet2_512_padding0_scale_x2
python main_twonet.py -c=complex_twonet2_512_padding0_scale_x4
python main_twonet.py -c=complex_twonet2_512_padding0_scale_k2_x0
python main_twonet.py -c=complex_twonet2_512_padding0_scale_k2_x2
python main_twonet.py -c=complex_twonet2_512_padding0_scale_k2_x4

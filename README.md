## prediction maps, pretrained model, backbone model and datasets
1. The prediction map can be downloaded at https://pan.baidu.com/s/1ywbVzS7SHfXTEmVOtnG9_g?pwd=2qjc.

2. The pretrained model can be downloaded at https://pan.baidu.com/s/10gWYyESaV6LaCZv8_1PVBA?pwd=z8d1.

3. The backbone model can be downloaded at https://pan.baidu.com/s/16bKJaxewVfFXd73ICV6zDg?pwd=vn6v.

4. The datasets can be downloaded at https://pan.baidu.com/s/1Kpxaqv0n5YP-kDT8Id4wzA?pwd=pugv.

## environment
1. Python 3.8.13
2. others packages can be found at requirement.txt


## start


1. git clone https://github.com/CHANTILLY2023/MAMIFNet

2. conda create --name MAMIFNet python=3.8.12

3. conda activate MAMIFNet

4. pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

6. pip install -r requirements.txt 

7. pip install -U openmim 

8. mim install mmcv-full==1.5.2

## test
The relevant setting paths for data and weights are located in `config.py`. The default test is as follows:

1. create folder './data', download datasets file COD_dataset at https://pan.baidu.com/s/1Kpxaqv0n5YP-kDT8Id4wzA?pwd=pugv, unzip and move it to './data'

2. download backbone model at https://pan.baidu.com/s/16bKJaxewVfFXd73ICV6zDg?pwd=vn6v and unzip it.

3. download pretrained model at https://pan.baidu.com/s/10gWYyESaV6LaCZv8_1PVBA?pwd=z8d1 and unzip it.

4. run test.py



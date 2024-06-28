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

2. conda create --name MAMIFNet python=3.8.13

3. conda activate MAMIFNet

4. pip install -r requirements.txt 

5. pip install -U openmim 

6. mim install mmcv-full==1.5.2



## test
The relevant settings for data and weights are in cofig.py. The default test is as follows：

1. create folder './data'

2. download datasets file COD_dataset at https://pan.baidu.com/s/1Kpxaqv0n5YP-kDT8Id4wzA?pwd=pugv, unzip and move it to './data'

3. create folder './PVTv2_Seg'

4. download backbone model at https://pan.baidu.com/s/10gWYyESaV6LaCZv8_1PVBA?pwd=z8d1 and unzip it.

5. download pretrained model at https://pan.baidu.com/s/10gWYyESaV6LaCZv8_1PVBA?pwd=z8d1.

6. run test.py





## Updates

- 28 Jun 2024: Open-sourced model and test code
- 19 Aug 2024: Released prediction map, pretrained model, backbone, and datasets

## Downloads

| Item | Baidu Pan | Google Drive |
|------|-----------|--------------|
| Prediction Map | [Link](https://pan.baidu.com/s/1GhUNZ02qzGDXSfhE0tj2iw?pwd=ty3a) | [Link](https://drive.google.com/file/d/1p9S4RSqCu_1FggRwlImP66royenjAg-R/view?usp=sharing) |
| Pretrained Model | [Link](https://pan.baidu.com/s/1P1zZpFGI0_74S5nofSjjRg?pwd=s3c4) | [Link](https://drive.google.com/file/d/1rfnFfUZL3bv-zg8hOLWbHoStI3IpL2Fe/view?usp=sharing) |
| Backbone Model | [Link](https://pan.baidu.com/s/1pyGVL2mpxbX3g39T1bTMIg?pwd=qdhg) | [Link](https://drive.google.com/file/d/1-ueav8i01-kOTsRcjbdYrO6M0VZo5OpI/view?usp=sharing) |
| Datasets | [Link](https://pan.baidu.com/s/1hV0ffZbIA9I2nCZ4nYUaSg?pwd=x6w5) | [Link](https://drive.google.com/drive/folders/11_eIwD3yrc9KdD-6RJtzVGbwQuddnQSq?usp=sharing) |

Note: Training source code will be released upon paper acceptance.

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



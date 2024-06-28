
## prediction maps and pretrained model
1. The prediction map of our method can be downloaded at https://drive.google.com/drive/folders/1rZ9IrbFz4dggik1vnK53PnNNy7l-1X_r?usp=sharing.
2. The pretrained model of our model can be downloaded at https://drive.google.com/drive/folders/1n80O0RIAe4KT08SZG-UK6HsWgD_qouoQ?usp=sharing.
## environment
1. Python 3.8.13
2. others packages can be found at requirement.txt


## start
git clone https://github.com/forever-rz/MAMIFNET \

conda create --name MAMIFNet python=3.8.13

conda activate MAMIFNet

pip install -r requirements.txt 

pip install -U openmim \
      mim install mmcv-full==1.5.2

## prediction maps, pretrained model, backbone model and datasets
1.The prediction map of our method can be downloaded at https://pan.baidu.com/s/1yfELKFq4cQJt1GSRiCaI4Q?pwd=t0t3.

2.The pretrained model of our network can be downloaded at https://pan.baidu.com/s/1kNyRyYclrTnY-NbxQbbukg?pwd=q7fn.

3.The backbone model of our model can be downloaded at https://pan.baidu.com/s/1ESkV-ol_VzVAbDkDCXxNgQ?pwd=ib2d.

4.The datasets of our method can be downloaded at https://pan.baidu.com/s/1F2-vuEkGz6CaCg5QE-xuOQ?pwd=v2lt.


## environment
1. Python 3.8.13
2. others packages can be found at requirement.txt


## start
git clone https://github.com/forever-rz/MAMIFNET 

conda create --name MAMIFNet python=3.8.13

conda activate MAMIFNet

pip install -r requirements.txt 

pip install -U openmim 

mim install mmcv-full==1.5.2


## test
Create folder './data'

download datasets file COD_dataset(https://pan.baidu.com/s/1F2-vuEkGz6CaCg5QE-xuOQ?pwd=v2lt), unzip and move it to './data'

download pretrained model( https://pan.baidu.com/s/1kNyRyYclrTnY-NbxQbbukg?pwd=q7fn)

run test.py

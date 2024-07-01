from collections import OrderedDict
import os
import time
import datetime
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean
import torch.nn.functional as F
import torchvision.utils as utils
from config import *
from misc import *
import sys
import cv2
from numpy import mean
import torch.nn.functional as F
import torchvision.utils as utils
from config import *
from misc import *
import sys
import cv2
import time
import torch
from PIL import Image
from torchvision import transforms
torch.manual_seed(2023)
sys.path.append('../')
from metric_caller import CalTotalMetric
from excel_recorder import MetricExcelRecorder
#ACC


def main(exp_name,net,scale,results_path,pth_path,pth_list,excel_path):
    path_excel = excel_path
    pth_root_path = pth_path
    results_root_path = results_path
    for pth in pth_list:
        pth=str(pth)
        results_path = os.path.join(results_root_path,pth)
        check_mkdir(results_path)
        pth_path = os.path.join(pth_root_path,pth)+'.pth'
        to_test = OrderedDict([
            ('CHAMELEON', chameleon_path),
            ('CAMO', camo_path),
            ('COD10K', cod10k_path),
            ('NC4K', nc4k_path)
        ])
        results = OrderedDict()
        img_transform = transforms.Compose([
            transforms.Resize((scale,scale)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        to_pil = transforms.ToPILImage()
        # net.load_state_dict(torch.load(pth_path))
        # Load the state_dict

        state_dict = torch.load(pth_path)

        missing, un =net.load_state_dict(state_dict,strict=False)

        print("Missing keys in state_dict:", missing)
        print("Number of missing keys:", len(missing))
        print("Unexpected keys in state_dict:", un)
        print("Number of unexpected keys:", len(un))
        
        print('Load {} succeed!'.format(exp_name+'_'+pth+'.pth'))
        
        net.eval()

        with torch.no_grad():
            excel_logger = MetricExcelRecorder(xlsx_path=path_excel, dataset_names=[name for name, root in to_test.items()],metric_names = ["mae", "meanem", "smeasure", "meanfm", "wfmeasure","adpfm",  "maxfm", "adpem", "maxem"])
            start = time.time()
            # v1
            for name, root in to_test.items():
                cal_total_seg_metrics = CalTotalMetric()
                time_list = []
                if 'NC4K' in name:
                    image_path = os.path.join(root, 'Imgs')

                else:
                    image_path = os.path.join(root, 'Image')
        

                mask_path = os.path.join(root, 'GT')
                check_mkdir(os.path.join(results_path, name))
                img_suffix = 'jpg'
                img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith(img_suffix)]

                for img_name in tqdm(img_list, desc="Processing images"):
                    # Use the tqdm instance's write method to ensure the message starts on a new line
                    tqdm.write(f"\nProcessing image {img_name}")
                   
                    img = Image.open(os.path.join(image_path, img_name + '.' + img_suffix)).convert('RGB')

                    mask = np.array(Image.open(os.path.join(mask_path, img_name + '.png')).convert('L'))

                    w, h = img.size
                    # w_mask, h_mask = mask.size
                    h_mask, w_mask = mask.shape
                    img_var = Variable(img_transform(img).unsqueeze(0)).cuda()

                    start_each = time.time()

                    predictions = net(img_var)
                    prediction = predictions[-1]
                    prediction = torch.sigmoid(prediction)
                    time_each = time.time() - start_each
                    time_list.append(time_each)

                    prediction = np.array(transforms.Resize((h_mask, w_mask))(to_pil(prediction.data.squeeze(0).cpu())))
                    Image.fromarray(prediction).convert('L').save(
                        os.path.join(results_path, name, img_name + '.png'))

                    cal_total_seg_metrics.step(prediction, mask, mask_path)
                print(('{}'.format(exp_name+'_'+str(pth))))
                print("{}'s average Time Is : {:.3f} s".format(name, mean(time_list)))
                print("{}'s average Time Is : {:.1f} fps".format(name, 1 / mean(time_list)))
                results = cal_total_seg_metrics.get_results()
                excel_logger(row_data=results, dataset_name=name, method_name=exp_name+'-'+pth)
                print(results)
        end = time.time()
        print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))


def evluation_with_resultspath(results_path,path_excel):
    print(results_path)
    _, exp_name = os.path.split(results_path)
    to_test = OrderedDict([
        ('CHAMELEON', chameleon_path),
        ('CAMO', camo_path),
        ('COD10K', cod10k_path),
        ('NC4K', nc4k_path)
    ])
    excel_logger = MetricExcelRecorder(xlsx_path=path_excel, dataset_names=[name for name, root in to_test.items()],
                                       metric_names=["mae", "meanem", "smeasure", "meanfm", "wfmeasure", "adpfm",
                                                     "maxfm", "adpem", "maxem"])

    for name, root in to_test.items():
        print(os.path.join(results_path,name))
        if not os.path.exists(os.path.join(results_path,name)):
            continue
        print(name)
        cal_total_seg_metrics = CalTotalMetric()
        image_path = os.path.join(root, 'Imgs')
        mask_path = os.path.join(root, 'GT')
        img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
        for idx, img_name in enumerate(img_list):
            
            result_img_path = os.path.join(results_path, name, img_name + '.png')
        
            prediction = Image.open(result_img_path).convert('L')

            mask = Image.open(os.path.join(mask_path, img_name + '.png')).convert('L')
            if not prediction.size == mask.size:
                mask = mask.resize(prediction.size)
            cal_total_seg_metrics.step(np.array(prediction), np.array(mask), mask_path)

        results = cal_total_seg_metrics.get_results()
        print(results)
        excel_logger(row_data=results, dataset_name=name, method_name=exp_name)

def evaluation_COD(exp_name,net,scale,results_path,pth_path,pth_list,excel_path):
    main(exp_name, net, 384, results_path, pth_path,pth_list,excel_path)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    exp_name = 'MAMIFNet_v1'
    version = exp_name.split("_v")[1].split("'")[0]
    from MAMIFNet  import MAMIFNet
    net = MAMIFNet('pvt_v2_b4').cuda()

    pth_list =  [60]

    pth_path = os.path.join('ckpt/',exp_name)

    result_name = 'results'+version
    results_path = os.path.join(result_name,exp_name)
    excel_name = 'results_'+version
    excel_path = './'+excel_name+'.xlsx'
    main(exp_name, net, 384, results_path, pth_path,pth_list,excel_path)

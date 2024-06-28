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

def val_model(exp_name,net,scale,results_path,pth_path,epoch,excel_path,save_predict_images=False):
    path_excel = excel_path
    pth_root_path = pth_path
    results_root_path = results_path

    pth=str(epoch)
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

    # Create a new state_dict with modified keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key=key
        # if 'module' in new_key and 'module.module' not in new_key:
        #     new_key = new_key.replace('module', 'module.module')
        # if 'aa' in new_key and 'oaa' not in new_key:
        #     new_key = new_key.replace('aa', 'oaa')
        new_state_dict[new_key] = value
    # # Load the new state_dict into the network
    missing, un = net.load_state_dict(new_state_dict)
    print('Load {} succeed!'.format(exp_name + '_' + pth + '.pth'))
    net.eval()
    # path_excel = './results_excel5.xlsx'

    with torch.no_grad():
        # excel_logger = MetricExcelRecorder(xlsx_path=path_excel, dataset_names=[name for name, root in to_test.items()])
        excel_logger = MetricExcelRecorder(xlsx_path=path_excel, dataset_names=[name for name, root in to_test.items()],metric_names = ["mae", "meanem", "smeasure", "meanfm", "wfmeasure","adpfm",  "maxfm", "adpem", "maxem"])
        start = time.time()
        # v1
        for name, root in to_test.items():
            cal_total_seg_metrics = CalTotalMetric()
            time_list = []
            # if 'NC4K' in name:
            #     image_path = os.path.join(root, 'Imgs')
            #
            # else:
            image_path = os.path.join(root, 'Imgs')
            mask_path = os.path.join(root, 'GT')
            # check_mkdir(os.path.join(results_path, exp_name, name))
            check_mkdir(os.path.join(results_path, name))
            img_suffix = 'jpg'
            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith(img_suffix)]
            # sorted_img_list = sorted(img_list, key=lambda x: int(os.path.splitext(x)[0]))

            # img_list = sorted(img_list, key=lambda x: int(os.path.splitext(x.split('-')[1])[0]))

            for img_name in tqdm(img_list, desc="Processing images"):
                # Use the tqdm instance's write method to ensure the message starts on a new line
                tqdm.write(f"\nProcessing image {img_name}")
                # if '60' in img_name:
                #     print(1)
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
                # prediction = torch.sigmoid(prediction[0])
                time_each = time.time() - start_each
                time_list.append(time_each)

                # prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))
                prediction = np.array(transforms.Resize((h_mask, w_mask))(to_pil(prediction.data.squeeze(0).cpu())))
                if save_predict_images:
                    Image.fromarray(prediction).convert('L').save(
                        # os.path.join(results_path, exp_name, name, img_name + '.png'))
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
        print(missing)
        print(len(missing))
        print(un)
        print(len(un))

        print('Load {} succeed!'.format(exp_name+'_'+pth+'.pth'))



        net.eval()
        # path_excel = './results_excel5.xlsx'

        with torch.no_grad():
            # excel_logger = MetricExcelRecorder(xlsx_path=path_excel, dataset_names=[name for name, root in to_test.items()])
            excel_logger = MetricExcelRecorder(xlsx_path=path_excel, dataset_names=[name for name, root in to_test.items()],metric_names = ["mae", "meanem", "smeasure", "meanfm", "wfmeasure","adpfm",  "maxfm", "adpem", "maxem"])
            start = time.time()
            # v1
            for name, root in to_test.items():
                cal_total_seg_metrics = CalTotalMetric()
                time_list = []
                # if 'NC4K' in name:
                #     image_path = os.path.join(root, 'Imgs')
                #
                # else:
                #     image_path = os.path.join(root, 'Image')
                image_path = os.path.join(root, 'Imgs')

                mask_path = os.path.join(root, 'GT')
                # check_mkdir(os.path.join(results_path, exp_name, name))
                check_mkdir(os.path.join(results_path, name))
                img_suffix = 'jpg'
                img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith(img_suffix)]
                # sorted_img_list = sorted(img_list, key=lambda x: int(os.path.splitext(x)[0]))

                # img_list = sorted(img_list, key=lambda x: int(os.path.splitext(x.split('-')[1])[0]))

                for img_name in tqdm(img_list, desc="Processing images"):
                    # Use the tqdm instance's write method to ensure the message starts on a new line
                    tqdm.write(f"\nProcessing image {img_name}")
                    # if '60' in img_name:
                    #     print(1)
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
                    # prediction = torch.sigmoid(prediction[0])
                    time_each = time.time() - start_each
                    time_list.append(time_each)

                    # prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))
                    prediction = np.array(transforms.Resize((h_mask, w_mask))(to_pil(prediction.data.squeeze(0).cpu())))
                    Image.fromarray(prediction).convert('L').save(
                        # os.path.join(results_path, exp_name, name, img_name + '.png'))
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
        # ('CHAMELEON', chameleon_path),
        ('CAMO', camo_path),
        # ('COD10K', cod10k_path),
        # ('NC4K', nc4k_path)
    ])
    # excel_logger = MetricExcelRecorder(xlsx_path=path_excel, dataset_names=[name for name, root in to_test.items()])
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
            # if not os.path.exists(result_img_path):
            #     os.rename(os.path.join(results_path, name, img_name + '.tif'), result_img_path)
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
    # from SARNet import SARNet
    from MAMIFNet  import MAMIFNet
    net = MAMIFNet('pvt_v2_b4').cuda()

    pth_list =  [60]

    pth_path = os.path.join('ckpt/',exp_name)

    result_name = 'results'+version
    results_path = os.path.join(result_name,exp_name)
    excel_name = 'results_'+version
    excel_path = './'+excel_name+'.xlsx'
    main(exp_name, net, 384, results_path, pth_path,pth_list,excel_path)

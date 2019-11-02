# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import cv2
import math
from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test
from torch.utils.data  import Dataset


class galleryDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataList = []
        for f in os.listdir(path):
            self.dataList.append(os.path.join(self.path, f))

    def __getitem__(self,index):
        png_path = self.dataList[index]
        img = cv2.imread(png_path)
        assert img is not None, "File Not Found in " + png_path
        h,w,_ = img.shape
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) 
        img /= 255.0 
        return torch.from_numpy(img)

    def __len__(self):
        return len(self.dataList)

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
# parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
opt = parser.parse_args()

# 加载配置文件 
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16'] 
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 
str_ids = opt.gpu_ids.split(',')
which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

# 数据处理，加载以及变化
data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
        #   [transforms.ToTensor()(crop) 
        #      for crop in crops]
        # )),
        #transforms.Lambda(lambda crops: torch.stack(
        #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
        #       for crop in crops]
        # ))
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])


data_dir = test_dir

# if opt.multi:
    # image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['query','multi-query']}
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
    #                                          shuffle=False, num_workers=16) for x in ['query','multi-query']}
# else:
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['query']}

# gallery部分进行重构，重写datasets
gallery_datasets = galleryDataset(os.path.join(data_dir, "gallery"))
gallery_dataloaders = torch.utils.data.DataLoader(gallery_datasets, batch_size=opt.batchsize, 
                                              shuffle=False, num_workers=16)

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

# 加载模型
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


# 提取特征
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

# 为了顺序，以及对应关系
# 与query_feature相对应
# f=open("./query_label_list.txt","w")
# query_label_list = []
# q_path = os.path.join(data_dir,'query')
# for i in os.listdir(q_path):
#     t_path = os.path.join(q_path,i)
#     if os.path.isdir(t_path):
#         for j in os.listdir(t_path):
#             query_label_list.append(j)

# print(query_label_list)
# f.write(str(query_label_list))
# f.close()

# 提取query中的特征, label是按照顺序读入的，从0到1347
def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        print("+"*5, label)
        print("-"*5,type(img))
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

#提取gallery中的特征
def extract_gallery_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img = data
        print("-"*5,type(img),img.size())
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

# def get_id(img_path):
#     camera_id = []
#     labels = []
#     for path, v in img_path:
#         #filename = path.split('/')[-1]
#         filename = os.path.basename(path)
#         label = filename[0:4]
#         camera = filename.split('c')[1]
#         if label[0:2]=='-1':
#             labels.append(-1)
#         else:
#             labels.append(int(label))
#         camera_id.append(int(camera[0]))
#     return camera_id, labels

def get_labels(img_path):
    labels = []
    for path, _ in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
    return labels

# 数据集中的gallery和query
# gallery_path = gallery_datasets.imgs
query_path = image_datasets['query'].imgs

# 获取他们的ID和label, 是通过解析路径判断的
# 这里的gallery是没有ID且没有camera
# gallery_cam,gallery_label = get_id(gallery_path)
# query 有ID， 但是没有camera
query_label = get_labels(query_path)

# if opt.multi:
#     mquery_path = image_datasets['multi-query'].imgs
#     mquery_cam,mquery_label = get_id(mquery_path)

# 构建模型，并开始加载
print('-----------testing-----------')
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses)
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses)
else:
    model_structure = ft_net(opt.nclasses, stride = opt.stride)
if opt.PCB:
    model_structure = PCB(opt.nclasses)

#if opt.fp16:
#    model_structure = network_to_half(model_structure)

# 加载模型
model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if opt.PCB:
    if opt.fp16:
       model = PCB_test(model[1])
    else:
        model = PCB_test(model)
else:
    if opt.fp16:
        model[1].model.fc = nn.Sequential()
        model[1].classifier = nn.Sequential()
    else:
        model.classifier.classifier = nn.Sequential()

# 改成eval模式，将模型传送至gpu
model = model.eval()
if use_gpu:
    model = model.cuda()

# 分别提取query和gallery中的feature
with torch.no_grad():
    # 分别得到两个部分的features
    print("-"*10,"extracting feature from query","-"*10)
    query_feature = extract_feature(model,dataloaders['query'])
    print("-"*10,"extracting feature from gallery","-"*10)
    gallery_feature = extract_gallery_feature(model,gallery_dataloaders)
    # if opt.multi:
    #     mquery_feature = extract_feature(model,dataloaders['multi-query'])
    
# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(), # gallery 中图片的feature
        #   'gallery_label':gallery_label, # 通过解析文件路径得到的ID
        #   'gallery_cam':gallery_cam, # gallery 中得到的camera ID
            'query_f':query_feature.numpy(), # query 文件夹中图片的feature
            'query_label':query_label} # 通过解析文件路径得到的ID
        #   'query_cam':query_cam}  # query 中得到的camera ID

scipy.io.savemat('pytorch_result.mat',result)

# print(opt.name)
# result = './model/%s/result.txt'%opt.name
# os.system('python evaluate_gpu.py | tee -a %s'%result)

# if opt.multi:
#     result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
#     scipy.io.savemat('multi_query.mat',result)

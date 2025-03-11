#  在CS-KD基础上改进

from medmnist import OCTMNIST
import argparse
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
import sys
import math
import numpy as np
from sklearn import metrics
from Net import *
from datapreprocess import *
from utils import *
import shutil
import setproctitle

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
 
class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss
    
class DKDLoss(nn.Module):
    def __init__(self, temp_factor, t_error):
        super(DKDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.t_error = t_error
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input_1, input_2, target):
        
        tea_error = self.t_error * self.temp_factor * torch.ones_like(input_1)
        indexs = torch.arange(len(input_1))
        tea_error[indexs, target] = self.temp_factor
        
        log_p = torch.log_softmax(input_1/tea_error, dim=1)
        q = torch.softmax(input_2/tea_error, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input_1.size(0)
        
        
#         log_p = torch.log_softmax(input/self.temp_factor, dim=1)
#         q = torch.softmax(target/self.temp_factor, dim=1)
#         loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss

    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=128)    # for medmnist 
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--temp', default=4.0, type=float, help='temperature scaling')
    parser.add_argument('--lamda', default=1.0, type=float, help='cls loss weight ratio')
    parser.add_argument('--alpha_T',default=0.8 ,type=float, help='alpha_T')
    parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--t_error',default=0.1 ,type=float, help='alpha_T')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--save',default='./result/OCTMnist_32_ours_gamma_resnet18_lr0.01_error10')

    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_known_args()[0]
    device = torch.device('cuda:0')

    setproctitle.setproctitle(args.save)
    criterion = nn.CrossEntropyLoss()
#     criterion_2 = KDLoss(args.temp)
    criterion_2 = DKDLoss(args.temp,args.t_error)


    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    normTransform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    trainTransform = transforms.Compose([
        transforms.RandomResizedCrop(32,scale=(0.8, 1.0)),
#         transforms.Resize((32,32)),     # for medmnist
#         transforms.RandomResizedCrop(224),   # for image size 224x244
#         transforms.RandomRotation(10),  # for image size 224x244
        gammaCrop(0.5,2),
        transforms.ColorJitter(contrast=0.5, saturation=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.Resize((32,32)),
#         transforms.Resize((224,224)),
        transforms.ToTensor(),
        normTransform
    ])
    
    BCdata_trian = OCTMNIST(split='train', transform=ContrastiveLearningViewGenerator(trainTransform,n_views=2),download=True)
    BCdata_test = OCTMNIST(split='test', transform=ContrastiveLearningViewGenerator(testTransform,n_views=1),download=True)
#     BCdata_trian = PathMNIST(split='train', transform=ContrastiveLearningViewGenerator(trainTransform,n_views=2),download=True)
#     BCdata_test = PathMNIST(split='test', transform=ContrastiveLearningViewGenerator(testTransform,n_views=1),download=True)
#     BCdata_trian = DermaMNIST(split='train', transform=ContrastiveLearningViewGenerator(trainTransform,n_views=2),download=True)
#     BCdata_test = DermaMNIST(split='test', transform=ContrastiveLearningViewGenerator(testTransform,n_views=1),download=True)
#     BCdata_trian = OrganAMNIST(split='train', transform=ContrastiveLearningViewGenerator(trainTransform,n_views=2),download=True)
#     BCdata_test = OrganAMNIST(split='test', transform=ContrastiveLearningViewGenerator(testTransform,n_views=1),download=True)
#     BCdata_trian = OrganCMNIST(split='train', transform=ContrastiveLearningViewGenerator(trainTransform,n_views=2),download=True)
#     BCdata_test = OrganCMNIST(split='test', transform=ContrastiveLearningViewGenerator(testTransform,n_views=1),download=True)
#     BCdata_trian = OrganSMNIST(split='train', transform=ContrastiveLearningViewGenerator(trainTransform,n_views=2),download=True)
#     BCdata_test = OrganSMNIST(split='test', transform=ContrastiveLearningViewGenerator(testTransform,n_views=1),download=True)
    
    trainLoader = DataLoader(BCdata_trian, batch_size=args.batchSz, shuffle=True, drop_last=True, num_workers=4)
    testLoader = DataLoader(BCdata_test, batch_size=args.batchSz, shuffle=False, drop_last=False, num_workers=4)
    
    net = Student_resnet18(first_conv=False, class_num=args.class_num)
#     net = Student_resnet18(first_conv=True, class_num=2)
    net = net.to(device) 

    
    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr = args.lr,betas=(0.9, 0.999), weight_decay=1e-4)
    
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size= 10,
        iter_max= 100,
#         iter_max= 200,    #  200 for ISIC, 100 for medmnist
        power= 0.9,
    )

    
    for epoch in range(1, args.nEpochs + 1):
        train(args, epoch, net, trainLoader, optimizer, criterion,criterion_2,scheduler)
        test(args, epoch, net, testLoader,optimizer, criterion)
        torch.save(net, os.path.join(args.save, str(epoch)+'.pth'))


    trainF.close()
    testF.close()
    

    
def train(args, epoch, net, trainLoader, optimizer, criterion,criterion_2,scheduler):
    net.train()                                       # 设置网络为训练模式
    
 
    print('\n epoch   **********   ',epoch)
    nProcessed = 0
    total_loss = 0
    total_correct = 0
    nTrain = len(trainLoader.dataset)
#     print(nTrain)
    for batch_idx, (pos_1, target) in enumerate(trainLoader):

        # cross entropy loss 
        images_1 = pos_1[0].cuda(0)
        images_2 = pos_1[1].cuda(0)
        correct = 0
        target = target.cuda(0).long().view(-1)
        
#         images_2,_ = gammamix(images_2,0.5,2)
        # gammamix
        with torch.no_grad():
            # another view
            _,output_last = net(images_1)
            
        _,output = net(images_2)
        
        loss_1 = criterion(output,target)
#         loss_2 = criterion_2(output, output_last.detach())
        loss_2 = criterion_2(output, output_last.detach(),target)
        loss = loss_1 + args.lamda * loss_2
        

        total_loss = loss.item()+total_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        prediction = torch.argmax(output, 1)
        correct += (prediction == target).sum().int().cpu().numpy()
        total_correct = total_correct+correct

        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        if((batch_idx%100)==0):
            print("partialEpoch={},total_loss = {},ACC = {} \n".format(partialEpoch,loss.item(),correct/args.batchSz))
    
    scheduler.step()
    
    print("Train epoch = {}, total_loss = {},  ACC = {}  \n".format(epoch,(total_loss/nTrain),(total_correct/nTrain)))




def test(args, epoch, net, testLoader,optimizer, criterion):
    net.eval()
    total_loss = 0
    total_correct = 0
    conMatrix_pre = []
    conMatrix_tar = []
    AUC_data = []
    AUC_target = []
    sensitivity = 0
    specificity = 0
    nTrain = len(testLoader.dataset)
    

    with torch.no_grad():
        for pos_1, target in testLoader:
#         for pos_1, target, _ in testLoader:
#         for pos_1, _ , target in testLoader:
            # cross entropy loss
            images = pos_1.cuda(0)
            target = target.cuda(0).long().view(-1)
            
            _,output = net(images)
            loss = criterion(output,target)
            total_loss = loss.item()+total_loss
            b,_ = output.size()
            output = F.softmax(output,dim=1)                 
#             print(prediction)
            for i in range(len(output)):
                conMatrix_pre.append(output[i][1].cpu().detach())   #  for 2 class
#                 conMatrix_pre.append(output[i].cpu().detach().numpy())
                conMatrix_tar.append(target[i].cpu().detach().numpy())
                
            prediction = torch.argmax(output, 1)      
            total_correct += (prediction == target).sum().int().cpu().numpy()
            
#     print(conMatrix_pre)
    test_AUC = metrics.roc_auc_score(np.array(conMatrix_tar), np.array(conMatrix_pre))
#     test_AUC = metrics.roc_auc_score(np.array(conMatrix_tar), np.array(conMatrix_pre), multi_class='ovo')

    print("Test epoch = {}, total_loss = {:.6f}, ACC = {:.4f}, AUC = {:.4f} \n"
          .format(epoch,(total_loss/nTrain),(total_correct/nTrain),test_AUC))


if __name__=='__main__':
    main()

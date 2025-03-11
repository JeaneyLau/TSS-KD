import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

    
class Student_resnet18(nn.Module):
    def __init__(self,first_conv,class_num):
        super(Student_resnet18, self).__init__()
       
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 128)
        if(first_conv==False):
            self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
            self.resnet18.maxpool = nn.Identity()
            
        self.projector = nn.Sequential(nn.Linear(128, 128, bias=False), nn.BatchNorm1d(128),
                               nn.ReLU(inplace=True), nn.Linear(128, class_num, bias=False))
       
    def forward(self, x):
        features = self.resnet18(x)
        y = self.projector(features)

        return features,y   
    



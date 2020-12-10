# deep learning
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18, resnet50, resnet34
from torchvision.models.mobilenet import mobilenet_v2
from torch.nn import functional as F
import timm


# Lyft Model for ResNet
class LyftModel(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        # set pretrained=True while training
        self.backbone = resnet50(pretrained=True) 
        
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            # num_in_channels = 25
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )
        
        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        backbone_out_features = 2048
        
        # X, Y coords for the future positions (output shape: Bx50x2)
        num_targets = 2 * cfg["model_params"]["future_num_frames"]

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.logit = nn.Linear(4096, out_features=num_targets)
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.head(x)
        x = self.logit(x)
        
        return x


class LyftMixModel(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        # set pretrained=True while training
        self.backbone = timm.create_model('mixnet_xl',pretrained=True)
        
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv_stem = nn.Conv2d(
            num_in_channels,
            self.backbone.conv_stem.out_channels,
            kernel_size=self.backbone.conv_stem.kernel_size,
            stride=self.backbone.conv_stem.stride,
            padding=self.backbone.conv_stem.padding,
            bias=False,
        )
        
        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        backbone_out_features = 1536
        
        # X, Y coords for the future positions (output shape: Bx50x2)
        num_targets = 2 * cfg["model_params"]["future_num_frames"]

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.logit = nn.Linear(4096, out_features=num_targets)
        
    def forward(self, x):
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        
        x = self.backbone.blocks(x)

        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)
        x = self.backbone.global_pool(x)
        x = torch.flatten(x, 1)
        
        x = self.head(x)
        x = self.logit(x)
        
        return x


# Lyft Model for MixNet
class LyftMixnet(nn.Module):
    
    def __init__(self, cfg, classify='mixnet_l'):  # if classify = 'mixnet_m', then this is a mixnet_m model
        super().__init__()
        
        # set pretrained=True while training
        self.backbone = timm.create_model(classify, pretrained=False) 
        
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        
        self.backbone.conv_stem = nn.Conv2d(
            # num_in_channels = 25
            num_in_channels,
            self.backbone.conv_stem.out_channels,
            kernel_size=self.backbone.conv_stem.kernel_size,
            stride=self.backbone.conv_stem.stride,
            padding=self.backbone.conv_stem.padding,
            bias=False,
        )
        
        # X, Y coords for the future positions (output shape: Bx50x2)
        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        self.logit = nn.Linear(self.backbone.classifier.out_features, out_features=num_targets)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.logit(x)
        return x


# Lyft Model for MobileNet
class LyftMobile(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        # set pretrained=True while training
        self.backbone = mobilenet_v2(pretrained=True) 
        
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.features[0][0] = nn.Conv2d(
            # num_in_channels = 25
            num_in_channels,
            self.backbone.features[0][0].out_channels,
            kernel_size=self.backbone.features[0][0].kernel_size,
            stride=self.backbone.features[0][0].stride,
            padding=self.backbone.features[0][0].padding,
            bias=False,
        )
                
        # X, Y coords for the future positions (output shape: Bx50x2)
        num_targets = 2 * cfg["model_params"]["future_num_frames"]

        # Fully connected layer.
        self.backbone.classifier[1] = nn.Linear(
            in_features=self.backbone.classifier[1].in_features,
            out_features=num_targets
        )

        
    def forward(self, x):
        x = self.backbone(x)
        return x

import torch
import logging
import torchvision
from torch import nn
from typing import Tuple
import math
from cosplace_model.layers import Flatten, L2Norm, GeM
from backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
}


class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone : str, fc_output_dim : int, train_all_layers : bool = False):
        """Return a model for GeoLocalization.
        
        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
            train_all_layers (bool): whether to freeze the first layers of the backbone during training or not.
        """
        super().__init__()
        assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.backbone, features_dim = get_backbone(backbone, train_all_layers)
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            L2Norm()
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x

class MutualVPR(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, pretrained_foundation = False, foundation_model_path = None,output_dim=512):
        super().__init__()
        self.backbone = get_backbone_vit(pretrained_foundation, foundation_model_path)
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(768, output_dim),
            L2Norm()
        )
        # self.linear = nn.Linear(768, output_dim)
        
    def forward(self, x):
        x = self.backbone(x)
        B,P,D = x["x_prenorm"].shape
        W = H = int(math.sqrt(P-1))
        x_p = x["x_norm_patchtokens"].view(B,W,H,D).permute(0, 3, 1, 2) 
        # x_cls = x["x_norm_clstoken"]
        # x_cls = self.linear(x_cls)
        # x_cls = torch.nn.functional.normalize(x_cls, p=2, dim=-1)
        x = self.aggregation(x_p)  
        return x



def get_pretrained_torchvision_model(backbone_name : str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model


def get_backbone(backbone_name : str, train_all_layers : bool) -> Tuple[torch.nn.Module, int]:
    backbone = get_pretrained_torchvision_model(backbone_name)
    if backbone_name.startswith("ResNet"):
        if train_all_layers:
            logging.debug(f"Train all layers of the {backbone_name}")
        else:
            for name, child in backbone.named_children():
                if name == "layer3":  # Freeze layers before conv_3
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")

        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        if train_all_layers:
            logging.debug("Train all layers of the VGG-16")
        else:
            for layer in layers[:-5]:
                for p in layer.parameters():
                    p.requires_grad = False
            logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
    backbone = torch.nn.Sequential(*layers)
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim

def get_backbone_vit(pretrained_foundation, foundation_model_path):
    backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)  
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    return backbone
import torch
import torch.nn as nn

import timm


class VGG16_regressor(nn.Module):
    """Custom VGG16 model for regression task"""
    def __init__(self, num_classes=1):
        """Constructor
        Parameters
        ----------
            num_classes : int
                Number of classes
        """
        super(VGG16_regressor, self).__init__()
        self.model = timm.create_model("vgg16", pretrained=True)
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes)

    def forward(self, x):
        """Forward pass"""
        return self.model(x)


def get_model(model_name, num_classes=1):
    """Returns the model
    Parameters
    ----------
        model_name : str
            Name of the model
        num_classes : int
            Number of classes
    Returns
    -------
        model : nn.Module
            Model
    """
    if model_name == 'vgg16':
        return VGG16_regressor(num_classes)
    else:
        raise NotImplementedError



if __name__ == "__main__":
    model = get_model('vgg16')
    print(model)

    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {p}")
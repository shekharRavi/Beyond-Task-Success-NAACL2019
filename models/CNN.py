import torch
import torch.nn as nn
from torchvision.models import resnet

use_cuda = torch.cuda.is_available()

class ResNet(nn.Module):
    """
    This is a placeholder till the features are stored. Also, this exists to get final mile improvements by fine-tuning the ResNet.
    """
    def __init__(self):
        super(ResNet, self).__init__()

        cnn = resnet.resnet152(pretrained=True)
        self.modifiedCNN = nn.Sequential(*list(cnn.children())[:-2])
        self.avgpool = nn.AvgPool2d(7, stride=7)

    def forward(self, img):
        """Short summary.

        Parameters
        ----------
        img : Tensor
            Image tensor after the torch.transform preprocessing as input to the ResNet 152

        Returns
        -------
        img_features: Last conv layer of the ResNet 152
        avg_img_features : Avg pool layer of the ResnNet 152

        """

        img_features = self.modifiedCNN(img)
        avg_img_features = self.avgpool(img_features)

        img_features = img_features.view(img_features.size(0), img_features.size(1), -1).transpose(1,2)
        avg_img_features = avg_img_features.view(avg_img_features.size(0), -1)

        return img_features, avg_img_features


## TODO: Write for VGG if required.

import torch
import torch.nn as nn
import torchvision.models as models

class HybridLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_perceptual=0.1):
        super(HybridLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.l1_loss = nn.L1Loss()
        vgg = models.vgg19(pretrained=True).features[:16]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        l1 = self.l1_loss(input, target)
        feat_input = self.vgg(input)
        feat_target = self.vgg(target)
        perceptual = self.l1_loss(feat_input, feat_target)
        return self.lambda_l1 * l1 + self.lambda_perceptual * perceptual

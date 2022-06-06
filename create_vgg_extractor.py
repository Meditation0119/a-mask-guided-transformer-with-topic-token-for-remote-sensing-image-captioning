from torch import nn
from vgg16.models.vgg_vae import VggVAE
import torch


class VggExtractor(nn.Module):

    def __init__(self, vae):
        super(VggExtractor, self).__init__()
        self.fixed_modules = vae.vgg.features[:-4]
        self.finetune_modules = nn.Sequential(
            vae.vgg.features[-4:],
            vae.vgg.avgpool,
            vae.vgg.classifier,
            vae.classifier.classifier[:-2],
            nn.Softmax(dim=-1)
        )
        # self.finetune0 = vae.vgg.features[-4:]
        # self.finetune1 = vae.vgg.avgpool
        # self.finetune2 = vae.vgg.classifier
        # self.finetune3 = vae.classifier.classifier[:-1]
        # self.finetune4 = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fixed_modules(x)
        x = self.finetune_modules[0](x)
        y = self.finetune_modules[1](x)
        y = y.view(-1, 512 * 7 * 7)
        y = self.finetune_modules[2:](y)
        # x = self.finetune0(self.fixed_modules(x))
        # y = self.finetune1(x)
        # y = y.view(-1, 512 * 7 * 7)
        # y = self.finetune2(y)
        # y = self.finetune3(y)
        # y = self.finetune4(y)
        return x, y


def create_extractor(model_path):
    vgg_vae = VggVAE(pretrained=False)
    vgg_vae.load_state_dict(torch.load(model_path))
    vgg_extractor = VggExtractor(vgg_vae)
    return vgg_extractor


if __name__ == '__main__':
    # pretrained_model = VggVAE(pretrained=False)
    # ex = VggExtractor(pretrained_model)
    ex = create_extractor("/root/PycharmProjects/vgg_vae_best_model.pth")
    ex.cuda()
    # ex = nn.DataParallel(ex)
    # x, y = ex(torch.rand(4, 3, 224, 224))
    # print(x.shape, y.shape)
    for module in ex.finetune_modules:
        print(module)
        print('*********************************************************')

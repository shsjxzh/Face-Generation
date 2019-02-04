import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 224 * 224
class AttributeDecoder(nn.Module):
    # pic_size need to be caculated carefully
    def __init__(self, use_gpu=True, size_after_max_pool=512 * 4 * 4, vector_length=4096, PRETRAINED = False):
        super(AttributeDecoder, self).__init__()
        self.use_gpu = use_gpu
        # self.features = models.vgg19_bn(pretrained=True).features
        self.features = models.resnet50(pretrained=PRETRAINED)
        num_ftrs = self.features.fc.in_features
        self.features.fc = nn.Linear(num_ftrs, vector_length)

        self.log_var = nn.Sequential(
            # nn.Linear(size_after_max_pool, vector_length)
            nn.Linear(vector_length, vector_length)
        )
        self.my_mean = nn.Sequential(
            # nn.Linear(size_after_max_pool, vector_length)
            nn.Linear(vector_length, vector_length)
        )
        '''
        self.log_var = nn.Sequential(
            nn.Linear(size_after_max_pool, size_after_max_pool),
            nn.ReLU(),
            nn.Linear(size_after_max_pool, size_after_max_pool),
            nn.ReLU(),
            nn.Linear(size_after_max_pool, size_after_max_pool)
        )
        self.my_mean = nn.Sequential(
            nn.Linear(size_after_max_pool, size_after_max_pool),
            nn.ReLU(),
            nn.Linear(size_after_max_pool, size_after_max_pool),
            nn.ReLU(),
            nn.Linear(size_after_max_pool, size_after_max_pool)
        )
        '''
    def encode(self, x):
        x = self.features(x)
        # this is not needed
        # x = x.view(x.size(0), -1)
        return self.my_mean(x), self.log_var(x)

    def reparameterize(self, my_mean, log_var):
        std = (0.5 * log_var).exp()
        
        eps = torch.normal(torch.zeros_like(std), 1)
        if self.use_gpu:
            device = torch.device("cuda:" + str(my_mean.get_device()))
        else:
            device = torch.device("cpu")

        eps = eps.to(device)

        return eps * std + my_mean

    def forward(self, x):
        my_mean, log_var = self.encode(x)
        z = self.reparameterize(my_mean, log_var)
        return z, my_mean, log_var

'''
def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

def AttributeDecoder_19_b(**kwargs):
    model = AttributeDecoder(make_layers(cfg), **kwargs)
    return model

# print(AttributeDecoder19_b())
'''

'''
# 128 * 128
class AttributeDecoder(nn.Module):
    # pic_size need to be caculated carefully
    def __init__(self, use_gpu=True, size_after_max_pool=512 * 4 * 4, vector_length=4096):
        super(AttributeDecoder, self).__init__()
        self.use_gpu = use_gpu
        # self.features = models.vgg19_bn(pretrained=True).features
        self.features = models.resnet18(pretrained=True)
        num_ftrs = self.features.fc.in_features
        self.features.fc = nn.Linear(num_ftrs, vector_length)

        self.log_var = nn.Sequential(
            # nn.Linear(size_after_max_pool, vector_length)
            nn.Linear(vector_length, vector_length)
        )
        self.my_mean = nn.Sequential(
            # nn.Linear(size_after_max_pool, vector_length)
            nn.Linear(vector_length, vector_length)
        )
'''
'''
        self.log_var = nn.Sequential(
            nn.Linear(size_after_max_pool, size_after_max_pool),
            nn.ReLU(),
            nn.Linear(size_after_max_pool, size_after_max_pool),
            nn.ReLU(),
            nn.Linear(size_after_max_pool, size_after_max_pool)
        )
        self.my_mean = nn.Sequential(
            nn.Linear(size_after_max_pool, size_after_max_pool),
            nn.ReLU(),
            nn.Linear(size_after_max_pool, size_after_max_pool),
            nn.ReLU(),
            nn.Linear(size_after_max_pool, size_after_max_pool)
        )
'''
'''
    def encode(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.my_mean(x), self.log_var(x)

    def reparameterize(self, my_mean, log_var):
        std = (0.5 * log_var).exp()
        
        eps = torch.normal(torch.zeros_like(std), 1)
        if self.use_gpu:
            device = torch.device("cuda:" + str(my_mean.get_device()))
        else:
            device = torch.device("cpu")

        eps = eps.to(device)

        return eps * std + my_mean

    def forward(self, x):
        my_mean, log_var = self.encode(x)
        z = self.reparameterize(my_mean, log_var)
        return z, my_mean, log_var

'''
'''
def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

def AttributeDecoder_19_b(**kwargs):
    model = AttributeDecoder(make_layers(cfg), **kwargs)
    return model

# print(AttributeDecoder19_b())
'''
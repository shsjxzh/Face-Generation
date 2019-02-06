from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# ngpu = int(opt.ngpu)
class Discriminator(nn.Module):
    def __init__(self, nc = 3, ndf = 64, input_size = 32768, hidden_node = 4096):
        super(Discriminator, self).__init__()
        # self.ngpu = ngpu
        # the input size must be 224 * 224!!
        self.feature = nn.Sequential(
            # input is (nc) x 224 * 224
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 112 x 112
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 56 x 56
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 28 x 28
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 14 x 14
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf*8) x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 7, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        '''
        self.feature = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
            # nn.Sigmoid()
        )
        '''
        '''
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_node),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_node, hidden_node // 2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_node // 2, 1),
            nn.Sigmoid()
        )
        '''

        # the last fully connected layer must be extracted!!

    def forward(self, input):
        '''
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        '''
        feature = self.feature(input)
        result = self.classifier(feature).squeeze()
        feature = feature.view(feature.size(0), -1)
        # print(feature.size())
        return result, feature





'''
# 128 * 128
class Discriminator(nn.Module):
    def __init__(self, nc = 3, ndf = 64, input_size = 32768, hidden_node = 4096):
        super(Discriminator, self).__init__()
        # self.ngpu = ngpu
        # the output size must be 128 * 128!!
        self.feature = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf*8) x 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        '''
'''
        self.feature = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
            # nn.Sigmoid()
        )
'''
'''
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_node),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_node, hidden_node // 2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_node // 2, 1),
            nn.Sigmoid()
        )
'''

        # the last fully connected layer must be extracted!!
'''
    def forward(self, input):
'''
'''
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
'''
'''
        feature = self.feature(input)
        result = self.classifier(feature).squeeze()
        feature = feature.view(feature.size(0), -1)
        # print(feature.size())
        return result, feature
'''
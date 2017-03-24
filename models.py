import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self, input_nc, target_nc, ngf):
        super(_netG, self).__init__()
        self.encoder_1 = nn.Sequential(
            # input is (input_nc) x 256 x 256
            nn.Conv2d(input_nc, ngf, 4, 2, 1, bias=False),
            # state size. (ndf) x 128 x 128
        )

        self.encoder_2 = nn.Sequential(
            # input is (ngf) x 128 x 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            # state size. (ngf * 2) x 64 x 64
        )

        self.encoder_3 = nn.Sequential(
            # input is (ngf * 2) x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            # state size. (ngf * 4) x 32 x 32
        )

        self.encoder_4 = nn.Sequential(
            # input is (ngf * 4) x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 8) x 16 x 16
        )

        self.encoder_5 = nn.Sequential(
            # input is (ngf * 8) x 16 x 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 8) x 8 x 8
        )

        self.encoder_6 = nn.Sequential(
            # input is (ngf * 8) x 8 x 8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 8) x 4 x 4
        )

        self.encoder_7 = nn.Sequential(
            # input is (ngf * 8) x 4 x 4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 8) x 2 x 2
        )

        self.encoder_8 = nn.Sequential(
            # input is (ngf * 8) x 2 x 2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 8) x 1 x 1
        )

        self.decoder_8 = nn.Sequential(
            # input is (ngf * 8) x 1 x 1
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 8) x 2 x 2
        )

        self.decoder_7 = nn.Sequential(
            # input is (ngf * 8 * 2) x 2 x 2
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 8) x 4 x 4
        )

        self.decoder_6 = nn.Sequential(
            # input is (ngf * 8 * 2) x 4 x 4
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 8) x 8 x 8
        )

        self.decoder_5 = nn.Sequential(
            # input is (ngf * 8 * 2) x 8 x 8
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf * 4) x 16 x 16
        )

        self.decoder_4 = nn.Sequential(
            # input is (ngf * 8 * 2) x 16 x 16
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            # state size. (ngf * 4) x 32 x 32
        )

        self.decoder_3 = nn.Sequential(
            # input is (ngf * 4 * 2) x 32 x 32
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            # state size. (ngf * 2) x 64 x 64
        )

        self.decoder_2 = nn.Sequential(
            # input is (ngf * 2 * 2) x 64 x 64
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            # state size. (ngf) x 128 x 128
        )

        self.decoder_1 = nn.Sequential(
            # input is (ngf * 2) x 128 x 128
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, target_nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (target_nc) x 256 x 256
        )

    def forward(self, input):
        output_e1 = self.encoder_1(input)
        output_e2 = self.encoder_2(output_e1)
        output_e3 = self.encoder_3(output_e2)
        output_e4 = self.encoder_4(output_e3)
        output_e5 = self.encoder_5(output_e4)
        output_e6 = self.encoder_6(output_e5)
        output_e7 = self.encoder_7(output_e6)
        output_e8 = self.encoder_8(output_e7)

        output_d8 = self.decoder_8(output_e8)
        output_d7 = self.decoder_7(torch.cat((output_d8, output_e7), 1))
        output_d6 = self.decoder_6(torch.cat((output_d7, output_e6), 1))
        output_d5 = self.decoder_5(torch.cat((output_d6, output_e5), 1))
        output_d4 = self.decoder_4(torch.cat((output_d5, output_e4), 1))
        output_d3 = self.decoder_3(torch.cat((output_d4, output_e3), 1))
        output_d2 = self.decoder_2(torch.cat((output_d3, output_e2), 1))
        output_d1 = self.decoder_1(torch.cat((output_d2, output_e1), 1))

        return output_d1

class _netD(nn.Module):
    def __init__(self, input_nc, target_nc, ndf):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
            # input is (nc * 2) x 64 x 64
            nn.Conv2d(input_nc + target_nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, target):
        output = self.main(torch.cat((input, target), 1))
        return output

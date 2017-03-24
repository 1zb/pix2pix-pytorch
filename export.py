from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import glob

from scipy.misc import imread, imsave
import numpy as np

import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--model', type=str, help='model file')
parser.add_argument('--image_dir', type=str, required=True, help='directory of images')
parser.add_argument('--format', type=str, default='jpg', choices=['jpg', 'png'], help='image format')
parser.add_argument('--direction', type=str, default='rl', choices=['lr', 'rl'], help='which direction')

args = parser.parse_args()

from models import _netG

model_filenames = glob.glob(os.path.join('checkpoint/netG_epoch_{}_*.pth'.format(args.direction)))

netG = []
if args.model == '':
    for model_filename in model_filenames:
        netG.append(torch.load(model_filename))
else:
    netG.append(torch.load(args.model))

if args.cuda:
    netG = [model.cuda() for model in netG]

image_filenames = glob.glob(os.path.join(args.image_dir, '*.' + args.format))

for image_filename in image_filenames:
    img = np.transpose(imread(image_filename), (2,0,1)).astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)
    width = img.shape[3]
    if args.direction == 'lr':
        input = torch.from_numpy(img[:, :, :, :width//2])
    else:
        input = torch.from_numpy(img[:, :, :, width//2:])

    if args.cuda:
        input = input.cuda()

    input = Variable(input)
    # prediction = netG(input)
    predictions = [model(input).data.cpu().numpy() for model in netG]

    if not os.path.exists("result"):
        os.mkdir("result")

    predictions = [np.transpose(np.squeeze(p) * 255.0, (1,2,0)).clip(0, 255).astype(np.uint8) for p in predictions]
    out_img = np.hstack([imread(image_filename), *predictions])
    imsave('result/{}'.format(os.path.split(image_filename)[-1]), out_img)

# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.backends.cudnn as cudnn

from .DSFD_vgg import build_net_vgg
from .DSFD_resnet import build_net_resnet


def build_net(phase, num_classes=2, model='vgg'):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return

    if model != 'vgg' and 'resnet' not in model:
        print("ERROR: model:" + model + " not recognized")
        return

    if model == 'vgg':
        return build_net_vgg(phase, num_classes)
    else:
        return build_net_resnet(phase, num_classes, model)



def basenet_factory(model='vgg'):
	if model=='vgg':
		basenet = 'vgg16_reducedfc.pth'

	elif 'resnet' in model:
		basenet = '{}.pth'.format(model)
	return basenet


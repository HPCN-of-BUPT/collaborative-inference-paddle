#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
"""
Contains common utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import distutils.util
import numpy as np
import six
from collections import deque
from paddle.fluid import core
import argparse
import functools
from config import *


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self):
        self.loss_sum = 0.0
        self.iter_cnt = 0

    def add_value(self, value):
        self.loss_sum += np.mean(value)
        self.iter_cnt += 1

    def get_mean_value(self):
        return self.loss_sum / self.iter_cnt


def parse_args():
    """return all args
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable
    # ENV
    add_arg('use_gpu',          bool,   True,      "Whether use GPU.")
    add_arg('model_save_dir',   str,    'complete_pruned_model_new',     "The path to save model.")
    add_arg('pretrain',         str,    'complete_model/model_final', "The pretrain model path.")#weights/darknet53
    add_arg('weights',          str,    'pruned_model/model_final1', "The weights path.")#weights/yolov3
    add_arg('freezed_model',          str,    "freezed_model", "export freezed model")
    add_arg('dataset',          str,    'coco2014',  "Dataset: coco2014, coco2017.")
    add_arg('class_num',        int,    80,          "Class number.")
    add_arg('data_dir',         str,    'dataset/coco',        "The data root path.")
    add_arg('start_iter',       int,    0,      "Start iteration.")
    add_arg('use_multiprocess', bool,   True,   "add multiprocess.")
    #SOLVER
    add_arg('batch_size',       int,    8,      "Mini-batch size per device.")
    add_arg('learning_rate',    float,  0.001,  "Learning rate.")
    add_arg('max_iter',         int,    50000, "Iter number.")
    add_arg('snapshot_iter',    int,    20000,   "Save model every snapshot stride.")
    add_arg('label_smooth',     bool,   True,   "Use label smooth in class label.")
    add_arg('no_mixup_iter',    int,    40000,  "Disable mixup in last N iter.")
    # TRAIN TEST INFER
    add_arg('input_size',       int,    608,    "Image input size of YOLOv3.")
    add_arg('random_shape',     bool,   True,   "Resize to random shape for train reader.")
    add_arg('valid_thresh',     float,  0.005,  "Valid confidence score for NMS.")
    add_arg('nms_thresh',       float,  0.45,   "NMS threshold.")
    add_arg('nms_topk',         int,    400,    "The number of boxes to perform NMS.")
    add_arg('nms_posk',         int,    100,    "The number of boxes of NMS output.")
    add_arg('debug',            bool,   False,  "Debug mode")
    add_arg('print_par',            bool,   True,  "print Parameters")
    add_arg('prune_par',            str,   "stage.1.0.0.conv.weights,stage.1.1.0.conv.weights,stage.1.1.1.conv.weights,yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights,yolo_block.0.1.1.conv.weights,yolo_block.0.2.conv.weights,yolo_block.0.tip.conv.weights,yolo_block.1.0.0.conv.weights,yolo_block.1.0.1.conv.weights,yolo_block.1.1.0.conv.weights,yolo_block.1.1.1.conv.weights,yolo_block.1.2.conv.weights,yolo_block.1.tip.conv.weights,yolo_block.2.0.0.conv.weights,yolo_block.2.0.1.conv.weights,yolo_block.2.1.0.conv.weights,yolo_block.2.1.1.conv.weights,yolo_block.2.2.conv.weights,yolo_block.2.tip.conv.weights",  "Prune Parameters")
    add_arg('prune_ratio',          str,   "0.2,0.3,0.4,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,",  "Prune Ratios")
    #0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,   #,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.8,0.8,0.8,0.8,0.8
    # SINGLE EVAL AND DRAW
    add_arg('image_path',       str,   'dataset/coco/val2014', 
            "The image path used to inference and visualize.")
    add_arg('image_name',       str,    'COCO_val2014_000000566975.jpg',   
            "The single image used to inference and visualize. None to inference all images in image_path")
    add_arg('draw_thresh',      float,  0.5,    
            "Confidence score threshold to draw prediction box in image in debug mode")
    # yapf: enable
    args = parser.parse_args()
    file_name = sys.argv[0]
    merge_cfg_from_args(args)
    return args

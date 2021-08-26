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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import json
import numpy as np
import paddle
import paddle.fluid as fluid
import reader
from models.yolov3_exp import YOLOv3
from utility import print_arguments, parse_args
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config import cfg
from paddleslim.prune import Pruner
from paddleslim.analysis import flops
paddle.enable_static()

def get_pruned_params(train_program):
    params = []
    #skip_vars = ['yolo_input']  # skip the first conv2d layer
    for block in train_program.blocks:
        for param in block.all_parameters():
            if ('conv' in param.name)  and ('yolo_input' not in param.name) and ('downsample' not in param.name) : #and ('stage.0' not in param.name)and ('stage.1' not in param.name)and ('stage.2' not in param.name)
                if  ('yolo_block' in param.name) or ('stage.4' in param.name):#or ('stage.3' in param.name) 
                    params.append(param.name)#or ('batch_norm' in param.name)
    return params

def eval():
    image = fluid.layers.data(name='image', shape= [-1, 3, cfg.input_size, cfg.input_size], dtype='float32')
    image_shape = fluid.layers.data(name="image_shape", shape=[2], dtype='int32')

    
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    model = YOLOv3(is_train=False)
    model.build_input()
    model.build_model(image, image_shape)
    outputs = model.get_pred()
    startup_prog = fluid.default_startup_program()
    train_program = fluid.default_main_program()
    exe.run(startup_prog, scope=fluid.global_scope())


    # yapf: disable
    if cfg.weights:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrain, var.name))
        fluid.io.load_vars(exe, cfg.pretrain, predicate=if_exist,main_program = train_program)

    

    #########prune
    
    pruned_params = get_pruned_params(train_program)
    pruned_ratios = []
    for param in pruned_params:
        if 'yolo_block.0.' in param:
            pruned_ratios.append(0.5)
        elif 'yolo_block.1.' in param:
            pruned_ratios.append(0.5)
        elif 'yolo_block.2.' in param:
            pruned_ratios.append(0.5)
        else:
            pruned_ratios.append(0.2)

    #pruned_params = cfg.prune_par.strip().split(",") #此处也可以通过写正则表达式匹配参数名
    print("pruned params: {}".format(pruned_params))
    #pruned_ratios = [float(n) for n in cfg.prune_ratio]
    print("pruned ratios: {}".format(pruned_ratios))

    pruner = Pruner()
    train_program = pruner.prune(
        train_program,
        fluid.global_scope(),
        params=pruned_params,
        ratios=pruned_ratios,
        place=place,
        only_graph=False)[0]

    exe.run(startup_prog)
    fluid.io.load_persistables(exe, cfg.weights, train_program)
    fluid.io.save_inference_model(cfg.freezed_model, ['image','image_shape'], outputs, exe, train_program,model_filename='completed_pruned_model', params_filename='completed_pruned_params')
    
 

    



if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    eval()

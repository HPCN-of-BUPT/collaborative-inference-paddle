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
from models.yolov3_client import YOLOv3_client
from models.yolov3_server import YOLOv3_server
from utility import print_arguments, parse_args
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config import cfg
from paddleslim.prune import Pruner
from paddleslim.analysis import flops
paddle.enable_static()


def eval():
    image_fake = fluid.layers.data(name='image_fake', shape= [-1, 3, cfg.input_size, cfg.input_size], dtype='float32')
    image1 = fluid.layers.data(name='image1', shape= [-1, 256, 76, 76], dtype='float32')#512,410
    image2 = fluid.layers.data(name='image2', shape= [-1, 512, 38, 38], dtype='float32')#256,205
    image_shape = fluid.layers.data(name="image_shape", shape=[2], dtype='float32')

    model = YOLOv3_server(is_train=False)
    model.build_input()
    model.build_model(image1,image2,image_shape)
    outputs = model.get_pred()
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    startup_prog = fluid.default_startup_program()
    train_program = fluid.default_main_program()
    exe.run(startup_prog, scope=fluid.global_scope())


    # yapf: disable
    if cfg.weights:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrain, var.name))
        fluid.io.load_vars(exe, cfg.pretrain, predicate=if_exist,main_program = train_program)
    
    train_program = train_program.clone(for_test=True)
    exe.run(startup_prog)
    fluid.io.load_persistables(exe, cfg.pretrain, train_program)
    fluid.io.save_inference_model(cfg.freezed_model, ['image1','image2','image_shape'], outputs, exe, train_program,model_filename='split_server_model', params_filename='split_server_params')


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    eval()

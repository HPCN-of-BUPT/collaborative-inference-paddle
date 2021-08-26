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
from utility import print_arguments, parse_args
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config import cfg
import cv2
from PIL import Image, ImageFont
from PIL import ImageDraw
paddle.enable_static()


def eval():
    if '2014' in cfg.dataset:
        test_list = 'annotations/instances_val2014.json'
    elif '2017' in cfg.dataset:
        test_list = 'annotations/instances_val2017.json'

    if cfg.debug:
        if not os.path.exists('output'):
            os.mkdir('output')
    place = fluid.CUDAPlace(5) if cfg.use_gpu else fluid.CPUPlace()
    exe1 = fluid.Executor(place)
    exe2 = fluid.Executor(place)   
    path = 'freezed_model/test_model'  # 'model/freeze_model'
    [inference_program1, feed_target_names1, fetch_targets1] = fluid.io.load_inference_model(dirname=path, executor=exe1,
                                                                                      model_filename='quant_split_pruned_client_model',
                                                                                      params_filename='quant_split_pruned_client_params')
    [inference_program2, feed_target_names2, fetch_targets2] = fluid.io.load_inference_model(dirname=path, executor=exe2,
                                                                                      model_filename='quant_split_pruned_server_model',
                                                                                      params_filename='quant_split_pruned_server_params')
    
    
    input_size = cfg.input_size
    test_reader = reader.test(input_size, 1)
    label_names, label_ids = reader.get_label_infos()
    if cfg.debug:
        print("Load in labels {} with ids {}".format(label_names, label_ids))

    def get_pred_result(boxes, scores, labels, im_id):
        result = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            bbox = [x1, y1, w, h]
            
            res = {
                    'image_id': im_id,
                    'category_id': label_ids[int(label)],
                    'bbox': list(map(float, bbox)),
                    'score': float(score)
            }
            result.append(res)
        return result

    dts_res = []
    total_time = 0
    for batch_id, batch_data in enumerate(test_reader()):
        #print(batch_data[0][0])
        #if batch_id == 100:
        #    break
        start_time = time.time()
        img = np.array(batch_data[0][0]).astype('float32')
        img = img[np.newaxis, :]
        img_shape = np.array(batch_data[0][2]).astype('int32')
        img_shape = img_shape[np.newaxis, :]
        batch_outputs_temp = exe1.run(inference_program1,
            fetch_list=fetch_targets1,
            feed={feed_target_names1[0]: img},
            return_numpy=False,
            use_program_cache=True)
        batch_outputs = exe2.run(inference_program2,
            fetch_list=fetch_targets2,
            feed={feed_target_names2[0]: batch_outputs_temp[0],
                                  feed_target_names2[1]: batch_outputs_temp[1],
                                  #feed_target_names2[2]: batch_outputs_temp[2],
                                  feed_target_names2[2]: img_shape},
            return_numpy=False)
        lod = batch_outputs[0].lod()[0]
        nmsed_boxes = np.array(batch_outputs[0])
        if nmsed_boxes.shape[1] != 6:
            continue
        for i in range(len(lod) - 1):
            im_id = batch_data[i][1]
            start = lod[i]
            end = lod[i + 1]
            if start == end:
                continue
            nmsed_box = nmsed_boxes[start:end, :]
            labels = nmsed_box[:, 0]
            scores = nmsed_box[:, 1]
            boxes = nmsed_box[:, 2:6]
            dts_res += get_pred_result(boxes, scores, labels, im_id)

        end_time = time.time()
        print("batch id: {}, time: {}".format(batch_id, end_time - start_time))
        total_time += end_time - start_time

    with open("yolov3_result.json", 'w') as outfile:
        json.dump(dts_res, outfile)
    print("start evaluate detection result with coco api")
    coco = COCO(os.path.join(cfg.data_dir, test_list))
    cocoDt = coco.loadRes("yolov3_result.json")
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print("evaluate done.")

    print("Time per batch: {}".format(total_time / batch_id))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    eval()

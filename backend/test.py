import requests, json, time ,os
import numpy as np
import paddle
import core
from load_model import edge_load_model_yolo2
paddle.enable_static()

r = requests.get('http://127.0.0.1:5000/transmit_image')
images = r.json()['file_list']
# file_list = images["file_list"]

exe = paddle.static.Executor(paddle.CPUPlace())
[inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(core.EDGE_MODEL_DIR, exe))
for index, image in enumerate(images):
    start_time = time.time()

    results = exe.run(inference_program,
              feed={feed_target_names[0]: np.array(json.loads(image['tensor']), dtype=np.float32)},
              fetch_list=fetch_targets)
    shape = np.array(json.loads(image['shape']), dtype=np.int32)
    end_time = time.time()
    

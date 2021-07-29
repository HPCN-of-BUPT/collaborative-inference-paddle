from sys import prefix
import paddle
import glob
import numpy as np
import time
def edge_load_model(path_prefix):
    paddle.enable_static()
    startup_prog = paddle.static.default_startup_program()
    start_time = time.time()

    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(startup_prog)

    # 保存预测模型

    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(path_prefix, exe))
    tensor_img = np.array(np.random.random((1, 3, 32, 32)), dtype=np.float32)
    results = exe.run(inference_program,
              feed={feed_target_names[0]: tensor_img},
              fetch_list=fetch_targets)
    end_time = time.time()
    return np.array(results[0]), round(end_time - start_time, 3)

def cloud_load_tensor(path_prefix, tensor):
    paddle.enable_static()
    startup_prog = paddle.static.default_startup_program()
    start_time = time.time()

    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(startup_prog)

    # 保存预测模型

    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(path_prefix, exe))

    results = exe.run(inference_program,
              feed={feed_target_names[0]: tensor},
              fetch_list=fetch_targets)
    result = results[0].tolist()
    end_time = time.time()
    return [result[i].index(max(result[i])) for i in range(len(result))], round(end_time - start_time, 3)

if __name__ == "__main__":
    tensor,edge_infer_time = edge_load_model(path_prefix="../data/send/model/client_infer_resnet18_cifar10")
    result,cloud_infer_time = cloud_load_tensor(path_prefix="../data/send/model/server_infer_resnet18_cifar10",tensor=tensor)
    print(result)







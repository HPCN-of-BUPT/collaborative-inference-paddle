import performance_test.paddle_eval as paddle_eval
import numpy as np

def client_analyse(filename):
    c_flops, c_model_size, c_output_size = paddle_eval.model_analyse(filename,
                                                                     input_list=[
                                                                         paddle_eval.make_fake_input((1, 3, 608, 608))]
                                                                     )
    infos = {'flops':str(c_flops),
             'params':str(c_model_size),
             'output_size':str(c_output_size)}
    return infos


def server_analyse(filename):
    s_flops, s_model_size, s_output_size = paddle_eval.model_analyse(filename,
                                                                     input_list=[
                                                                         paddle_eval.make_fake_input((1, 256, 76, 76),
                                                                                                     dtype='float32'),
                                                                         paddle_eval.make_fake_input((1, 512, 38, 38),
                                                                                                     dtype='float32'),
                                                                         np.asarray([576, 768]).reshape(1, 2)
                                                                         ],
                                                                     dtype='float'
                                                                     )

    infos = {'flops': str(s_flops),
             'params': str(s_model_size),
             'output_size': str(s_model_size)}
    return infos

def client_():
    print("Client Model Analyse:")
    c_infer_model_filepath = './data/edge/client_infer_yolov3'  # 确定模型文件路径。注意没有后缀。

    c_program = paddle_eval.construct_simple_program(c_infer_model_filepath)  # 使用flops和model_size函数前，先传入文件构造Program

    c_fps = paddle_eval.calc_flops(c_program)  # 计算flops的函数，返回模型的flops。
    c_msize = paddle_eval.calc_model_size(
        c_program)  # 计算model_size的函数，返回模型的model_size（单位byte）和变量个数var_count（单位个）。请根据需要取值。
    c_osize = paddle_eval.calc_output_tensor_size(c_infer_model_filepath,
                                                  # 计算模型输出向量大小的函数。返回模型输出的向量size（单位byte）和该向量的shape
                                                  input_shape=(3, 32, 32),
                                                  batch_size=1)

    c_flops, c_var_count, c_output_size = paddle_eval.model_analyse(c_infer_model_filepath, input_shape=(3, 32, 32),
                                                                     batch_size=1)
    return c_flops, c_var_count, c_output_size


def server_():
    print("Server Model Analyse:")
    s_infer_model_filepath = './data/cloud/server_infer_yolov3'
    s_program = paddle_eval.construct_simple_program(s_infer_model_filepath)  # 使用flops和model_size函数前，先传入文件构造Program

    s_fps = paddle_eval.calc_flops(s_program)  # 计算flops的函数，返回模型的flops。
    s_msize = paddle_eval.calc_model_size(
        s_program)  # 计算model_size的函数，返回模型的model_size（单位byte）和变量个数var_count（单位个）。请根据需要取值。

    return s_fps,s_msize,s_msize

if __name__ == '__main__':
    client_analyse('./data/send/client_infer_pruned')
    server_analyse('./data/send/server_infer_pruned')
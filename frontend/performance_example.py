import backend.performance_test as paddle_eval
# 此example.py为示例。本文件夹以外的地方，可以import performance_test来代替直接引入paddle_eval。
# 所有函数都有详细注释。

import numpy as np

if __name__ == '__main__':
    print("Client Model Analyse:")
    c_infer_model_filepath = '../backend/model_transmit/data/send/client_infer_yolov3'
    c_flops, c_model_size, c_output_size = paddle_eval.model_analyse(c_infer_model_filepath,
                                                         input_list=[paddle_eval.make_fake_input((1, 3, 608, 608))]
                                                         )
    print("Server Model Analyse:")
    s_infer_model_filepath = '../backend/model_transmit/data/send/server_infer_yolov3'
    s_flops, s_model_size, s_output_size = paddle_eval.model_analyse(s_infer_model_filepath,
                                                         input_list=[paddle_eval.make_fake_input((1, 256, 76, 76), dtype='float32'),
                                                                     paddle_eval.make_fake_input((1, 512, 38, 38), dtype='float32'),
                                                                     np.asarray([576, 768]).reshape(1,2)
                                                                     ],
                                                         dtype='float'
                                                         )
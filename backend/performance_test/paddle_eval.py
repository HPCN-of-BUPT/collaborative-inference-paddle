import paddle
import numpy as np
import sys
from paddleslim.core import GraphWrapper
from paddleslim.analysis import flops

def make_fake_input(input_shape, dtype = 'float32'):
    """已知输入的shape时可用，构造一个伪输入。

    :params input_shape: 输入的向量形状shape
    :params dtype: 输入的向量元素数据类型dtype
    :return: fake_input，构造的假样本
    """
    fake_input = np.zeros(input_shape).astype(dtype)
    return fake_input


def calc_flops(infer_program):
    """封装paddleslim.analysis.flops

    :param infer_program: 构造好的program。可调用construct_simple_program构造此变量。
    :return: flops; 该模型的浮点计算量。
    """
    return flops(infer_program)


# 函数改写自paddleslim.analysis.flops
def calc_model_size(model, inputs=None, dtypes=None, only_conv=True, detail=False):
    """封装的模型大小计算函数。暂不支持除model以外的其他参数。

    :param model: 传入的model模型，必须是Program类型。可调用construct_simple_program构造此变量
    :return: (model_size, var_count); 返回model_size模型大小（仅参数量大小，不含其他信息），以及var_count模型参数个数。
    """
    if isinstance(model, paddle.static.Program):
        return _static_model_size(model, only_conv=only_conv, detail=detail)
    else:
        raise TypeError(
            'This function is for the static model. If you want to evaluate a dygraph, please use functions in paddleslim.analysis.')


def construct_simple_program(model_and_param_path):
    """从已有的文件构造一个简单的Program

    :param model_and_param_path: 输入“fpath/fname”即可，无需后缀；需要同一个模型的fname.pdmodel, fname.pdiparam两个文件，并保存在同一路径fpath下。例：test/model，加载test/model.pdmodel和test/model.pdiparams两个文件。
    :return: Program; 返回该模型的Program，类型为paddle.static.Program。
    """
    paddle.enable_static()
    paddle_executor = paddle.static.Executor(paddle.CPUPlace())
    loaded_infer_model = paddle.static.load_inference_model(model_and_param_path, paddle_executor)
    return loaded_infer_model[0]


def calc_output_tensor_size():
    """此函数被移动到了_paddle_eval_deprecated.py中，即将被废弃。
    """
    pass

def calc_complex_output(infer_model_path, feed_list, batch_size=1, dtype='float', only_weight = False):
    """该函数通过直接输入样本计算，返回模型的输出向量大小。函数经过扩展，适用于多个输入输出的神经网络，但需要自行设置每个输入。

    :param infer_model_path: 输入“fpath/fname”即可，无需后缀；需要同一个模型的fname.pdmodel, fname.pdiparam两个文件，并保存在同一路径fpath下。例：test/model，加载test/model.pdmodel和test/model.pdiparams两个文件。
    :param feed_list: 静态模型的输入列表，将按顺序传入。使用未处理的原始数据传入时，请自行进行预处理。也可使用make_fake_input制造假样本输入（见样例）。
    :param batch_size: 模型输入批次大小。默认为1（单个输入）
    :param dtype: 输入tensor每个元素的类型，默认float32。
    :param only_weight: 是否仅计算参数所占空间。默认否，返回包括tensor其他信息所占空间的tensor大小。
    :return: (output_tensor_size, output_tensor_shape, outputs); 返回模型实际输出向量的size（含参数以外其他信息，单位byte）、shape，以及实际输出向量。
    """
    paddle.enable_static()
    startup_prog = paddle.static.default_startup_program()
    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(startup_prog)
    inference_program, feed_target_names, fetch_targets = (
        paddle.static.load_inference_model(infer_model_path, exe))

    # 依次将输入填入静态模型的feed中。
    feed = {}
    for index, t in enumerate(feed_list):
        feed[feed_target_names[index]] = t

    # 运行模型
    outputs = exe.run(inference_program,
                      feed=feed,
                      fetch_list=fetch_targets,
                      return_numpy=False)
    if not isinstance(outputs, list):
        outputs = [outputs]

    # 计算输出大小
    output_shape = []
    output_size = 0
    if only_weight:
        for output_data in outputs:
            if not hasattr(output_data, 'dtype'):# 这里判断类型最好用isinstance(output_data, Lod_Tensor)，但是我没找到这个类的定义应该从哪里import
                output_data = np.array(output_data)
            output_size += output_data.dtype.itemsize * output_data.size # 仅含参数的size大小，单位byte。
            output_shape.append(output_data.shape)
    else:
        for output_data in outputs:
            if not hasattr(output_data, 'dtype'):
                output_data = np.array(output_data)
            output_size += sys.getsizeof(output_data)
            output_shape.append(output_data.shape)

    return output_size, output_shape, outputs


def _static_model_size(program, only_conv=True, detail=False):
    """将Program封装为静态图。
    """
    graph = GraphWrapper(program)
    return _graph_model_size(graph, only_conv=only_conv, detail=detail)


def _graph_model_size(graph, only_conv=True, detail=False):
    """通过统计模型文件中所有含有shape的Variable确定是否为参数变量。

    :param graph: 输入模型。该模型必须为Paddle的GraphWrapper类。
    :param only_conv: 仅统计卷积层（不支持）
    :param detail: 输出每层统计量（不支持）
    :return: 返回model_size模型大小（仅参数量大小，不含其他信息），以及var_count模型参数个数。
    """
    assert isinstance(graph, GraphWrapper)
    model_size = 0  # 模型参数大小，单位：byte
    var_count = 0  # 模型参数个数，单位：个
    for variable_name in graph.persistables:
        if variable_name not in ['feed', 'fetch']:
            variable = graph.persistables[variable_name]  # 类成员有shape的统计一下，没有shape的去掉
            if hasattr(variable, 'shape'):
                var_params = np.product(getattr(variable, 'shape')) # 检查以后发现persistable后的shape里不会出现-1项。
                var_count += var_params
                model_size += var_params * sys.getsizeof(variable.dtype)
    return model_size, var_count


def model_analyse(infer_model_filepath, input_list, batch_size=1, only_weight = False, dtype='float',detail=True):
    """一键分析模型指标，对几个函数进行简单的封装。
    仅提供静态图模型文件分析，暂不支持动态模型dygraph。如果已经构造好了model的Program，可以使用calc_model_size和calc_flops函数.

    :param infer_model_filepath: 输入“fpath/fname”即可，无需后缀；需要同一个模型的fname.pdmodel, fname.pdiparam两个文件，并保存在同一路径fpath下。例：test/model，加载test/model.pdmodel和test/model.pdiparams两个文件。
    :param input_list: 模型输入。多个输入的模型将输入依次放入list，单个输入的模型请确保传入的为list变量。
    :param batch_size: 模型输入批次大小。默认为1（单个输入）
    :param detail: 运行时打印
    :return: (flops, model_size, output_size); 返回模型文件的浮点运算量flops、模型大小model_size、模型输出向量大小output_size
    """
    infer_program = construct_simple_program(infer_model_filepath)

    flops = calc_flops(infer_program)

    model_size, var_count = calc_model_size(infer_program)

    output_size, output_shape, _ = calc_complex_output(infer_model_path=infer_model_filepath,
                                                       feed_list=input_list,
                                                       batch_size=batch_size,
                                                       dtype=dtype,
                                                       only_weight=only_weight)

    if detail:
        print('FLOPs:{0}'.format(flops))
        print('Model Size:{0} Params, {1} Bytes'.format(var_count, model_size))
        print("Output data size is {0} bytes, shape is {1}".format(output_size, output_shape))

    return flops, var_count, output_size


if __name__ == '__main__':
    print("Client Model Analyse:")
    c_infer_model_filepath = '../model_transmit/data/send/client_infer_yolov3'
    c_flops, c_model_size, c_output_size = model_analyse(c_infer_model_filepath,
                                                         input_list=[make_fake_input((1, 3, 608, 608))]
                                                         )

    # 第一个输入：LoDTensor, dtype=float, shape=[1,256,76,76]
    # 第二个输入：LoDTensor，dtype=float，shape=[1,512,38,38]
    # 第三个输入：[[576 768]]，ndarray, dtype=int, shape=(1,2)
    print("Server Model Analyse:")
    s_infer_model_filepath = '../model_transmit/data/send/server_infer_yolov3'
    s_flops, s_model_size, s_output_size = model_analyse(s_infer_model_filepath,
                                                         input_list=[make_fake_input((1, 256, 76, 76), dtype='float32'),
                                                                     make_fake_input((1, 512, 38, 38), dtype='float32'),
                                                                     np.asarray([576, 768]).reshape(1,2)
                                                                     ],
                                                         dtype='float'
                                                         )
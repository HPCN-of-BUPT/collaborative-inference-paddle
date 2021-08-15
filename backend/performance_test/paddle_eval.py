import paddle
import numpy as np
import sys
from paddleslim.core import GraphWrapper
from paddleslim.analysis import flops

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

    :param infer_model_filepath: 输入“fpath/fname”即可，无需后缀；需要同一个模型的fname.pdmodel, fname.pdiparam两个文件，并保存在同一路径fpath下。例：test/model，加载test/model.pdmodel和test/model.pdiparams两个文件。
    :return: Program; 返回该模型的Program，类型为paddle.static.Program。
    """
    paddle.enable_static()
    paddle_executor = paddle.static.Executor(paddle.CPUPlace())
    loaded_infer_model = paddle.static.load_inference_model(model_and_param_path, paddle_executor)
    return loaded_infer_model[0]

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
                var_params = np.product(getattr(variable, 'shape'))
                var_count += var_params
                model_size += var_params * sys.getsizeof(variable.dtype)
    return model_size, var_count


def model_analyse(infer_model_filepath, input_shape, batch_size=1, only_weight = False, detail=True):
    """此函数仅提供静态图模型文件分析，暂不支持动态模型dygraph。如果已经构造好了model的Program，可以使用calc_model_size和calc_flops函数.

    :param infer_model_filepath: 输入“fpath/fname”即可，无需后缀；需要同一个模型的fname.pdmodel, fname.pdiparam两个文件，并保存在同一路径fpath下。例：test/model，加载test/model.pdmodel和test/model.pdiparams两个文件。
    :param input_shape: 模型输入向量维度。仅考虑单个输入。
    :param batch_size: 模型输入批次大小。默认为1（单个输入）
    :param detail: 运行时打印
    :return: (flops, model_size, output_size); 返回模型文件的浮点运算量flops、模型大小model_size、模型输出向量大小output_size
    """
    infer_program = construct_simple_program(infer_model_filepath)

    flops = calc_flops(infer_program)

    model_size, var_count = calc_model_size(infer_program)

    output_size, output_shape = calc_output_tensor_size(infer_model_filepath, input_shape=input_shape,
                                                        batch_size=batch_size, only_weight=only_weight)

    if detail:
        print('FLOPs:{0}'.format(flops))
        print('Model Size:{0} Params, {1} Bytes'.format(var_count, model_size))
        print("Output data size is {0} bytes, shape is {1}".format(output_size, output_shape))

    return flops, model_size, output_size


def calc_output_tensor_size(infer_model_path, input_shape, batch_size=1, dtype='float32', only_weight = False):
    """该函数返回模型的输出向量大小，是通过对模型构建预测器，直接计算得到的。
    涉及paddle inference api请参考https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_doc/

    :param infer_model_path: 输入“fpath/fname”即可，无需后缀；需要同一个模型的fname.pdmodel, fname.pdiparam两个文件，并保存在同一路径fpath下。例：test/model，加载test/model.pdmodel和test/model.pdiparams两个文件。
    :param input_shape: 模型输入向量维度。仅考虑单个输入。
    :param batch_size: 模型输入批次大小。默认为1（单个输入）
    :param dtype: 输入tensor每个元素的类型，默认float32。
    :param only_weight: 是否仅计算参数所占空间。默认否，返回包括tensor其他信息所占空间的tensor大小。
    :return: (output_tensor_size, output_tensor_shape); 返回模型实际输出向量的size（含参数以外其他信息，单位byte）和shape。
    """
    # 创建 config。
    config = paddle.inference.Config(infer_model_path + '.pdmodel', infer_model_path + '.pdiparams')
    # config.enable_profile() # 输出运行时信息
    # config.disable_glog_info() # 关闭冗余信息
    # 根据 config 创建 predictor
    predictor = paddle.inference.create_predictor(config)

    # 获取输入的名称
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    # 设置输入
    input_shape = [batch_size] + list(input_shape)
    fake_input = np.zeros(input_shape).astype(dtype)
    input_handle.reshape(input_shape)
    input_handle.copy_from_cpu(fake_input)

    # 运行predictor
    predictor.run()

    # 获取输出
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()  # numpy.ndarray类型
    if only_weight:
        output_size = output_data.dtype.itemsize * output_data.size # 仅含参数的size大小，单位byte。
    else:
        output_size = sys.getsizeof(output_data)
    return output_size, output_data.shape


if __name__ == '__main__':
    print("Client Model Analyse:")
    c_infer_model_filepath = './client_model_infer'
    c_flops, c_model_size, c_output_size = model_analyse(c_infer_model_filepath, input_shape=(3, 32, 32),
                                                         batch_size=256)

    print("Server Model Analyse:")
    s_infer_model_filepath = './server_model_infer'
    s_flops, s_model_size, s_output_size = model_analyse(s_infer_model_filepath, input_shape=(64, 8, 8), batch_size=256)
# 此函数下版本可能废弃，改用calc_complex_output
def calc_output_tensor_size(infer_model_path, input_shape, batch_size=1, dtype='float', only_weight = False):
    """该函数返回模型的输出向量大小，是通过对模型构建预测器，直接计算得到的。
    注：此函数下版本可能废弃，因为它不能处理多个输入。请改用calc_complex_output。
    注：此函数不能用于计算多个输入&输出的模型。如有需要，请调用calc_complex_output函数。
    注：涉及paddle inference api请参考https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_doc/

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
    fake_input = make_fake_input(input_shape, batch_size=batch_size, dtype=dtype)
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
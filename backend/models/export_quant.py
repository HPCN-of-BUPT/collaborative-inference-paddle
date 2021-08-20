import paddle
import paddle.fluid as fluid
from paddleslim.quant import quant_post_dynamic
from config import cfg

paddle.enable_static()
quant_post_dynamic(
        model_dir='freezed_model',
        save_model_dir='freezed_model',
        model_filename='split_pruned_client_model',
        params_filename='split_pruned_client_params',
        weight_bits=8,
        save_model_filename='quant_split_pruned_client_model',
        save_params_filename='quant_split_pruned_client_params',
        generate_test_model= True)

quant_post_dynamic(
        model_dir='freezed_model',
        save_model_dir='freezed_model',
        model_filename='split_pruned_server_model',
        params_filename='split_pruned_server_params',
        weight_bits=8,
        save_model_filename='quant_split_pruned_server_model',
        save_params_filename='quant_split_pruned_server_params',
        generate_test_model= True)

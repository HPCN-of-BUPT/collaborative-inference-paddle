import paddleslim

__all__ = ['model_analyse',
           'calc_model_size',
           'calc_flops',
           'calc_output_tensor_size',
           'construct_simple_program']

from .paddle_eval import model_analyse
from .paddle_eval import calc_model_size
from .paddle_eval import calc_flops
from .paddle_eval import calc_output_tensor_size
from .paddle_eval import construct_simple_program
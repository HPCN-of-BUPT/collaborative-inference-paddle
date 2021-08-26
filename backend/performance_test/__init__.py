import paddleslim

__all__ = ['model_analyse',
           'calc_model_size',
           'calc_flops',
           'calc_output_tensor_size',
           'construct_simple_program',
           'calc_complex_output',
           'make_fake_input']

from .paddle_eval import model_analyse, calc_model_size, calc_flops, \
    calc_output_tensor_size, construct_simple_program, calc_complex_output, \
    make_fake_input
import numpy as np
import core
def reverse_int8(tensor, p = core.ERROR_RATE):
    p_complement = 1 - p
    # 根据误码率生成噪声
    binomial_noise = np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 1 + \
        np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 2 + \
        np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 4 + \
        np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 8 + \
        np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 16 + \
        np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 32 + \
        np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 64 + \
        np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 128
    # 按位翻转
    x_tmp_filter = np.array(tensor, dtype=np.uint8) & binomial_noise
    # 恢复为int8类型
    x_tmp_filter = np.array(x_tmp_filter, dtype=np.int8)
    return x_tmp_filter

def reverse_float32(tensor, p = core.ERROR_RATE):
    tensor_tmp = np.round(tensor * 256)
    tensor_tmp = np.array(tensor_tmp, dtype=np.uint8)
    p_complement = 1 - p
    binomial_noise = np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 1 + \
        np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 2 + \
        np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 4 + \
        np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 8 + \
        np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 16 + \
        np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 32 + \
        np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 64 + \
        np.random.binomial(1,p_complement,tensor.shape).astype(np.uint8) * 128
    x_tmp_filter = tensor_tmp & binomial_noise
    x_tmp_filter = np.array(x_tmp_filter, dtype=np.float32)
    return x_tmp_filter/255.0
from paddleslim.prune import Pruner
paddle.enable_static()


def get_pruned_params(train_program):
    params = []
    skip_vars = ['conv2d_0.w_0','batch_norm_0.w_0','batch_norm_0.b_0','batch_norm_0.w_1','batch_norm_0.w_2']  # skip the first conv2d layer
    for block in train_program.blocks:
        for param in block.all_parameters():
            if ('conv2d' in param.name)  :
                if  ('24' in param.name) or ('26' in param.name) :
                    if (param.name not in skip_vars):
                        params.append(param.name)#or ('batch_norm' in param.name)
                ## 
            elif ('yolo_block' in param.name) and  ('conv' in param.name):
                params.append(param.name)

    return params




pruned_params = cfg.prune_par.strip().split(",") #此处也可以通过写正则表达式匹配参数名
    print("pruned params: {}".format(pruned_params))
    pruned_ratios = [float(n) for n in cfg.prune_ratio]
    print("pruned ratios: {}".format(pruned_ratios))

    pruner = Pruner()
    train_program = pruner.prune(
        train_program,
        fluid.global_scope(),
        params=pruned_params,
        ratios=pruned_ratios,
        place=place,
        only_graph=False)[0]
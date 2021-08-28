from sqlalchemy import *

def add_client(db,Submodel,infos):
    flops = infos['flops']
    params = infos['params']
    output_size = infos['output_size']
    type = 2
    model_divide_id = 1
    submodel = Submodel(flops=flops,params=params,output_size=output_size,type=type,model_divide_id=model_divide_id)

    db.session.add(submodel)
    db.session.commit()

def add_server(db,Submodel,infos):
    flops = infos['flops']
    params = infos['params']
    output_size = infos['output_size']
    type = 1
    model_divide_id = 1
    submodel = Submodel(flops=flops, params=params, output_size=output_size, type=type,model_divide_id=model_divide_id)

    db.session.add(submodel)
    db.session.commit()

def get_eval_result(db,Submodel,model_divide_id):
    c_submodel = Submodel.query.filter(and_(Submodel.model_divide_id == model_divide_id,Submodel.type == 2)).first()
    c_results = {
        'flops': c_submodel.flops, 'params': c_submodel.params, 'output_size': c_submodel.output_size
    }

    # s_submodel = Submodel.query.filter(and_(Submodel.model_divide_id == model_divide_id,Submodel.type == 1)).first()
    # s_results = {
    #     'flops': s_submodel.flops, 'params': s_submodel.params, 'output_size': s_submodel.output_size
    # }
    s_submodel = Submodel.query.filter(and_(Submodel.model_divide_id == model_divide_id, Submodel.type == 1)).first()
    s_results = {
        'flops': c_submodel.flops, 'params': c_submodel.params, 'output_size': c_submodel.output_size
    }
    results = {
        'cloud':s_results,
        'edge':c_results
    }
    return results
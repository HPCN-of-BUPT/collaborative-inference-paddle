from db_model import Model,Restriction,Submodel,ModelDivide
from app import db

def add_restriction(data):
    #print(data['delay'])
    delay = float(data['delay'])
    bandwidth = float(data['bandwidth'])
    cloud_computation = float(data['cloud_computation'])
    edge_computation = float(data['edge_computation'])
    model_id = int(data['model_id'])
    restriction = Restriction(delay=delay,bandwidth=bandwidth,
                              cloud_computation=cloud_computation,edge_computation=edge_computation)
    db.session.add(restriction)
    db.session.commit()

    model_divide = ModelDivide(model_id=model_id,restric_id=restriction.id)
    db.session.add(model_divide)
    db.session.commit()

    return model_divide.id

def add_submodel(model_divide_id):
    edge_model = Submodel(flops=11,params=12,type=2)
    cloud_model = Submodel(flops=50,params=76,type=1)
    edge_cloud_model = Submodel(flops=61, params=120, type=0)

    db.session.add_all([edge_model,cloud_model,edge_cloud_model])
    db.session.commit()

    return edge_model,cloud_model,edge_cloud_model
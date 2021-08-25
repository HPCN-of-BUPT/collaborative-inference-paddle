def add_client(db,Submodel,infos):
    flops = infos['flops']
    params = infos['params']
    output_size = infos['output_size']
    type = 2
    submodel = Submodel(flops=flops,params=params,output_size=output_size,type=type)

    db.session.add(submodel)
    db.session.commit()

def add_server(db,Submodel,infos):
    flops = infos['flops']
    params = infos['params']
    output_size = infos['output_size']
    type = 1
    submodel = Submodel(flops=flops, params=params, output_size=output_size, type=type)

    db.session.add(submodel)
    db.session.commit()

def get_result(db,Submodel,id):
    system = System.query.filter(System.id == id).first()
    results = {
        'filename':system.filename,
        'edgetime': system.edge_time,
        'cloudtime': system.cloud_time,
        'transmitsize': system.transmit_size,
        'transmittime': system.transmit_time,
        'cloudedgeratio': system.cloud_edge_ratio,
        'time': system.time
    }
    return results
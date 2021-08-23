
def add_result(db,System,infos):
    transmit_size = infos.transmitsize
    edge_time = infos.edgetime
    cloud_time = infos.cloudtime
    transmit_time = infos.transmittime
    cloud_edge_ratio = cloud_time / edge_time
    time = edge_time + cloud_time + transmit_time

    system = System(transmit_size=transmit_size,edge_time=edge_time,cloud_time=cloud_time,
                    transmit_time=transmit_time,cloud_edge_ratio=cloud_edge_ratio,time=time)
    db.session.add(system)
    db.session.commit()
    return system.id

def get_result(db,System,id):
    system = System.query.filter(System.id == id).first()
    results = {
        'edgetime': system.edge_time,
        'cloudtime': system.cloud_time,
        'transmitsize': system.transmit_size,
        'transmittime': system.transmit_time,
        'cloudedgeratio': system.cloud_edge_ratio,
        'time': system.time
    }
    return results
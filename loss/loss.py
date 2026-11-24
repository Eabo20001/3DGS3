import copy
loss_fn = {}


def register(name):
    def decorator(cls):
        loss_fn[name] = cls
        return cls
    return decorator

def loss(loss_cfg, **kwargs):
    loss_args = loss_cfg['args']
    if kwargs is not None:
        loss_args.update(kwargs)
        
    return loss_fn[loss_cfg['name']](**loss_args)
import importlib
def get_loss(config, X, wp_bp, batch_size = None):
    module = importlib.import_module('deepcensor.loss.' + config['loss'])
    return getattr(module, "get_loss_" + config["censoring"])(X, wp_bp, batch_size)

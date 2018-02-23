import importlib

def get_model(config, training_loss, X_train, bias_init, embed_input_dim, ncol):
    module = importlib.import_module('deepcensor.model.' + config["structure"])
    return module.get_model(config, training_loss, X_train, bias_init, embed_input_dim, ncol)

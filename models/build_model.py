# +
import torch
import torchtuples as tt

from pycox.models import DeepHitSingle,CoxTime,CoxCC
from pycox.models.cox_time import MLPVanillaCoxTime
from torchtuples.practical import MLPVanilla
from torchtuples import Model, optim
from torch.nn import functional as F
# -

def get_activation(activation_name):
    """ Define the pytorch activation function
    # Argument
        activation_name : name of the activation name 
    # Return
        activation funcion as defined in the torch package
    """
    acti_dict = {
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh
    }
    return acti_dict[activation_name]


def get_optimizer(optimizer_name):
    """ Define the pytorch optimization algorithm
    #Argument
        optimizer_name: name of the optimization algorithm
    #Return
        optimization function as defined in the torch package
    """
    opti_dict = {
        'rmsprop': tt.optim.RMSprop,
        'adam': tt.optim.Adam,
        'adam_amsgrad': tt.optim.Adam,
        'sgdwr':tt.optim.SGD
    }
    return opti_dict[optimizer_name]


def build_model_net(config,in_features,labtrans =""):          
    """ Define the structure of the neural network
    # Arguments
        config: parser that contains all the parameters of the neural networks, for instance activation name, dropout rate, 
        neurons, number of layers
        in_features: number of input variables in the dataset
        labtrans: transformed input variables, including the time variable if CoxTime model or the discrete time variable 
        if DeepHit
    # Returns
        model: pycox model (based on pytorch) with the architecture defined previously
        callbacks: callbacks function
    """
    if labtrans !="":
        out_features = labtrans.out_features
    else:
        out_features = 1
    nb_neurons = [config.neurons]*config.layers
    callbacks = [tt.cb.EarlyStopping(load_best=False,checkpoint_model=False)]
    act = get_activation(config.acti_func)
    optim = get_optimizer(config.optim)(lr=config.lr, weight_decay=config.pen_l2)
    if config.acti_func=="relu":
        weights = lambda w: torch.nn.init.kaiming_normal_(w, nonlinearity='relu')
    elif config.acti_func == "tanh":
        weights = lambda w: torch.nn.init.xavier_normal_(w)

    if config.name == "CoxCC":
        net = MLPVanilla(in_features, 
                         nb_neurons, 
                         out_features, 
                         batch_norm = True,
                         dropout = config.dr, 
                         activation = act,
                         w_init_=weights,
                         output_bias=False)
        model = CoxCC(net, optim)
    elif config.name == "CoxTime":
        net = MLPVanillaCoxTime(in_features, 
                                nb_neurons, 
                                batch_norm = True, 
                                dropout = config.dr, 
                                activation = act,
                                w_init_=weights)
        model = CoxTime(net, optim, labtrans = labtrans)
    elif config.name == "DeepHit":    
        net = MLPVanilla(in_features, 
                          nb_neurons, 
                          out_features, 
                          batch_norm = True,
                          dropout = config.dr, 
                          activation = act, 
                          w_init_=weights,
                          output_bias=False)
        model = DeepHitSingle(net, 
                              optim,
                              alpha = config.alpha, 
                              sigma = config.sigma, 
                              duration_index = labtrans.cuts)
    if config.optim == "adam_amsgrad" : 
        optim.param_groups[0]['amsgrad'] = True
    elif config.optim == "sgdwr":
        optim.param_groups[0]['momentum'] = 0.9
        callbacks.append(tt.cb.LRCosineAnnealing())
        
    return model,callbacks
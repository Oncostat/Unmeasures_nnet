import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.build_model import *
from sklearn.model_selection import KFold

def objective_net(trial,df,df_mapper,dir_res,config):
    """ Define the structure of the neural network for a Cox-MLP (CC), CoxTime and  DeepHit
    # Arguments
        trial: number of the trial of the study to search for the best hyperparameters
        df: dataframe with all the data, input and output
        df_mapper: function to standardize the data
        dir_res: directory to the folder where to put results
        config: parser that contains all the parameters of the neural networks, for instance activation name, dropout rate, 
        neurons, number of layers
    # Returns
        min(val_mean): minimum of the mean error on the 5 validation folds
    """

    
    config.acti_func = trial.suggest_categorical("activation", ["relu","tanh"])
    config.batch_size = trial.suggest_categorical("batch_size",[8,16,32,64,128])
    config.layers = trial.suggest_int("n_layers", 1, 4)
    config.lr  = trial.suggest_uniform("learning_rate", 1e-3, 1e-2)
    config.neurons = trial.suggest_int("neurons", 4, 128)
    config.optim = trial.suggest_categorical("optimizer", ["adam","adam_amsgrad","rmsprop", "sgdwr"])

    if config.uncertainty == "FBMask":
        config.dr = 0.1
        
    elif config.uncertainty == "MCDropout":
        config.tau = trial.suggest_uniform("tau",0.025, 0.2)
        in_features = df.shape[1]
        in_samples = df.shape[0]
        lengthscale = 1e-2
        config.dr = 0.1
        reg = lengthscale**2 * (1 - config.dr) / (2. * in_samples * config.tau)
        config.pen_l2 = reg
        
    else: 
        config.pen_l2 = trial.suggest_uniform("l2", 0,0.1)
        config.dr = trial.suggest_uniform("dropout", 0.0, 0.3)
    
    if config.name=="DeepHit":
        config.num_durations = trial.suggest_categorical("num_durations", [10,50,100,200,400])
        labtrans = DeepHitSingle.label_transform(config.num_durations)
        config.alpha = trial.suggest_uniform("alpha",0,1)
        config.sigma = trial.suggest_categorical('sigma',[0.1,0.25,0.5,1,2.5,5,10,100])
    elif config.name=="CoxTime":
        labtrans = CoxTime.label_transform()
    else:
        labtrans=""
    t=0
    kf = KFold(n_splits=5,shuffle=True,random_state =397)
    train_loss = pd.DataFrame()
    val_loss = pd.DataFrame()
    for train_index, val_index in kf.split(df):
        
        df_train, df_val = df.iloc[train_index], df.iloc[val_index]
        df_train = df_mapper.fit_transform(df_train)
        df_val = df_mapper.transform(df_val).astype('float32')
        x_train = np.array(df_train.drop(['yy','status','id'], axis=1)).astype('float32')
        x_val = np.array(df_val.drop(['yy','status','id'], axis=1)).astype('float32')
        y_train = (df_train['yy'].values, df_train['status'].values)
        y_val = (df_val['yy'].values, df_val['status'].values)
            
        if labtrans !="":
            y_train = labtrans.fit_transform(*y_train)
            y_val = labtrans.transform(*y_val)

        val = tt.tuplefy(x_val, y_val)
        val = val.repeat(10).cat()
        in_features = x_train.shape[1]
        
        if config.uncertainty == "VAE":
            model,callbacks = build_model_vae(config,
                                     in_features,
                                     labtrans)
        else:
            model,callbacks = build_model_net(config,
                                     in_features,
                                     labtrans)
        log = model.fit(x_train, 
                        y_train, 
                        int(config.batch_size),
                        epochs = 500, 
                        callbacks = callbacks,
                        verbose = False,
                        val_data = val)

        df_log = log.to_pandas()[['train_loss', 'val_loss']]
        train_loss[t] = df_log['train_loss']
        val_loss[t] = df_log['val_loss']
        t+=1
    val_mean = np.mean(val_loss,axis=1)
    if config.plot_mode == True:
        b1, = plt.plot(np.mean(train_loss, axis = 1), label = 'train')
        b2, = plt.plot(val_mean, label = 'val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (mean on 5 folds)')
        plt.legend([b1,b2], ['train', 'val'], loc='upper right')
        plt.savefig(dir_res+'_'+str(config.layers)+'_'+str(config.neurons)+'_'+
                    str(config.lr)+'_'+str(config.dr)+'_'+config.acti_func+'_'+'_'+config.optim+".png")
        plt.close()
    return min(val_mean)
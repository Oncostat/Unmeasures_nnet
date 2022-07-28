import numpy as np 

def determine_surv_prob(surv,t):
    """ Function to compute survival probability at a given time t"""
    for (i,t2) in enumerate(surv.index):
        if (t2>t):
            if (i==0):
                
                result = (t/t2)*surv.iloc[i]+ (t2-t)/t2*1
                break
            else :
                result = (t-surv.index.values[i-1])/(t2-surv.index.values[i-1])*surv.iloc[i]+(t2-t)/(t2-surv.index.values[i-1])*surv.iloc[i-1]
                break
        else :
            
            result = surv.iloc[i]
    return(result)

def output_ic(survie,alpha):
    """ Function to compute confidence intervals
    Arguments
        survie: survival probability at a given timepoint
        alpha: level of confidence
    Returns
        lower: left value of the confidence interval
        moy: mean value of the M survival probabilities
        upper: right value of the confidence interval
    """
    ordered = sorted(survie, key=float)
    lower = round(np.percentile(ordered, 100*((1-alpha)/2)),3)
    upper = round(np.percentile(ordered, (alpha+(1-alpha)/2)*100),3)
    moy = round(np.mean(survie),3)
    return lower, moy, upper


def output_cr(survie_hat, survie,time,alpha=0.95):
    """ Function to output the coverage rate 
    Arguments
        survie_hat: estimated survival probability for a given time
        survie: theoretical survival probaiblity for  the same given time
        time: time that is chosen for computing survival probabilities
        alpha: confidence level
    Returns
        cr: coverage rate
    """
    lower, moy, upper = output_ic(survie_at,alpha)
    val_the = determine_surv_prob(survie,time)
    cr = sum(np.where((lower<val_the) & (val_the<upper) , 1, 0))/len(val_the)
    return cr

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
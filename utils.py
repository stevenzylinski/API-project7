def sigmoid_function(x):
        y = np.exp(x)/(1+np.exp(x))
        return(y)

def convert_sv(sv):
    """
    This function take a shap.Explanation object in input and convert all values to plot values according
    to the probability predicted.
    """
    
    # Total contribution of every feature
    total_values = [sum(x) for x in sv.values]
    
    # Base value predicted convert to probability
    proba_base = sigmoid_function([x for x in sv.base_values])
    
    # Final value predicted convert to probability
    proba_predict = sigmoid_function([x for x in sv.base_values+total_values])
    
    # Converting each contribution of feature regarding probability
    contrib = np.ndarray(shape = sv.values.shape)
    for x in range(0,len(sv.values)):
        contrib[x] = [(value/sum((sv.values[x]))*(proba_predict[x]-proba_base[x])) for value in sv.values[x]]
    
    sv.values = contrib
    sv.base_values = proba_base
    
    return(sv)
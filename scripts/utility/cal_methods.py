import numpy as np
from scipy.optimize import minimize 
from sklearn.metrics import log_loss


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)
    

class HistogramBinning(): 
    
    def __init__(self, M=15):
        self.bin_size = 1./M  # Calculate bin size
        self.conf = []  # Initiate confidence list
        self.upper_bounds = np.arange(self.bin_size, 1+self.bin_size, self.bin_size)  # Set bin bounds for intervals

    
    def _get_conf(self, conf_thresh_lower, conf_thresh_upper, probs, true):

        filtered = [x[0] for x in zip(true, probs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
        nr_elems = len(filtered)

        if nr_elems < 1:
            return 0
        else:
            conf = sum(filtered)/nr_elems
            return conf
    

    def fit(self, probs, true):

        conf = []

        # Got through intervals and add confidence to list
        for conf_thresh in self.upper_bounds:
            temp_conf = self._get_conf((conf_thresh - self.bin_size), conf_thresh, probs = probs, true = true)
            conf.append(temp_conf)

        self.conf = conf

            


    # Fit based on predicted confidence
    def predict(self, probs):

        # Go through all the probs and check what confidence is suitable for it.
        for i, prob in enumerate(probs):
            idx = np.searchsorted(self.upper_bounds, prob)
            probs[i] = self.conf[idx]

        return probs
        
        
class TemperatureScaling():
    
    def __init__(self, temp = 0, maxiter = 200, solver = "BFGS"):
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
    
    def _loss_fun(self, x, y_probs, y_true):
        scaled_probs = self.predict(y_probs, x)    
        loss = log_loss(y_true=y_true, y_pred=scaled_probs)
        return loss
    
    # Find the temperature
    def fit(self, y_logits_val, y_val):
        y_val = y_val.flatten() # Flatten y_val
        opt = minimize(self._loss_fun, x0 = 1, args=(y_logits_val, y_val), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x[0]
        
    # Scales logits based on the temperature
    def predict(self, y_logits, temp = None):
        if not temp:
            return softmax(y_logits/self.temp)
        else:
            return softmax(y_logits/temp)
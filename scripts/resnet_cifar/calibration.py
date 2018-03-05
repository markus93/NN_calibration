# Functions for calibration of results
from __future__ import division, print_function
import sklearn.metrics as metrics
import numpy as np

def evaluate_model(model, weights_file, x_test, y_test, bins = 15, verbose = True):
    """
    Evaluates the model, in addition calculates the calibration errors
    
    Args:
        model (keras.model): constructed model
        weights (string): path to weights file
        x_test: (numpy.ndarray) with x_test data (already in right format)
        y_test: (numpy.ndarray) with y_test data (1-hot vector)
        
    Returns:
        (acc, ece, mce): accuracy of model, ECE and MCE (calibration errors)
    """
    
    # First load in the weights
    model.load_weights(weights_file, by_name=True)
    
    # Next get predictions
    y_probs = model.predict(x_test)
    y_preds = np.argmax(y_probs, axis=1)
    y_true = y_test
    
    # Find accuracy and error
    y_true = [ np.where(r==1)[0][0] for r in y_true]  # 1-hot vector back to numeric
    accuracy = metrics.accuracy_score(y_true, y_preds) * 100
    error = 100 - accuracy
    
    # Confidence of prediction
    y_confs = np.max(y_probs, axis=1)  # Take only maximum confidence
    
    # Calculate ECE
    ece = ECE(y_confs, y_preds, y_true, bin_size = 1/bins)
    # Calculate MCE
    mce = MCE(y_confs, y_preds, y_true, bin_size = 1/bins)
    
    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)
    
    return (accuracy, ece, mce)



def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin
  

def ECE(conf, pred, true, bin_size = 0.1):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        ece: expected calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)        
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece
        
      
def MCE(conf, pred, true, bin_size = 0.1):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        mce: maximum calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    cal_errors = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        cal_errors.append(np.abs(acc-avg_conf))
        
    return max(cal_errors)
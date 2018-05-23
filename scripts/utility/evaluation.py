# Functions for calibration of results
from __future__ import division, print_function
import sklearn.metrics as metrics
import numpy as np
import pickle
import keras

    
def evaluate_model(model, weights_file, x_test, y_test, bins = 15, verbose = True, pickle_file = None, x_val = None, y_val = None):
    """
    Evaluates the model, in addition calculates the calibration errors and 
    saves the logits for later use, if "pickle_file" is not None.
    
    Parameters:
        model (keras.model): constructed model
        weights (string): path to weights file
        x_test: (numpy.ndarray) with test data
        y_test: (numpy.ndarray) with test data labels
        verbose: (boolean) print out results or just return these
        pickle_file: (string) path to pickle probabilities given by model
        x_test: (numpy.ndarray) with validation data
        y_test: (numpy.ndarray) with validation data labels

        
    Returns:
        (acc, ece, mce): accuracy of model, ECE and MCE (calibration errors)
    """
    
    # Change last activation to linear (instead of softmax)
    last_layer = model.layers.pop()
    last_layer.activation = keras.activations.linear
    i = model.input
    o = last_layer(model.layers[-1].output)
    model = keras.models.Model(inputs=i, outputs=[o])
    
    # First load in the weights
    model.load_weights(weights_file)
    model.compile(optimizer="sgd", loss="categorical_crossentropy")
    
    # Next get predictions
    y_logits = model.predict(x_test, verbose=1)
    y_probs = softmax(y_logits)
    y_preds = np.argmax(y_probs, axis=1)
    y_true = y_test
    
    # Find accuracy and error
    if y_true.shape[1] > 1:  # If 1-hot representation, get back to numeric   
        y_true = np.array([[np.where(r==1)[0][0]] for r in y_true]) # Back to np array also
       
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
        
     
    # Pickle probabilities for test and validation
    if pickle_file:
    
        #Get predictions also for x_val
        y_logits_val = model.predict(x_val)
        y_probs_val = softmax(y_logits_val)
        y_preds_val = np.argmax(y_probs_val, axis=1)
        
        # 
        if y_val.shape[1] > 1:  # If 1-hot representation, get back to numeric   
            y_val = np.array([[np.where(r==1)[0][0]] for r in y_val])  # Also convert back to np.array, TODO argmax?
            
        if verbose:
            print("Pickling the probabilities for validation and test.")
            print("Validation accuracy: ", metrics.accuracy_score(y_val, y_preds_val) * 100)
            
        # Write file with pickled data
        with open(pickle_file + '_logits.p', 'wb') as f:
            pickle.dump([(y_logits_val, y_val),(y_logits, y_true)], f)
    
    # Return the basic results
    return (accuracy, ece, mce)


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
    

def get_bin_info(conf, pred, true, bin_size = 0.1):

    """
    Get accuracy, confidence and elements in bin information for all the bins.
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        (acc, conf, len_bins): tuple containing all the necessary info for reliability diagrams.
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    accuracies = []
    confidences = []
    bin_lengths = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        accuracies.append(acc)
        confidences.append(avg_conf)
        bin_lengths.append(len_bin)
        
        
    return accuracies, confidences, bin_lengths
    
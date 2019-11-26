"""
Author      : Yi-Chieh Wu, Sriram Sankararman
Description : Twitter
"""

from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.
    
    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """
    
    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return
    
    np.savetxt(outfile, vec)    


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    index = 0
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        lines = fid.readlines() #read file line by line 
        for line in lines:
        	words = extract_words(line) #get list of words using extract_words
        	for word in words: #update dictionay if we have encountered new words
        		if word not in word_list.keys():
        			word_list.update({word : index})
        			index += 1


        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        lines = fid.readlines()
        for i in range(num_lines):
        	words = extract_words(lines[i]) #get list of words using extract_words
        	for word in words:
        		idx = word_list[word]
        		feature_matrix[i][idx] = 1
        ### ========== TODO : END ========== ###
        
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    if metric == 'accuracy':
    	return metrics.accuracy_score(y_true, y_label)
    elif metric == 'f1-score':
    	return metrics.f1_score(y_true, y_label)
    elif metric == 'auroc':
    	return metrics.roc_auc_score(y_true, y_label)
    elif metric == 'precision':
    	return metrics.precision_score(y_true, y_label)

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_label).ravel()
    if metric == 'sensitivity':
    	return float(tp) / (tp + fn)
    elif metric == 'specificity':
    	return float(tn) / (tn + fp)

    return 0
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance    
    score = 0
    n_splits = kf.get_n_splits(X,y)
    for i in range(n_splits):
    	for train_index, val_index in kf.split(X, y):
	    	# print("TRAIN:", train_index, "TEST:", val_index)

	    	X_train, X_val = X[train_index], X[val_index]
	    	y_train, y_val = y[train_index], y[val_index]
	    	clf.fit(X_train, y_train)

	    	y_pred = clf.decision_function(X_val)

	    	score += performance(y_val, y_pred, metric = metric)
    return score / n_splits
    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ':'
    C_range = 10.0 ** np.arange(-3, 3)
    ### ========== TODO : START ========== ###
    # part 2c: select optimal hyperparameter using cross-validation
    max_c = 0
    max_acc = 0
    for c in C_range:
    	clf = SVC(kernel = 'linear', C = c)
    	curr_acc = cv_performance(clf, X, y, kf, metric = metric)
    	if curr_acc > max_acc:
    		max_acc = curr_acc
    		max_c = c
    	print c, curr_acc
    return max_c 

    ### ========== TODO : END ========== ###


def select_param_rbf(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """
    
    print 'RBF SVM Hyperparameter Selection based on ' + str(metric) + ':'
    
    ### ========== TODO : START ========== ###
    # part 3b: create grid, then select optimal hyperparameters using cross-validation
    C_range = 10.0 ** np.arange(-3, 3)
    G_range = 2.0 ** np.arange(-11, 3, 2)
    max_c = 0 
    max_gamma = 0
    max_acc = 0
    for c in C_range:
    	for g in G_range:
    		clf = SVC(kernel = 'rbf', C = c, gamma = g)
    		curr_acc = cv_performance(clf, X, y, kf, metric = metric)
                print c, g, curr_acc
    		if curr_acc > max_acc:
    			max_acc = curr_acc
    			max_c = c
    			max_gamma = g

    print "Max_score:" , max_acc, "C:", c, "Gamma", g
    return max_gamma, max_c
    ### ========== TODO : END ========== ###


def shuffle_in_unison(a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
                shuffled_a[new_index] = a[old_index]
                shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 4b: return performance on test data by first computing predictions and then calling performance
    print "Perfomance Test for" , metric
    y_pred = clf.decision_function(X)
    score = performance(y, y_pred, metric = metric)
    return score
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the tweets and its labels   
    dictionary = extract_dictionary('./trainingX.txt')
    train_X = extract_feature_vectors('./trainingX.txt', dictionary)
    train_y = read_vector_file('./trainingy.txt')
    print dictionary
    metric_list = ["accuracy"] #["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    
    ### ========== TODO : START ========== ###
    # part 1c: split data into training (training + cross-validation) and testing set
    n_entries = train_y.shape[0]
    split_val = n_entries - (n_entries / 10)
    train_X, train_y = shuffle_in_unison(train_X, train_y)
   # train_X, test_X = np.split(X, [split_val])
   # train_y, test_y = np.split(y, [split_val])

    # part 2b: create stratified folds (5-fold CV)
    kf = StratifiedKFold(n_splits=5)
  #   for metric in metric_list:
		# c = select_param_linear(train_X, train_y, kf, metric = metric)


        # print score
    # part 2d: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    
    # part 3c: for each metric, select optimal hyperparameter for RBF-SVM using CV
    for metric in metric_list:
        g, c = select_param_rbf(train_X, train_y, kf, metric = metric)
        print "Best Gamma:" , g
        print "Best C:", c

	
    # part 4a: train linear- and RBF-kernel SVMs with selected hyperparameters
    #best_rbf_clf = SVC(kernel='rbf', C = c, gamma = g)
    #best_rbf_clf.fit(train_X, train_y)
    #for metric in metric_list:
    	#print "RBF:", performance_test(best_rbf_clf, test_X, test_y, metric = metric), "\n"
    # part 4c: report performance on test data
    
    #Best Gamma: 0.03125
    #Best C: 1.0
    ### ========== TODO : END ========== ###
    
    
if __name__ == "__main__" :
    main()

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, precision_recall_curve
import argparse
import os


"""
Calculate confusion matrix from normalized anomaly score [0,1] 
input:
path_y_test is the path to .npy containing the arrays of the ground truth label [m]
path_anomaly_score is path to  .npy containing the arrays of the normalized anomaly score from the network output [m]
threshold is the value set to convert score to categorical label
e.g.
score <= threshold = 0 
and score > threshold = 1
score defined as the confidence level of a sample being classified as non-anomalous

returns:
None 
creates figure for confusion matrix
"""
# parser = argparse.ArgumentParser()
# parser.add_argument('--path_y_test', help='the .npy path for the ground truth label array',required=True, type=str)
# parser.add_argument('--path_anomaly_score', help='the .npy path for the anomaly score',required=True, type=str)
# parser.add_argument('--threshold',help='treshold to convert score to classes', type = float,  default = 0.5)

# opt = parser.parse_args()
# y_test_path = opt.path_y_test
# anomaly_score = opt.path_anomaly_score
# threshold = opt.threshold


    

def compute_confusion_matrix(yTest_input,probs_predictions_norm_all, threshold, output_path):
    # convert score to categorical result
    probs_predictions_norm_all[probs_predictions_norm_all>threshold] = 1
    probs_predictions_norm_all[probs_predictions_norm_all<= threshold] = 0
    # compute confusion metrics
    conf_matrix = confusion_matrix(yTest_input, probs_predictions_norm_all)

    # plot it and save the figure
    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ["Anomalous", "Normal"])
    cm_display.plot()
    title = output_path.split("/")[1]
    cm_display.ax_.set_title(title+' threshold '+ str(threshold) +'\n')
    cm_display.figure_.savefig(output_path+' threshold '+ str(threshold) +'.png',dpi=176)
    return conf_matrix

def precision_recall_f1(yTest_input,probs_predictions):
    precision, recall, _ = precision_recall_curve(yTest_input, probs_predictions)
    f1 = 2*np.mean(precision)*np.mean(recall) / (np.mean(precision)+np.mean(recall))
    return np.mean(precision), np.mean(recall), f1

    


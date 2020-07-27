from __future__ import division, print_function

import numpy as np
import os
import pickle as cPickle
import itertools
import tfplot
import matplotlib
from sklearn.metrics import confusion_matrix

labels = ['G1', 'G2', 'G3','G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11']
label_dict = {0:'G1', 1:'G2', 2:'G3',3:'G4', 4:'G5', 5:'G6', 6:'G8', 7:'G9', 8:'G10', 9:'G11'}
ORIG_CLASS_IDS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11] # suturing labels
NEW_CLASS_IDS = range(len(ORIG_CLASS_IDS))

def compute_accuracy(prediction_seq, label_seq):
    """ Compute accuracy, the fraction of correct predictions.

    Args:
        prediction_seq: A 2-D int NumPy array with shape
            `[duration, 1]`.
        label_seq: A 2-D int NumPy array with the same shape.

    Returns:
        A float.
    """

    return np.mean(prediction_seq == label_seq, dtype=np.float)
	
def compute_N_accuracy(prediction_seq, label_seq):
    """ Compute accuracy, the fraction of correct predictions.

    Args:
        prediction_seq: A 2-D int NumPy array with shape
            `[duration, 3]`.
        label_seq: A 2-D int NumPy array with the same shape.

    Returns:
        A float.
    """
    match = np.zeros((prediction_seq.shape[0]))
    for i in range(prediction_seq.shape[0]):
       res = np.isin(prediction_seq[i,:],label_seq[i])
       match[i] = np.sum(res)
    return np.mean(match)


def compute_edit_distance(prediction_seq, label_seq):
    """ Compute segment-level edit distance.

    First, transform each sequence to the segment level by replacing any
    repeated, adjacent labels with one label. Second, compute the edit distance
    (Levenshtein distance) between the two segment-level sequences.

    Simplified example: pretend each input sequence is only 1-D, with
    `prediction_seq = [1, 3, 2, 2, 3]` and `label_seq = [1, 2, 2, 2, 3]`.
    The segment-level equivalents are `[1, 3, 2, 3]` and `[1, 2, 3]`, resulting
    in an edit distance of 1.

    Args:
        prediction_seq: A 2-D int NumPy array with shape
            `[duration, 1]`.
        label_seq: A 2-D int NumPy array with shape
            `[duration, 1]`.

    Returns:
        A nonnegative integer, the number of operations () to transform the
        segment-level version of `prediction_seq` into the segment-level
        version of `label_seq`.
    """

    def edit_distance(seq1, seq2):

        seq1 = [-1] + list(seq1)
        seq2 = [-1] + list(seq2)

        dist_matrix = np.zeros([len(seq1), len(seq2)], dtype=np.int)
        dist_matrix[:, 0] = np.arange(len(seq1))
        dist_matrix[0, :] = np.arange(len(seq2))

        for i in range(1, len(seq1)):
            for j in range(1, len(seq2)):
                if seq1[i] == seq2[j]:
                    dist_matrix[i, j] = dist_matrix[i-1, j-1]
                else:
                    operation_dists = [dist_matrix[i-1, j],
                                       dist_matrix[i, j-1],
                                       dist_matrix[i-1, j-1]]
                    dist_matrix[i, j] = np.min(operation_dists) + 1

        return dist_matrix[-1, -1]

    def segment_level(seq):
        segment_level_seq = []
        for label in seq.flatten():
            if len(segment_level_seq) == 0 or segment_level_seq[-1] != label:
                segment_level_seq.append(label)
        return segment_level_seq

    return edit_distance(segment_level(prediction_seq),
                         segment_level(label_seq))




def plot_confusion_matrix(correct_labels, predict_labels,labels, log_dir,title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=True):
    ''' 
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor
    
    Returns:
        summary: TensorFlow summary 
    
    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
        - Currently, some of the ticks dont line up due to rotations.
    '''
    
    cm = confusion_matrix(correct_labels, predict_labels)
    label_path = os.path.join(log_dir, 'confusion_matrix.pkl')
    with open(label_path, 'wb') as f:
                cPickle.dump(cm, f)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')
    
    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()
    
    fig = matplotlib.figure.Figure(figsize=(10,10), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')
    
    #classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    #classes = ['\n'.join(wrap(l, 40)) for l in classes]
    classes = labels
    tick_marks = np.arange(len(classes))
    
    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    
    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=20,verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary


def compute_metrics(prediction_seqs, label_seqs,log_dir,mode):
    """ Compute metrics averaged over sequences.

    Args:
        prediction_seqs: A list of int NumPy arrays, each with shape
            `[duration, 1]`.
        label_seqs: A list of int NumPy arrays, each with shape
            `[duration, 1]`.

    Returns:
        A tuple,
        mean_accuracy: A float, the average over all sequences of the
            accuracies computed on a per-sequence basis.
        mean_edit_distance: A float, the average over all sequences of
            edit distances computed on a per-sequence basis.
    """
    pred_label = []
    grnd_label = []
    #accuracies = [compute_accuracy(pred_seq, label_seq)
    #              for (pred_seq, label_seq) in zip(prediction_seqs, label_seqs)]
    accuracies = [compute_N_accuracy(pred_seq, label_seq)
                  for (pred_seq, label_seq) in zip(prediction_seqs, label_seqs)]
    accuracy_mean = np.mean(accuracies, dtype=np.float)
    edit_dists = [compute_edit_distance(pred_seq, label_seq)
                  for (pred_seq, label_seq) in zip(prediction_seqs, label_seqs)]
    edit_dist_mean = np.mean(edit_dists, dtype=np.float)
    for (pred_seq, label_seq) in zip(prediction_seqs, label_seqs):
        p = pred_seq.tolist()
        g = label_seq.tolist()
        pred_label.append(p)
        grnd_label.append(g)
    #print('*********************************ground',mode)
    pred_label = list(itertools.chain.from_iterable(pred_label))
    pred_label = list(itertools.chain.from_iterable(pred_label))
    grnd_label = list(itertools.chain.from_iterable(grnd_label))
    grnd_label = list(itertools.chain.from_iterable(grnd_label))
    for i in range(len(pred_label)):
        pred_label[i]= label_dict.get(pred_label[i])
        grnd_label[i]= label_dict.get(grnd_label[i])   
    #print(grnd_label)
    #print('*********************************predicted',mode )
    #print(pred_label)
    summary =  plot_confusion_matrix(grnd_label,pred_label,labels,log_dir,tensor_name=mode+'/image')
                  
    return accuracy_mean, edit_dist_mean,summary

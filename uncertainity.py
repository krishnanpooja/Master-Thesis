import tensorflow as tf
import data
import numpy as np
import os
import metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mode


def predict(sess, input_seqs, reset_seqs,path,entropy=True):
    saver = tf.train.import_meta_graph(path + 'model.ckpt.meta')
    saver.restore(sess, path + 'model.ckpt')
    if entropy:
        feature_map = tf.get_default_graph().get_tensor_by_name("logits/Softmax:0")
    else:
        feature_map = tf.get_default_graph().get_tensor_by_name("logits/logits:0")
    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name("inputs:0")
    resets = graph.get_tensor_by_name("resets:0")
    training = graph.get_tensor_by_name("training:0")

    batch_size = len(input_seqs)
    seq_durations = [len(seq) for seq in input_seqs]
    input_sweep, reset_sweep = data.sweep_generator(
            [input_seqs, reset_seqs], batch_size=batch_size).next()
    softmax = sess.run(feature_map, feed_dict={inputs: input_sweep, resets: reset_sweep, training: False})
    softmax_dur = [seq[:duration]
                   for (seq, duration) in zip(softmax, seq_durations)]
    if not entropy:
        prediction_seqs = [np.argmax(seq, axis=1).reshape(-1, 1)
                       for seq in softmax_dur]
        softmax_dur = prediction_seqs
    return softmax_dur

def calculate_entropy(entropy_matrix,test_label_seqs):
    entropy_matrix_mean = np.mean(entropy_matrix, axis=0)
    print('entropy_matrix_mean.shape:', entropy_matrix_mean.shape, entropy_matrix_mean[0, :])
    test_prediction_seqs = []
    start = 0
    for seq in test_label_seqs:
        size = seq.shape[0]+start
        arr = entropy_matrix_mean[start:size,:]
        #print(arr.shape)
        test_prediction_seqs.append(arr)
        start = size

    prediction_seqs = [np.argmax(seq, axis=1).reshape(-1, 1)
                       for seq in test_prediction_seqs]
    entropy_log = np.log2(entropy_matrix_mean, where=(entropy_matrix_mean != 0.0))
    entropy_log_mul = entropy_matrix_mean * entropy_log
    entropy = np.sum(entropy_log_mul, axis=1) * -1
    print('entropy[0]:', entropy.shape)
    #entropy = entropy/3.32
    normalized_entropy= (entropy - np.mean(entropy,axis=0))/np.std(entropy,axis=0)
    print('entropy[0]:', np.max(normalized_entropy))
    #entropy[entropy > 0.5] = 222
    #entropy[entropy < 0.5] = 111
    #print('entropy[0]:', np.unique(entropy))

    return normalized_entropy,prediction_seqs

def calculate_variation_ratio(entropy_matrix_1,test_label_seqs):
    value, count = mode(entropy_matrix_1, axis=0)
    print('count.shape:', value[0, 200, 0], count[0, 200, 0])
    varition_ratio = 1 - count / 50.0
    print(varition_ratio.shape)
    entropy = np.squeeze(varition_ratio)
    value = np.squeeze(value,axis=0)
    print('np.unique(entropy):',np.unique(value))
    test_prediction_seqs = []
    start = 0
    for seq in test_label_seqs:
        size = seq.shape[0]+start
        arr = value[start:size]
        #print(arr.shape)
        test_prediction_seqs.append(arr)
        start = size
    val = [s.shape for s in test_prediction_seqs]
    print(val)
    val = [s.shape for s in test_label_seqs]
    print(val)
    #entropy[entropy > 0.5] = 222
    #entropy[entropy < 0.5] = 111
    return entropy,test_prediction_seqs

def main():
    bEntropy = True
    input_model_paths = ['/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/ensemble_B/BidirectionalLSTM_0.10_1200_1_250_01_0.50_2.0000/C/999/','/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/ensemble_B/BidirectionalLSTM_0.10_1200_1_250_01_0.50_2.0000/D/999','/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/ensemble_B/BidirectionalLSTM_0.10_1200_1_250_01_0.50_2.0000/E/999/','/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/ensemble_B/BidirectionalLSTM_0.10_1200_1_250_01_0.50_2.0000/F/999/','/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/ensemble_B/BidirectionalLSTM_0.10_1200_1_250_01_0.50_2.0000/G/999/','/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/ensemble_B/BidirectionalLSTM_0.10_1200_1_250_01_0.50_2.0000/H/999/','/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/ensemble_B/BidirectionalLSTM_0.10_1200_1_250_01_0.50_2.0000/I/999/']

    '''
    input_model_paths = ['/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_MCdropout/BidirectionalLSTM_0.10_1200_1_250_01_0.50_2.0000/D/999/']
    
    # For NaiveEnsemble
    input_model_paths = ['/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_/BidirectionalLSTM_0.50_1200_1_512_03_0.50_2.0000/B_C_D_E_F_G_H_I/1/',
                         '/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_/BidirectionalLSTM_0.25_1200_1_512_03_0.50_2.0000/B_C_D_E_F_G_H_I/1/',
                         '/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_/BidirectionalLSTM_0.15_1200_1_512_03_0.50_2.0000/B_C_D_E_F_G_H_I/1/',
                         '/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_/BidirectionalLSTM_0.10_1200_1_512_03_0.50_2.0000/B_C_D_E_F_G_H_I/1/']
    '''
    '''
    #For Bootstrap Random Prior Ensemble
    input_model_paths=['/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_/BidirectionalLSTMWithRandomPrior_0.10_1200_1_250_01_0.50_2.0000/B/999/',
                       '/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_/BidirectionalLSTMWithRandomPrior_0.10_1200_1_250_01_0.50_2.0000/C/999/',
                       '/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_/BidirectionalLSTMWithRandomPrior_0.10_1200_1_250_01_0.50_2.0000/E/999/',
                       '/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_/BidirectionalLSTMWithRandomPrior_0.10_1200_1_250_01_0.50_2.0000/F/999/',
                       '/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_/BidirectionalLSTMWithRandomPrior_0.10_1200_1_250_01_0.50_2.0000/G/999/',
                       '/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_/BidirectionalLSTMWithRandomPrior_0.10_1200_1_250_01_0.50_2.0000/H/999/']

    '''
    '''
    #For Random Prior
    input_model_paths=['/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_/BidirectionalLSTMWithRandomPrior_0.10_1200_1_250_01_0.50_2.0000/B_C_D_E_F_G_H_I/1/',
                     '/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_/BidirectionalLSTMWithRandomPrior_0.25_1200_1_250_01_0.50_2.0000/B_C_D_E_F_G_H_I/1/',
                     '/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_/BidirectionalLSTMWithRandomPrior_0.50_1200_1_250_01_0.50_2.0000/B_C_D_E_F_G_H_I/1/']
    '''

    data_dir ='/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/'
    data_filename ='standardized_data_translate_crop.pkl'
    model_type = 'BidirectionalLSTM' #'BidirectionalLSTMWithRandomPrior' #
    test_users = 'D'
    test_users = test_users.split(' ')
    test_trial = 1

    dataset = data.Dataset(data_dir, data_filename, model_type)
    train_raw_seqs, test_raw_seqs = dataset.get_splits(test_users, test_trial)
    test_triplets = [data.prepare_raw_seq(seq) for seq in test_raw_seqs]
    test_input_seqs, test_reset_seqs, test_label_seqs = zip(*test_triplets)
    input_size = dataset.input_size


    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    no_of_samples = sum([len(seq) for seq in test_input_seqs])
    if bEntropy:
        entropy_matrix = np.zeros((len(input_model_paths), no_of_samples, 10))  # batch_size
    else:
        entropy_matrix = np.zeros((len(input_model_paths), no_of_samples, 1))  # batch_size
    # Add ops to save and restore all the variables.
    for k, path in enumerate(input_model_paths):
        #path = '/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/logs_translate_ensemble_/BidirectionalLSTM_0.50_1200_1_512_03_0.50_2.0000/B_C_D_E_F_G_H_I/1/'
        softmax_seqs = predict(sess,test_input_seqs,test_reset_seqs,path,bEntropy)
        entropy_matrix[k] = np.vstack(softmax_seqs)
        #print('softmax_seqs:',len(softmax_seqs))
    if bEntropy:
        measure,prediction_seqs = calculate_entropy(entropy_matrix,test_label_seqs)
    else:
        measure,prediction_seqs = calculate_variation_ratio(entropy_matrix,test_label_seqs)


    log_dir = '/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/ensemble_B/'
    vis_filename = 'MCDropout_User_D_Trial_1_LOUO.png'
    vis_path = os.path.join(log_dir, vis_filename)

    accuracies = [metrics.compute_accuracy(pred_seq, label_seq)
                  for (pred_seq, label_seq) in zip(prediction_seqs, test_label_seqs)]
    accuracy_mean = np.mean(accuracies, dtype=np.float)
    edit_dists = [metrics.compute_edit_distance(pred_seq, label_seq)
                  for (pred_seq, label_seq) in zip(prediction_seqs, test_label_seqs)]
    edit_dist_mean = np.mean(edit_dists, dtype=np.float)
    line = '%.6f  %08.3f     ' % (accuracy_mean,
                                   edit_dist_mean)
    fig, axes = data.visualize_predictions(prediction_seqs,test_label_seqs,10, measure)
    axes[0].set_title(line)
    plt.tight_layout()
    plt.savefig(vis_path)
    plt.close(fig)

if __name__ == '__main__':
    main()

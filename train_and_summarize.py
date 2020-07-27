from __future__ import division, print_function

import os
import time
import argparse
import shutil
import pickle as cPickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

import data
import models_kinematic_video as models
import optimizers
import metrics

import numpy as np
import math
from scipy.stats import mode

def define_and_process_args():
    """ Define and process command-line arguments.

    Returns:
        A Namespace with arguments as attributes.
    """

    description = main.__doc__
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=formatter_class)

    parser.add_argument('--data_dir', default='/volume/USERSTORE/kris_po/Suturing/features_Resnet_Imagenet/',
                        help='Data directory.')
    parser.add_argument('--data_filename', default='standardized_data_kinematic_video_correct.pkl',
                        help='''The name of the standardized-data pkl file that
                                resides in data_dir.''')
    parser.add_argument('--test_users', default='B C D E F G H I',
                        help='''A string of the users that make up the test set,
                                with users separated by spaces.''')

    parser.add_argument('--model_type', default='BidirectionalLSTM',
                        help='''The model type, either BidirectionalLSTM,
                                ForwardLSTM, or ReverseLSTM. or Conv3D or BidirectionalLSTMWithRandomPrior''')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='The number of hidden layers.')
    parser.add_argument('--hidden_layer_size', type=int, default=250,
                        help='The number of hidden units per layer.')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='''The fraction of inputs to keep whenever dropout
                                is applied.''')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='The number of sequences in a batch/sweep.')
    parser.add_argument('--num_train_sweeps', type=int, default=1200,
                        help='''The number of training sweeps. A sweep
                                is a collection of batch_size sequences that
                                continue together throughout time until all
                                sequences in the batch are exhausted. Short
                                sequences grow by being wrapped around in
                                time.''')
    parser.add_argument('--initial_learning_rate', type=float, default=2.0,
                         help='The initial learning rate.')
    parser.add_argument('--num_initial_sweeps', type=int, default=50,
                        help='''The number of initial sweeps before the
                                learning rate begins to decay.''')
    parser.add_argument('--num_sweeps_per_decay', type=int, default=200,
                        help='''The number of sweeps per learning-rate decay,
                                once decaying begins.''')
    parser.add_argument('--decay_factor', type=float, default=0.5,
                        help='The multiplicative learning-rate-decay factor.')
    parser.add_argument('--max_global_grad_norm', type=float, default=1.0,
                        help='''The global norm is the norm of all gradients
                                when concatenated together. If this global norm
                                exceeds max_global_grad_norm, then all gradients
                                are rescaled so that the global norm becomes
                                max_global_grad_norm.''')

    parser.add_argument('--init_scale', type=float, default=0.1,
                        help='''All weights will be initialized using a
                                uniform distribution over
                                [-init_scale, init_scale].''')
    parser.add_argument('--num_sweeps_per_summary', type=int, default=7,
                        help='''The number of sweeps between summaries. Note:
                                7 sweeps with 5 sequences per sweep corresponds
                                to (more than) 35 visited sequences, which is
                                approximately 1 epoch.''') #leaving one user ou 8-1=7 each 5 trials
    parser.add_argument('--num_sweeps_per_save', type=int, default=7,
                        help='The number of sweeps between saves.')
    parser.add_argument('--test_trial', type=int, default=999,
                        help='User Trial to be used for testing')
    parser.add_argument('--sample_times', type=int, default=50,
                        help='Number of Sample trials to measure uncertainity')

    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    args.test_users = args.test_users.split(' ')
    return args


def get_log_dir(args):
    """ Form a convenient log directory that summarizes arguments.

    Args:
        args: An object containing processed arguments as attributes.

    Returns:
        A string. The full path to log directory.
    """

    params_str = '_'.join([args.model_type,
                           '%.2f' % args.init_scale,
                           '%d' % args.num_train_sweeps,
                           '%d' % args.num_layers,
                           '%03d' % args.hidden_layer_size,
                           '%02d' % args.batch_size,
                           '%.2f' % args.dropout_keep_prob,
                           '%.4f' % args.initial_learning_rate])

    test_users_str = '_'.join(args.test_users)
    trial=str(args.test_trial)

    return os.path.join(args.data_dir, 'logs_kinematic_video', params_str, test_users_str,trial)

def compute_error(pred_seq,grnd_truth):
    np.append(err,np.count_nonzero(np.not_equal(pred_seq, grnd_truth)))
    return err

def compare(pred_seq,grnd_truth):
    #print('pred_seq.shape:',pred_seq.shape)
    err = np.equal(pred_seq, grnd_truth)
    return err

def train(sess, model, optimizer, log_dir, batch_size, num_sweeps_per_summary,
          num_sweeps_per_save, train_input_seqs, train_reset_seqs,
          train_label_seqs, test_input_seqs, test_reset_seqs, test_label_seqs,args):
    """ Train a model and export summaries.

    `log_dir` will be *replaced* if it already exists, so it certainly
    shouldn't be anything generic like `/home/user`.

    Args:
        sess: A TensorFlow `Session`.
        model: An `LSTMModel`.
        optimizer: An `Optimizer`.
        log_dir: A string. The full path to the log directory.
        batch_size: An integer. The number of sequences in a batch.
        num_sweeps_per_summary: An integer. The number of sweeps between
            summaries.
        num_sweeps_per_save: An integer. The number of sweeps between saves.
        train_input_seqs: A list of 2-D NumPy arrays, each with shape
            `[duration, input_size]`.
        train_reset_seqs: A list of 2-D NumPy arrays, each with shape
            `[duration, 1]`.
        train_label_seqs: A list of 2-D NumPy arrays, each with shape
            `[duration, 1]`.
        test_input_seqs: A list of 2-D NumPy arrays, each with shape
            `[duration, input_size]`.
        test_reset_seqs: A list of 2-D NumPy arrays, each with shape
            `[duration, 1]`.
        test_label_seqs: A list of 2-D NumPy arrays, each with shape
            `[duration, 1]`.
        args: An object containing processed arguments as attributes.
    """
    #print(test_label_seqs)
    ema = tf.train.ExponentialMovingAverage(decay=0.5)#0.5
    update_train_loss_ema = ema.apply([model.loss])
    train_loss_ema = ema.average(model.loss)
    tf.summary.scalar('train_loss_ema', train_loss_ema)

    train_accuracy = tf.placeholder(tf.float32, name='train_accuracy')
    train_edit_dist = tf.placeholder(tf.float32, name='train_edit_dist')
    test_accuracy = tf.placeholder(tf.float32, name='test_accuracy')
    test_edit_dist = tf.placeholder(tf.float32, name='test_edit_dist')
    #values = [train_accuracy, train_edit_dist, test_accuracy, test_edit_dist]
    #tags = [value.op.name for value in values]

    tf.summary.scalar('learning_rate', optimizer.learning_rate)
    for value in [train_accuracy, train_edit_dist, test_accuracy, test_edit_dist]:
        tf.summary.scalar(value.op.name, value)

    #tf.summary.scalar(tags, tf.stack(values))

    summary_op = tf.summary.merge_all()

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    summary_writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    num_sweeps_visited = 0
    start_time = time.time()
    train_gen = data.sweep_generator(
        [train_input_seqs, train_reset_seqs, train_label_seqs],
        batch_size=batch_size, shuffle=True, num_sweeps=None)
    while num_sweeps_visited <= optimizer.num_train_sweeps:

        if num_sweeps_visited % num_sweeps_per_summary == 0:
            test_prediction_seqs = []
            train_prediction_seqs,logits = models.predict(
                sess, model, train_input_seqs, train_reset_seqs)
            train_accuracy_, train_edit_dist_,train_confusion_matrix = metrics.compute_metrics(
                train_prediction_seqs, train_label_seqs,log_dir,'train')
            print('test_input_seqs:',len(test_input_seqs))
            err=0.0
            # init empty predictions
            no_of_samples = sum([len(seq) for seq in test_input_seqs])
            entropy_matrix = np.zeros((args.sample_times,no_of_samples,10)) #batch_size
            entropy_matrix_1 = np.zeros((args.sample_times, no_of_samples, 1))  # batch_size
            for sample_id in range(50):
                test_prediction_seqs,softmax_seqs = models.predict(sess, model, test_input_seqs, test_reset_seqs)
                #print('softmax_dur:', softmax_seqs)
                entropy_matrix[sample_id] = np.vstack(softmax_seqs)
                entropy_matrix_1[sample_id] = np.vstack(test_prediction_seqs)

            ''' #Variation Ratio
            value,count = mode(entropy_matrix_1,axis=0)
            print('count.shape:', value[0,200,0], count[0,200,0])
            varition_ratio=1-count/50.0
            print(varition_ratio.shape)
            entropy=np.squeeze(varition_ratio)
            entropy[entropy > 0.5] = 222
            entropy[entropy < 0.5] = 111'''
            #MC Dropout - Entropy

            #print('entropy_matrix.shape:',entropy_matrix.shape,entropy_matrix[0,0,:])
            entropy_matrix_mean=np.mean(entropy_matrix,axis=0)
            #print('entropy_matrix_mean.shape:', entropy_matrix_mean.shape,entropy_matrix_mean[0,:])
            entropy_log = np.log2(entropy_matrix_mean,where=(entropy_matrix_mean!=0.0))
            #print('entropy_log:',entropy_log[0,:])
            entropy_log_mul = entropy_matrix_mean * entropy_log
            #print('entropy_log_mul.shape:', entropy_log_mul.shape, entropy_log_mul[0, :])
            entropy=np.sum(entropy_log_mul,axis=1)*-1
            #print('entropy.shape:', entropy.shape,entropy[0])
            normalized_entropy = (entropy - np.mean(entropy, axis=0)) / np.std(entropy, axis=0)

            test_accuracy_, test_edit_dist_,test_confusion_matrix = metrics.compute_metrics(
                test_prediction_seqs, test_label_seqs,log_dir,'test')
            summary = sess.run(summary_op,
                               feed_dict={train_accuracy: train_accuracy_,
                                          train_edit_dist: train_edit_dist_,
                                          test_accuracy: test_accuracy_,
                                          test_edit_dist: test_edit_dist_})
            print('kris_po:: num_sweeps_visited:',num_sweeps_visited)
            summary_writer.add_summary(summary, global_step=num_sweeps_visited)
            summary_writer.add_summary(train_confusion_matrix, global_step=num_sweeps_visited)
            summary_writer.add_summary(test_confusion_matrix, global_step=num_sweeps_visited)

            status_path = os.path.join(log_dir, 'status.txt')
            with open(status_path, 'w') as f:
                line = '%05.1f      ' % ((time.time() - start_time)/60)
                line += '%04d      ' % num_sweeps_visited
                line += '%.6f  %08.3f     ' % (train_accuracy_,
                                               train_edit_dist_)
                line += '%.6f  %08.3f     ' % (test_accuracy_,
                                               test_edit_dist_)
                print(line, file=f)

            label_path = os.path.join(log_dir, 'test_label_seqs.pkl')
            with open(label_path, 'wb') as f:
                cPickle.dump(test_label_seqs, f)

            pred_path = os.path.join(log_dir, 'test_prediction_seqs.pkl')
            with open(pred_path, 'wb') as f:
                cPickle.dump(test_prediction_seqs, f)

            if num_sweeps_visited == 1197:
                vis_filename = 'test_visualizations_%06d.png' % num_sweeps_visited
                vis_path = os.path.join(log_dir, vis_filename)
                fig, axes = data.visualize_predictions(test_prediction_seqs,
                                                       test_label_seqs,
                                                       model.target_size,normalized_entropy)
                axes[0].set_title(line)
                plt.tight_layout()
                plt.savefig(vis_path)
                plt.close(fig)

        if num_sweeps_visited % num_sweeps_per_save == 0:
            saver.save(sess, os.path.join(log_dir, 'model.ckpt'))

        train_inputs, train_resets, train_labels = train_gen.next()
        # We squeeze here because otherwise the targets would have shape
        # [batch_size, duration, 1, num_classes].
        train_targets = data.one_hot(train_labels, model.target_size)
        train_targets = train_targets.squeeze(axis=2)

        _, _, num_sweeps_visited = sess.run(
            [optimizer.optimize_op,
             update_train_loss_ema,
             optimizer.num_sweeps_visited],
            feed_dict={model.inputs: train_inputs,
                       model.resets: train_resets,
                       model.targets: train_targets,
                       model.training: True})


def main():
    """ Run training and export summaries to data_dir/logs for a single test
    setup and a single set of parameters. Summaries include a) TensorBoard
    summaries, b) the latest train/test accuracies and raw edit distances
    (status.txt), c) the latest test predictions along with test ground-truth
    labels (test_label_seqs.pkl, test_prediction_seqs.pkl), d) visualizations
    as training progresses (test_visualizations_######.png)."""
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    args = define_and_process_args()
    print('\n', 'ARGUMENTS', '\n\n', args, '\n')

    log_dir = get_log_dir(args)
    print('\n', 'LOG DIRECTORY', '\n\n', log_dir, '\n')

    standardized_data_path = os.path.join(args.data_dir, args.data_filename)
    if not os.path.exists(standardized_data_path):
        message = '%s does not exist.' % standardized_data_path
        raise ValueError(message)

    dataset = data.Dataset(args.data_dir, args.data_filename,args.model_type)
    train_raw_seqs, test_raw_seqs = dataset.get_splits(args.test_users,args.test_trial)
    if args.model_type=='Conv3d':
        train_triplets = [data.prepare_raw_frame_seq(seq) for seq in train_raw_seqs]
        test_triplets = [data.prepare_raw_frame_seq(seq) for seq in test_raw_seqs]
        
    else:
        train_triplets = [data.prepare_raw_seq(seq) for seq in train_raw_seqs]
        test_triplets = [data.prepare_raw_seq(seq) for seq in test_raw_seqs]  

    train_input_seqs, train_reset_seqs, train_label_seqs = zip(*train_triplets)
    test_input_seqs, test_reset_seqs, test_label_seqs = zip(*test_triplets)

    
    Model = eval('models.' + args.model_type + 'Model')
    input_size = dataset.input_size
    target_size = dataset.num_classes

    # This is just to satisfy a low-CPU requirement on our cluster
    # when using GPUs.
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        config = tf.ConfigProto(intra_op_parallelism_threads=2,
                                inter_op_parallelism_threads=2)
    else:
        config = None

    with tf.Session(config=config) as sess:
        model = Model(input_size, target_size, args.num_layers,
                      args.hidden_layer_size, args.init_scale,
                      args.dropout_keep_prob)
        optimizer = optimizers.Optimizer(
            model.loss, args.num_train_sweeps, args.initial_learning_rate,
            args.num_initial_sweeps, args.num_sweeps_per_decay,
            args.decay_factor, args.max_global_grad_norm)
        train(sess, model, optimizer, log_dir, args.batch_size,
              args.num_sweeps_per_summary, args.num_sweeps_per_save,
              train_input_seqs, train_reset_seqs, train_label_seqs,
              test_input_seqs, test_reset_seqs, test_label_seqs,args)


if __name__ == '__main__':
    main()

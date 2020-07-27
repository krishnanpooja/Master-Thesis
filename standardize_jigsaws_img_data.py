#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:23:20 2019

@author: kris_po
"""

from __future__ import division, print_function

import os
import argparse
import itertools

import numpy as np
import pandas as pd
import pickle as cPickle

from imgaug import augmenters as ia
import cv2

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

import data

DATASET_NAME = 'JIGSAWS'
ORIG_CLASS_IDS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11] # suturing labels
NEW_CLASS_IDS = range(len(ORIG_CLASS_IDS))
CLASSES = ['G%d' % id for id in ORIG_CLASS_IDS]
NUM_CLASSES = len(CLASSES)

# Standard JIGSAWS experimental setup. In particular, it's the only
# recognition setup that exists, corresponding to
# JIGSAWS/Experimental/Suturing/unBalanced/GestureRecognition/UserOut
# (User H's 2nd trial is left out because no video was available for labeling.)    
USER_TO_TRIALS = {
  'B': [1, 2, 3, 4, 5],
   'C': [1, 2, 3, 4, 5],
   'D': [1, 2, 3, 4, 5],
   'E': [1, 2, 3, 4, 5],
   'F': [1, 2, 3, 4, 5],
   'G': [1, 2, 3, 4, 5],
   'H': [1,    3, 4, 5],
  'I': [1, 2, 3, 4, 5]
}

ALL_USERS = sorted(USER_TO_TRIALS.keys())

#_USECOLS = [c-1 for c in [39, 40, 41, 51, 52, 53, 57,
#                                    58, 59, 60, 70, 71, 72, 76]]
_USECOLS = [c-1 for c in range(0,100)]
KINEMATICS_COL_NAMES = ['pos_x', 'pos_y', 'pos_z', 'vel_x',
                        'vel_y', 'vel_z', 'gripper']*2

LABELS_USECOLS = [0, 1, 2]
LABELS_COL_NAMES = ['start_frame', 'end_frame', 'string_label']
LABELS_CONVERTERS = {2: lambda string_label: int(string_label.replace('G', ''))}

STANDARDIZED_COL_NAMES = KINEMATICS_COL_NAMES + ['label']

df_labels = pd.read_csv("/home/kris_po/label_final.csv")
#df_labels['label'] = df_labels['label']-1

def define_and_process_args():
    """ Define and process command-line arguments.

    Returns:
        A Namespace with arguments as attributes.
    """

    description = main.__doc__
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=formatter_class)

    parser.add_argument('--data_dir', default='/automount_offline/nil/home_mixed/frames_SU/',
                        help='Data directory.')
    parser.add_argument('--data_filename', default='standardized_img_data_H.pkl',
                        help='''The name of the standardized-data pkl file that
                                we'll create inside data_dir.''')

    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    return args


def get_trial_name_capture1(user, trial,capture):
    """ Form a trial name that matches standard JIGSAWS filenames.

    Args:
        user: A string.
        trial: An integer.

    Returns:
        A string.
    """
    return 'Suturing_%s%03d%s' % (user, trial,capture)

def load_kinematics(data_dir, trial_name):
    """ Load kinematics data.

    Args:
        data_dir: A string.
        trial_name: A string.

    Returns:
        A 2-D NumPy array with time on the first axis.
    """
    name = trial_name
    first_file=True
    capture = '_capture2' in trial_name
    trial_name=trial_name.replace('_capture1','')
    trial_name=trial_name.replace('_capture2','')
    file_list = df_labels[df_labels['filename'].str.contains(trial_name)]['filename']
    file_list = downsample(file_list,10)
    #file_list.pop(0)
    #print(file_list)
    for file in file_list:
        if capture:
            file=file.replace('test','_capture2test')
        else:
            file=file.replace('test','_capture1test')
        kinematics_path = os.path.join(data_dir, file)
        print(kinematics_path)
        np_image = cv2.imread(kinematics_path)
        shape = np_image.shape
        if first_file:
            if shape[0]==480:
                img_data = ia.Scale(0.5).augment_image(np_image)
            else:
                img_data = np_image
            first_file = False
            img_data = np.expand_dims(img_data, axis=0)
        else:
            if shape[0]==480:
                images = ia.Scale(0.5).augment_image(np_image)
            else:
                images = np_image
            images = np.expand_dims(images, axis=0)
            img_data=np.concatenate([img_data,images],axis=0)
    print('img_data',img_data.shape)
    return np.array(img_data)


def load_kinematics_and_labels(data_dir, trial_name):
    """ Load kinematics data and labels.

    Args:
        data_dir: A string.
        trial_name: A string.

    Returns:
        A 2-D NumPy array with time on the first axis. Labels are appended
        as a new column to the raw kinematics data (and are therefore
        represented as floats).
    """
    kinematics_data = load_kinematics(data_dir, trial_name)
    trial_name=trial_name.replace('_capture1','')
    trial_name=trial_name.replace('_capture2','')
    val = df_labels.loc[df_labels['filename'].str.match(trial_name),['label']]
    labels_data=np.array(val)
    labels_data = downsample(labels_data,10)
    print(kinematics_data.shape,labels_data.shape)
    #labels_data = np.append([kinematics_data, labels_data], axis=0)
    labeled_data_only_mask = labels_data.flatten() != 0

    return kinematics_data,labels_data

def load_kinematics_and_new_labels(data_dir, trial_name):
    """ Load kinematics data and standardized labels.

    Args:
        data_dir: A string.
        trial_name: A string.

    Returns:
        A 2-D NumPy array with time on the first axis. Labels are appended as
        a new column and are converted from arbitrary labels (e.g., 1, 3, 5)
        to ordered, nonnegative integers (e.g., 0, 1, 2).
    """
    data,labels = load_kinematics_and_labels(data_dir, trial_name)
    for orig, new in zip(ORIG_CLASS_IDS, NEW_CLASS_IDS):
        mask = labels[:, -1] == orig
        labels[mask, -1] = new
    return (data,labels)


def downsample(data, factor=6):
    """ Downsample a data matrix.

    Args:
        data: A 2-D NumPy array with time on the first axis.
        factor: The factor by which we'll downsample.

    Returns:
        A 2-D NumPy array.
    """
    return data[::factor]

def main():
    """ Create a standardized data file from raw data. """

    args = define_and_process_args()

    print('Standardizing JIGSAWS..')
    print()

    print('%d classes:' % NUM_CLASSES)
    print(CLASSES)
    print()

    user_to_trial_names = {}
    for user, trials in USER_TO_TRIALS.items():
        user_to_trial_names[user] = [get_trial_name_capture1(user, trial, capture)
                                     for trial in trials
                                     for capture in ['_capture1','_capture2']]
                                     #for action in ['Needle_Passing_','Knot_Tying_','Suturing_']]      

    print('Users and corresponding trial names:')
    for user in ALL_USERS:
        print(user, '   ', user_to_trial_names[user])
    print()


    all_trial_names = sorted(list(
        itertools.chain(*user_to_trial_names.values())
    ))
    print('All trial names, sorted:')
    print(all_trial_names)
    print()

    # Original data is at 30 Hz.
    all_data = {trial_name: 
                    load_kinematics_and_new_labels(args.data_dir, trial_name)   for trial_name in all_trial_names}
    print('Downsampled to 5 Hz.')
    print()

    export_dict = dict(
        dataset_name=DATASET_NAME, classes=CLASSES, num_classes=NUM_CLASSES,
        col_names=STANDARDIZED_COL_NAMES, all_users=ALL_USERS,
        user_to_trial_names=user_to_trial_names,
        all_trial_names=all_trial_names, all_data=all_data)
    standardized_data_path = os.path.join(args.data_dir, args.data_filename)
    with open(standardized_data_path, 'wb') as f:
        #cPickle.dump(all_data, f,protocol=2)
        cPickle.dump(export_dict,f,protocol=2)
    print('Saved standardized data file %s.' % standardized_data_path)
    print()


if __name__ == '__main__':
    main()

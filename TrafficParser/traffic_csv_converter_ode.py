#!/usr/bin/env python
"""
traffic_csv_conveter_overlap.py creates numpy arrays of overlap sessions for each class and vpn types.
the input for this module are pre-craeted csv created in generic_parser.py->traffic_csv_merger.py
"""

import os
import argparse
import csv
from sessions_plotter import *
import glob
import re
from sklearn.model_selection import train_test_split
import time

FLAGS = None
INPUT = "../raw_csvs/classes/browsing/reg/CICNTTor_browsing.raw.csv"#"../dataset/iscxNTVPN2016/CompletePCAPs" # ""
INPUT_DIR = "../raw_csvs/classes/chat/reg/"
CLASSES_DIR = "../raw_csvs/classes/**/**/"
CLASSES_DIR_REG = "../raw_csvs/classes/**/reg/"
EXCLUDE_SPEC_APP_OUTPUT_DIR = "../raw_csvs/exclude_specific_apps/"
DATASET_DIR = "../datasets/"

##### FOR OVERLAP ########
TPS = 60 # TimePerSession in secs
DELTA_T = 15 # Delta T between splitted sessions
MIN_TPS = 40
MIN_LENGTH = 10

TEST_SIZE = 0.1
LENGTH = 10

CLASSES_LIST = ["voip", "video", "file_transfer", "chat", "browsing"]
VPN_LIST = ["reg", "vpn", "tor"]

def export_dataset(dataset):
    print("Start export dataset")
    np.save(os.path.splitext(INPUT)[0], dataset)
    print(dataset.shape)


def export_class_dataset(dataset, class_dir, tag=None):
    print("Start export dataset")
    np.save(class_dir + "/" + "_".join(re.findall(r"[\w']+", class_dir)[-2:]) + tag, dataset)
    print(dataset.shape)


def export_exclude_specific_applications_dataset(dataset, class_dir, tag=None):
    print("Start export dataset")
    np.save(EXCLUDE_SPEC_APP_OUTPUT_DIR + "_".join(re.findall(r"[\w']+", class_dir)[-2:]) + tag, dataset)
    print(dataset.shape)


def export_separate_applications_list_dataset(dataset_dict, file_path):
    print("Start export dataset")
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    for name, array in dataset_dict.items():
        np.save(file_path + "_" + name, array)


def import_dataset():
    print("Import dataset")
    dataset = np.load(os.path.splitext(INPUT)[0] + ".npy")
    print(dataset.shape)
    return dataset


def traffic_csv_converter_ode_const_length(rows_list):
    print("Running in traffic_csv_converter_ode_const_length over " + str(len(rows_list)) + " rows")
    dataset = []

    counter = 0
    for i, row in enumerate(rows_list):
        # print row[0], row[7]
        session_tuple_key = tuple(row[:8])
        length = int(row[7])
        ts = np.array(row[8:8+length], dtype=float)
        sizes = np.array(row[9+length:], dtype=int)

        if length > MIN_LENGTH:

            for t in range(0, length, LENGTH):
                if t + LENGTH < length:
                    if np.all(np.diff(ts[t:t + LENGTH]) > 0.00000001) and not np.all(np.diff(ts[t:t + LENGTH]) > 0):
                        print("#####")
                    if np.all(np.diff(ts[t:t+LENGTH]) > 0.00000001):
                        dataset.append([sizes[t:t+LENGTH], ts[t:t+LENGTH]])
                        counter += 1
                        if counter % 100 == 0:
                            print(counter)
                    else:
                        print(ts[t:t + LENGTH])

    return np.asarray(dataset)


def traffic_class_converter_ode_const_length(dir_path):
    csvs_lines = []
    for file_path in [os.path.join(dir_path, fn) for fn in next(os.walk(dir_path))[2] if (".csv" in os.path.splitext(fn)[-1])]:
        print("Extracting rows from " + file_path)
        for line in csv.reader(open(file_path, 'r')):
            csvs_lines.append(line)

    print(len(csvs_lines))
    train_rows, test_rows = train_test_split(csvs_lines, test_size=TEST_SIZE)
    print(len(train_rows), len(test_rows))
    return traffic_csv_converter_ode_const_length(train_rows), traffic_csv_converter_ode_const_length(test_rows)


def iterate_overlap_all_classes():
    ## the main class of this module
    for class_dir in glob.glob(CLASSES_DIR):
        if "other" not in class_dir: # and "browsing" not in class_dir:
            print("working on " + class_dir)
            dataset_train, dataset_test = traffic_class_converter_ode_const_length(class_dir)
            print(dataset_train.shape, dataset_test.shape)
            export_class_dataset(dataset_train, class_dir, tag="_ode_const_" + str(LENGTH) + "_train")
            export_class_dataset(dataset_test, class_dir, tag="_ode_const_" + str(LENGTH) + "_test")


def random_sampling_dataset(input_array, size=2000):
    ## use this if you want to sample your data ranomly in order to decrease your data size.
    print("Import dataset " + input_array)
    dataset = np.load(input_array)
    print(dataset.shape)
    p = size*1.0/len(dataset)
    print(p)
    if p >= 1:
        print("Exception P>1 for " + input_array)
        return

    mask = np.random.choice([True, False], len(dataset), p=[p, 1-p])
    dataset = dataset[mask]
    print(dataset.shape)
    print("Start export dataset")

    # np.save(os.path.splitext(input_array)[0] + "_samp", dataset)
    np.save(os.path.splitext(input_array)[0], dataset)


def random_sampling_ode_const_length(class_type, vpn_type, size=3000):
    ## use this if you want to sample your data ranomly in order to decrease your data size.
    input_array_train = "../raw_csvs/classes/" + class_type + "/" + vpn_type + "/" + class_type + "_" + vpn_type + "_ode_const_" + str(LENGTH) + "_train.npy"
    input_array_test = "../raw_csvs/classes/" + class_type + "/" + vpn_type + "/" + class_type + "_" + vpn_type + "_ode_const_" + str(LENGTH) + "_test.npy"

    random_sampling_dataset(input_array_train, size=size)
    random_sampling_dataset(input_array_test, size=size*TEST_SIZE)


if __name__ == '__main__':
    iterate_overlap_all_classes()

    for traffic_class in CLASSES_LIST:
        for vpn in VPN_LIST:
            if traffic_class == "browsing" and vpn == "vpn":
                continue
            random_sampling_ode_const_length(traffic_class, vpn)

    # # FOR MEASURE
    # start_time = time.time()
    # file_path = "./test_pcaps/my_chat/my_facebook_chat.csv"
    # dataset = traffic_csv_converter_ode_const_length(list(csv.reader(open(file_path, 'r'))))
    # total_time = time.time() - start_time
    # print(dataset.shape)
    # print(dataset[0][0].shape)
    # print(dataset[0][1].shape)
    # print("--- %s seconds ---" % total_time)
    # print("--- %s seconds per sample ---" % (total_time / len(dataset)))

    ## OLD ###
    # iterate_overlap_all_classes()

    # random_sampling_overlap_dataset("browsing", "reg")
    # random_sampling_overlap_dataset("voip", "reg")

    # traffic_class_converter_exclude_specific_applications_overlap(INPUT_DIR, ['facebook'])
    # random_sampling_exclude_specific_applications_overlap_dataset("video", "reg", ['vimeo', 'youtube'])
    # traffic_class_converter_separate_applications_list_overlap(INPUT_DIR, ['facebook', 'hangouts', 'skype', 'voipbuster'])

    # iterate_non_overlap_all_classes(CLASSES_DIR_REG)
    # iterate_overlap_all_classes_over_specific_ICSX_dataset("tor")

    # dataset_train, dataset_test = traffic_class_converter_non_overlap(INPUT_DIR)


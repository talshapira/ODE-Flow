#!/usr/bin/env python
"""
datasets_generator_overlap.py creates a final dataset ready to be inserted to a model.
The input for this module are pre-created numpy array containing all classes overlap session 2d_histograms created in traffic_csv_conveter_overlap.py
"""
import glob
import numpy as np
import os


TEST_SIZE = 0.1
DATASET_DIR = "../datasets/ode/"

EXCLUDE_SPEC_APP_DIR = "../raw_csvs/exclude_specific_apps/"

VPN_TYPES = {
    "reg": glob.glob("../raw_csvs/classes/**/reg/*.npy"),
    "vpn": glob.glob("../raw_csvs/classes/**/vpn/*.npy"),
    "tor": glob.glob("../raw_csvs/classes/**/tor/*.npy")
}

CICNTTor_TYPES = {
    "reg": glob.glob("../raw_csvs/classes/**/reg/*CICNTTor.npy"),
    "tor": glob.glob("../raw_csvs/classes/**/tor/*CICNTTor.npy")
}

iscx_TYPES = {
    "reg": glob.glob("../raw_csvs/classes/**/reg/*iscx.npy"),
    "vpn": glob.glob("../raw_csvs/classes/**/vpn/*iscx.npy"),
}

CLASS_LABELS = {
    "voip": 0,
    "video": 1,
    "file": 2, #file_transfer
    "chat": 3,
    "browsing": 4
}

VPN_LABELS = {
    "reg": 0,
    "vpn": 1,
    "tor": 2
}


def import_array(input_array):
    print("Import dataset " + input_array)
    dataset = np.load(input_array)
    print(dataset.shape)
    return dataset


def export_dataset(dataset_dict, file_path):
    # with open(file_path + ".pkl", 'wb') as outfile:
    #     pickle.dump(dataset_list, outfile, pickle.HIGHEST_PROTOCOL)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    for name, array in dataset_dict.items():
        np.save(file_path + "_" + name, array)


def specific_vpn_type_ode_const_length_module(class_name, const_length, vpn_type="reg", mode="train", ratio=1.2):
    print("Running in specific_vpn_type_ode_const_length__module on " + class_name + " " + mode)

    class_array_file = [fn for fn in VPN_TYPES[vpn_type] if (class_name in fn and ("ode_const_" + str(const_length) + "_" + mode) in fn)][0]
    print(class_array_file)

    all_files = [fn for fn in VPN_TYPES[vpn_type] if (class_name not in fn and ("ode_const_" + str(const_length) + "_" + mode) in fn)]
    print(all_files)

    class_array = import_array(class_array_file)
    count = len(class_array)
    print(count)

    all_count = len(all_files)
    count_per_class = ratio*count/all_count
    print(count_per_class)

    for fn in all_files:
        print(fn)
        fn_array = import_array(fn)
        p = count_per_class*1.0/len(fn_array)
        print(p)
        if p < 1:
            mask = np.random.choice([True, False], len(fn_array), p=[p, 1-p])
            fn_array = fn_array[mask]

        print(len(fn_array))
        print(class_array.shape, fn_array.shape)
        class_array = np.append(class_array, fn_array, axis=0)
        print(len(class_array))
        del fn_array

    labels = np.append(np.zeros(count), np.ones(len(class_array) - count))
    print(len(class_array), len(labels), labels[0], labels[count-1], labels[count], labels[-1])

    return class_array, labels


def create_class_vs_all_specific_vpn_type_dataset_ode_const_length(class_name, const_length, vpn_type="reg", validation=False, ratio=1.2):
    class_array_train, labels_train = specific_vpn_type_ode_const_length_module(class_name, const_length, vpn_type=vpn_type, mode="train", ratio=ratio)
    class_array_test, labels_test = specific_vpn_type_ode_const_length_module(class_name, const_length, vpn_type=vpn_type, mode="test", ratio=ratio)

    dataset_dict = dict()

    if validation:
        print(len(labels_train), sum(labels_train), 1.0*sum(labels_train)/len(labels_train))
        print(len(labels_test), sum(labels_test), 1.0*sum(labels_test)/len(labels_test))

        dataset_dict["x_train"] = class_array_train
        dataset_dict["x_val"] = class_array_test
        dataset_dict["y_train"] = labels_train
        dataset_dict["y_val"] = labels_test
    else:
        dataset_dict["x_test"] = np.append(class_array_train, class_array_test, axis=0)
        dataset_dict["y_test"] = np.append(labels_train, labels_test, axis=0)
        print(len(dataset_dict["x_test"]), len(dataset_dict["y_test"]), sum(dataset_dict["y_test"]), 1.0 * len(dataset_dict["y_test"])/sum(dataset_dict["y_test"]))

    export_dataset(dataset_dict, DATASET_DIR + "ode_const_" + str(const_length) + "_" + class_name + "_vs_all_" + vpn_type + "/" + "ode_const_" + str(const_length) + "_" + class_name + "_vs_all_" + vpn_type)


def multiclass_specific_vpn_type_ode_const_length_module(const_length, vpn_type="reg", mode="train", ratio=1.2):
    print("Running in multiclass overlap module on " + mode)

    all_files = [fn for fn in VPN_TYPES[vpn_type] if ("ode_const_" + str(const_length) + "_" + mode) in fn]
    print(all_files)

    arrays_len = []
    for fn in all_files:
        arrays_len.append(len(import_array(fn)))

    count = min(arrays_len)
    print(count)

    count_per_class = ratio*count  #### NO ratio*count/all_count !!!
    print(count_per_class)

    labels = ()
    class_array = ()

    for fn in all_files:
        print(fn)
        fn_array = import_array(fn)
        p = count_per_class*1.0/len(fn_array)
        print(p)
        if p < 1:
            mask = np.random.choice([True, False], len(fn_array), p=[p, 1-p])
            fn_array = fn_array[mask]

        print(len(fn_array), str(os.path.basename(fn)).split("_")[0])
        class_label = CLASS_LABELS[str(os.path.basename(fn)).split("_")[0]]
        labels += (class_label * np.ones(len(fn_array)),)
        print(class_label, labels[-1][-1])
        class_array += (fn_array,)
        print(len(class_array), len(class_array[-1]))
        del fn_array

    labels = np.concatenate(labels, axis=0)
    class_array = np.concatenate(class_array, axis=0)
    print(len(class_array), len(labels), labels[0], labels[-1], set(labels))

    return class_array, labels


def create_multiclass_specific_vpn_type_dataset_ode_const_length(const_length, vpn_type="reg", validation=False, ratio=1.2):
    class_array_train, labels_train = multiclass_specific_vpn_type_ode_const_length_module(const_length, vpn_type=vpn_type, mode="train", ratio=ratio)
    class_array_test, labels_test = multiclass_specific_vpn_type_ode_const_length_module(const_length, vpn_type=vpn_type, mode="test", ratio=ratio)

    dataset_dict = dict()

    if validation:
        print(len(labels_train), sum(labels_train), 1.0*sum(labels_train)/len(labels_train))
        print(len(labels_test), sum(labels_test), 1.0*sum(labels_test)/len(labels_test))

        dataset_dict["x_train"] = class_array_train
        dataset_dict["x_val"] = class_array_test
        dataset_dict["y_train"] = labels_train
        dataset_dict["y_val"] = labels_test
    else:
        dataset_dict["x_test"] = np.append(class_array_train, class_array_test, axis=0)
        dataset_dict["y_test"] = np.append(labels_train, labels_test, axis=0)
        print(len(dataset_dict["x_test"]), len(dataset_dict["y_test"]), sum(dataset_dict["y_test"]), 1.0 * len(dataset_dict["y_test"])/sum(dataset_dict["y_test"]))

    export_dataset(dataset_dict, DATASET_DIR + "ode_const_" + str(const_length) + "_multiclass_" + vpn_type + "/" + "ode_const_" + str(const_length) + "_multiclass_" + vpn_type)



if __name__ == '__main__':
    CLASS = "voip"
    LENGTH = 10

    create_class_vs_all_specific_vpn_type_dataset_ode_const_length(CLASS, LENGTH, validation=True)


    # OLD
    # create_class_vs_all_specific_vpn_type_dataset_overlap(CLASS, validation=True)
    # create_class_vs_all_specific_vpn_type_dataset_overlap(CLASS, vpn_type="vpn", validation=False)
    # create_class_vs_all_specific_vpn_type_dataset_overlap(CLASS, vpn_type="tor", validation=False)
    # create_class_vs_all_merged_dataset_overlap(CLASS)

    # create_multiclass_specific_vpn_type_dataset_overlap(validation=True)
    # create_multiclass_specific_vpn_type_dataset_overlap(vpn_type="vpn", validation=False)
    # create_multiclass_specific_vpn_type_dataset_overlap(vpn_type="tor", validation=False)
    # create_multiclass_merged_dataset_overlap()
    # create_vpn_types_dataset_overlap()

    # create_class_vs_all_exclude_specific_applications_specific_vpn_type_dataset_overlap(CLASS, ['facebook'], validation=True)

    # create_class_vs_all_specific_vpn_type_dataset_overlap(CLASS, vpn_type="vpn", validation=True)
    # create_class_vs_all_specific_vpn_type_dataset_overlap(CLASS, vpn_type="tor", validation=True)

    # create_multiclass_specific_vpn_type_dataset_overlap(vpn_type="vpn", validation=True)
    # create_multiclass_specific_vpn_type_dataset_overlap(vpn_type="tor", validation=True)
    # TPS = 60
    # create_class_vs_all_specific_vpn_type_dataset_non_overlap("chat", validation=True, tps=TPS)
    # for category in CLASS_LABELS.keys():
    #     create_class_vs_all_specific_vpn_type_dataset_non_overlap(category, validation=True, tps=TPS)

        # if category != "browsing":
        #     create_class_vs_all_specific_vpn_type_dataset_non_overlap(category, vpn_type="vpn", validation=False)
        # create_class_vs_all_specific_vpn_type_dataset_non_overlap(category, vpn_type="tor", validation=False)
    #
    # create_vpn_types_specific_ICSX_dataset_overlap("tor")
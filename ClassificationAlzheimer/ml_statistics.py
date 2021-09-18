import numpy
import random
import pathlib
import re

import utilities
from utilities import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import naive_bayes, svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

X_train = []
X_test = []
Y_train = []
Y_test = []

X_train_dict = []
X_test_dict = []

P_all = []
R_all = []
F_all = []


def start_svm_algorithm(c: float, kernel: str, degree: int, gamma=str):
    # Fit the training set on SVM Classifier
    svm_classifier = svm.SVC(C=c, kernel=kernel, degree=degree, gamma=gamma)
    svm_classifier.fit(X_train, Y_train)

    # Predict on the test set
    predictions = svm_classifier.predict(X_test)

    # Print accuracy for SVM
    print("SVM Accuracy Score -> ", accuracy_score(Y_test, predictions) * 100)

    tn, fp, fn, tp = confusion_matrix(Y_test, predictions, labels=[0, 1]).ravel()

    print("TN " + str(tn) + " FP " + str(fp) + " FN " + str(fn) + " TP " + str(tp))

    P = 0
    R = 0
    F = 0
    if tp + fp != 0:
        P = tp / (tp + fp)
    else:
        x = 0
    if tp + fn != 0:
        R = tp / (tp + fn)
    else:
        x = 0

    if P + R != 0:
        F = (2 * P * R) / (P + R)
    else:
        x = 0

    print("P: " + str(P) + " R: " + str(R) + " F: " + str(F))
    P_all.append(P)
    R_all.append(R)
    F_all.append(F)


def encode_y_data():
    """
    Function that transforms Y arrays from arrays of words "Positive" and "Negative" to arrays of 1,0
    :return: None
    """
    global Y_train, Y_test

    label_encoder = LabelEncoder()
    Y_train = label_encoder.fit_transform(Y_train)
    Y_test = label_encoder.fit_transform(Y_test)


def clear_variables():
    """
    Function to clear variables for next start of the algorithm
    :return: None
    """

    global X_train, X_test, Y_train, Y_test, X_train_dict, stop_words, \
        X_test_dict, corpus

    if type(Y_train) is numpy.ndarray:
        Y_train = Y_train.tolist()
    if type(Y_test) is numpy.ndarray:
        Y_test = Y_test.tolist()

    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    X_test_dict = []
    X_train_dict = []

    utilities.PATH_TO_TAGGED_FILES = "../Tagged_Texts_2/"
    utilities.PATH_FOR_LOG_FILE = PATH_TO_TAGGED_FILES + "logs/"
    utilities.PATH_TO_STATISTIC_FOLDER = PATH_TO_TAGGED_FILES + "Statistic/"
    utilities.PATH_TO_TRAIN_TEST_CORPUS = PATH_TO_TAGGED_FILES + "Train_Test_Corpus/"
    utilities.PATH_TO_TRAIN_TEST_CORPUS_ML = PATH_TO_TAGGED_FILES + "Train_Test_Corpus_ML/"

    stop_words = []
    corpus = []


def extract_data(path_to_data_folder, corpus_type: CorpusType, classif_class: ClassificationClass):

    for path in pathlib.Path(path_to_data_folder).iterdir():
        if path.is_file():
            # Open next file in the folder
            file = open(path, encoding="utf8")
            if ".DS_Store" not in path.parts:
                lines = file.readlines()
                TTR = lines[7]
                W = lines[8]
                R = lines[9]
                N = lines[28]
                V = lines[29]
                ADJ = lines[30]
                ADV = lines[31]
                NV = lines[43]
                PV = lines[44]
                PN = lines[45]

                match_obj = re.match(REGEX_READ_FILE, TTR, re.M | re.I)
                if match_obj:
                    ttr_value = float(match_obj.group(1))

                match_obj = re.match(REGEX_READ_FILE, W, re.M | re.I)
                if match_obj:
                    w_value = float(match_obj.group(1))

                match_obj = re.match(REGEX_READ_FILE, R, re.M | re.I)
                if match_obj:
                    r_value = float(match_obj.group(1))

                match_obj = re.match(REGEX_READ_FILE, N, re.M | re.I)
                if match_obj:
                    n_value = float(match_obj.group(1))

                match_obj = re.match(REGEX_READ_FILE, V, re.M | re.I)
                if match_obj:
                    v_value = float(match_obj.group(1))

                match_obj = re.match(REGEX_READ_FILE, ADJ, re.M | re.I)
                if match_obj:
                    adj_value = float(match_obj.group(1))

                match_obj = re.match(REGEX_READ_FILE, ADV, re.M | re.I)
                if match_obj:
                    adv_value = float(match_obj.group(1))

                match_obj = re.match(REGEX_READ_FILE, NV, re.M | re.I)
                if match_obj:
                    nv_value = float(match_obj.group(1))

                match_obj = re.match(REGEX_READ_FILE, PV, re.M | re.I)
                if match_obj:
                    pv_value = float(match_obj.group(1))

                match_obj = re.match(REGEX_READ_FILE, PN, re.M | re.I)
                if match_obj:
                    pn_value = float(match_obj.group(1))

                array = [ttr_value, w_value, r_value, n_value, v_value, adv_value, adj_value, nv_value, pv_value, pn_value]

                if corpus_type == CorpusType.TRAIN:
                    if classif_class == ClassificationClass.POSITIVE:
                        X_train_dict.append((array, "Positive"))
                    elif classif_class == ClassificationClass.NEGATIVE:
                        X_train_dict.append((array, "Negative"))
                elif corpus_type == CorpusType.TEST:
                    if classif_class == ClassificationClass.POSITIVE:
                        X_test_dict.append((array, "Positive"))
                    elif classif_class == ClassificationClass.NEGATIVE:
                        X_test_dict.append((array, "Negative"))


def prepare_train_and_test_datasets(train_folder1, train_folder_1_class, train_folder2, train_folder_2_class,
                                    train_folder3, train_folder_3_class, train_folder4, train_folder_4_class,
                                    test_folder1, test_folder_1_class, test_folder2, test_folder_2_class):
    extract_data(PATH_TO_TRAIN_TEST_CORPUS + train_folder1, CorpusType.TRAIN, train_folder_1_class)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS + train_folder2, CorpusType.TRAIN, train_folder_2_class)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS + train_folder3, CorpusType.TRAIN, train_folder_3_class)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS + train_folder4, CorpusType.TRAIN, train_folder_4_class)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS + test_folder1, CorpusType.TEST, test_folder_1_class)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS + test_folder2, CorpusType.TEST, test_folder_2_class)

    # Shuffle X_train and X_test dictionaries
    random.shuffle(X_train_dict)
    random.shuffle(X_test_dict)

    for item in X_train_dict:
        X_train.append(item[0])
        Y_train.append(item[1])

    for item in X_test_dict:
        X_test.append(item[0])
        Y_test.append(item[1])


def clear_variables():
    """
    Function to clear variables for next start of the algorithm
    :return: None
    """

    global X_train, X_test, Y_train, Y_test, X_train_dict, X_test_dict

    if type(Y_train) is numpy.ndarray:
        Y_train = Y_train.tolist()
    if type(Y_test) is numpy.ndarray:
        Y_test = Y_test.tolist()

    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    X_test_dict = []
    X_train_dict = []

    utilities.PATH_TO_TAGGED_FILES = "../Tagged_Texts_2/"
    utilities.PATH_FOR_LOG_FILE = PATH_TO_TAGGED_FILES + "logs/"
    utilities.PATH_TO_STATISTIC_FOLDER = PATH_TO_TAGGED_FILES + "Statistic/"
    utilities.PATH_TO_TRAIN_TEST_CORPUS = PATH_TO_TAGGED_FILES + "Train_Test_Corpus/"


def do_machine_learning_statistics(train_folder1, train_folder_1_class, train_folder2, train_folder_2_class, train_folder3,
                        train_folder_3_class, train_folder4, train_folder_4_class, test_folder1, test_folder_1_class,
                        test_folder2, test_folder_2_class):
    """
    Function that starts machine learning algorithm.
    :param n_gram_type:
    :param test_folder_2_class: class of test folder2: POSITIVE or NEGATIVE
    :param test_folder_1_class: class of test folder1: POSITIVE or NEGATIVE
    :param train_folder_4_class: class of train folder4: POSITIVE or NEGATIVE
    :param train_folder_3_class: class of train folder3: POSITIVE or NEGATIVE
    :param train_folder_2_class: class of train folder2: POSITIVE or NEGATIVE
    :param train_folder_1_class: class of train folder1: POSITIVE or NEGATIVE
    :param train_folder1: Path to first train folder
    :param train_folder2: Path to second train folder
    :param train_folder3: Path to third train folder
    :param train_folder4: Path to fourth train folder
    :param test_folder1: Path to first test folder
    :param test_folder2: Path to second test folder
    :param n: number for n-grams
    :param algorithm: type MLAlgorithm - can be SVM or NaiveBayes
    :param use_lemma: True if lemma of the word should be used
    :param delete_stop_words: True if stop words should not be used
    :param delete_punct: True if PUNCT words should not be used
    :return: None
    """
    print("Start Machine Leaning Module")
    global Y_train, Y_test

    clear_variables()

    # Prepare train and test datasets
    prepare_train_and_test_datasets(train_folder1, train_folder_1_class, train_folder2, train_folder_2_class,
                                    train_folder3, train_folder_3_class, train_folder4, train_folder_4_class,
                                    test_folder1, test_folder_1_class, test_folder2, test_folder_2_class)

    encode_y_data()

    start_svm_algorithm(c=1.0, kernel='linear', degree=3, gamma='auto')

    print("P: " + str(average(P_all)) + " R: " + str(average(R_all)) + " F: " + str(average(R_all)))

    return ()
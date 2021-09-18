"""

Module for collecting data, diving into test and training data, word vectorization, starting SVM algorithm

"""
import numpy
import random
import pathlib
import re

import utilities
from utilities import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import naive_bayes, svm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

X_train = []
X_test = []
Y_train = []
Y_test = []
X_train_tfidf = []
X_test_tfidf = []

X_train_dict = []
X_test_dict = []

corpus = []
stop_words = []

P_all = 0
R_all = 0
F_all = 0


def start_naive_bayes_algorithm():
    """
    Function that starts Naive Bayes Classifier and prints accuracy
    :return: None
    """
    # Fit the training set on Naive Bayes Classifier
    naive_bayes_classifier = naive_bayes.MultinomialNB()
    naive_bayes_classifier.fit(X_train_tfidf, Y_train)

    # Predict on the test set
    predictions = naive_bayes_classifier.predict(X_test_tfidf)

    # Print accuracy for Naive Bayes Algorithm
    print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions, Y_test) * 100)

    precision, recall, fscore, support = precision_recall_fscore_support(Y_test, predictions, labels=[0, 1])
    print('precision:  {0}'.format(precision))
    print('recall:  {0}'.format(recall))
    print('fscore:  {0}'.format(fscore))

    tn, fp, fn, tp = confusion_matrix(Y_test, predictions, labels=[0, 1]).ravel()

    print("TN " + str(tn) + " FP " + str(fp) + " FN " + str(fn) + " TP " + str(tp))

    P = 0
    R = 0
    F = 0
    if tp + fp != 0:
        P = tp / (tp + fp)

    if tp + fn != 0:
        R = tp / (tp + fn)

    if P + R != 0:
        F = (2 * P * R) / (P + R)

    print("P: " + str(P) + " R: " + str(R) + " F: " + str(F))

    P_all = P
    R_all = R
    F_all = F


def start_svm_algorithm(c: float, kernel: str, degree: int, gamma=str):
    global P_all, R_all, F_all

    # Fit the training set on SVM Classifier
    svm_classifier = svm.SVC(C=c, kernel=kernel)
    svm_classifier.fit(X_train_tfidf, Y_train)

    # Predict on the test set
    predictions = svm_classifier.predict(X_test_tfidf)

    # Print accuracy for SVM
    print("SVM Accuracy Score -> ", accuracy_score(Y_test, predictions) * 100)

    tn, fp, fn, tp = confusion_matrix(Y_test, predictions, labels=[0, 1]).ravel()

    print("TN " + str(tn) + " FP " + str(fp) + " FN " + str(fn) + " TP " + str(tp))

    P = 0
    R = 0
    F = 0
    if tp + fp != 0:
        P = tp / (tp + fp)

    if tp + fn != 0:
        R = tp / (tp + fn)

    if P + R != 0:
        F = (2 * P * R) / (P + R)

    print("P: " + str(P) + " R: " + str(R) + " F: " + str(F))

    P_all = P
    R_all = R
    F_all = F


def word_vectorization(n, n_gram_type):
    """
    Function for feature extraction with tf-idf vectorizer.
    :param n_gram_type: word or character depending if n-gram chars or words should be used
    :param n: Number of n-grams
    n = (1,2) unigram model - bag of words
    n = (2,2) bigram model
    ...
    :return: None
    """

    global X_train_tfidf, X_test_tfidf

    if n_gram_type == "word":
        #tf_idf_vec = TfidfVectorizer(analyzer='word', use_idf=True, smooth_idf=False, ngram_range=n)
        tf_idf_vec = TfidfVectorizer(analyzer='word', ngram_range=n)
    elif n_gram_type == "character":
        tf_idf_vec = TfidfVectorizer(analyzer='char', use_idf=True, smooth_idf=False, ngram_range=n)
    else:
        print("N-gram type unknown")
        exit()

    tf_idf_vec.fit(X_train)

    X_train_tfidf = tf_idf_vec.transform(X_train)
    X_test_tfidf = tf_idf_vec.transform(X_test)


def encode_y_data():
    """
    Function that transforms Y arrays from arrays of words "Positive" and "Negative" to arrays of 1,0
    :return: None
    """
    global Y_train, Y_test

    label_encoder = LabelEncoder()
    Y_train = label_encoder.fit_transform(Y_train)
    Y_test = label_encoder.fit_transform(Y_test)


def extract_data(path_to_data_folder, corpus_type: CorpusType, classif_class: ClassificationClass, use_lemma,
                 delete_stop_words, delete_punct):
    """
    Function that extracts data from files and adds them to X_train, X_test, Y_train, Y_test.
    :param path_to_data_folder: path of the folder to be read
    :param corpus_type: type CorpusType - TRAIN or TEST
    :param classif_class: type ClassificationClass - POSITIVE or NEGATIVE
    :param use_lemma: True if lemma of the word should be used
    :param delete_stop_words: True if stop words should not be used
    :param delete_punct: True if PUNCT words should not be used
    :return: None
    """
    global Y_train, Y_test
    paths_processed = 0

    for path in pathlib.Path(path_to_data_folder).iterdir():
        if path.is_file():
            # Open next file in the folder
            file = open(path, mode="r+", encoding="utf8")

            if ".DS_Store" not in path.parts:

                paths_processed += 1
                list_of_words = ""

                lines = file.readlines()
                for line in lines:
                    match_obj = re.match(REGEX, line, re.M | re.I)
                    if match_obj:

                        not_add_word = False

                        if use_lemma is True:
                            word = str(match_obj.group(3))
                        else:
                            word = str(match_obj.group(1))

                        # If delete punct is True, all words with tag PUNCT should not be considered
                        if delete_punct is True:
                            if match_obj.group(2) == "PUNCT":
                                not_add_word = True

                        if delete_stop_words is True:
                            if word in stop_words:
                                not_add_word = True

                        if not not_add_word:
                            list_of_words += " "
                            list_of_words += word

                corpus.append(list_of_words)
                if corpus_type == CorpusType.TRAIN:
                    if classif_class == ClassificationClass.POSITIVE:
                        X_train_dict.append((list_of_words, "Positive"))
                    elif classif_class == ClassificationClass.NEGATIVE:
                        X_train_dict.append((list_of_words, "Negative"))
                elif corpus_type == CorpusType.TEST:
                    if classif_class == ClassificationClass.POSITIVE:
                        X_test_dict.append((list_of_words, "Positive"))
                    elif classif_class == ClassificationClass.NEGATIVE:
                        X_test_dict.append((list_of_words, "Negative"))


def prepare_train_and_test_datasets(train_folder1, train_folder_1_class, train_folder2, train_folder_2_class,
                                    train_folder3, train_folder_3_class, train_folder4, train_folder_4_class,
                                    test_folder1, test_folder_1_class, test_folder2, test_folder_2_class, use_lemma,
                                    delete_stop_words, delete_punct):
    """
    Function to prepare X_train, X_test, Y_train, Y_test. X_train is array of all files from all train folders.
    X_test is array of all files from test folders. X_train contains string "Postive" or "Negative" for every file in
    X_train. Y_test contains string "Postive" or "Negative" for every file in X_test.
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
    :param use_lemma: True if lemma of the word should be used
    :param delete_stop_words: True if stop words should not be used
    :param delete_punct: True if PUNCT words should not be used
    :return: None
    """
    # Extract data from
    extract_data(PATH_TO_TRAIN_TEST_CORPUS_ML + train_folder1, CorpusType.TRAIN, train_folder_1_class,
                 use_lemma, delete_stop_words, delete_punct)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS_ML + train_folder2, CorpusType.TRAIN, train_folder_2_class,
                 use_lemma, delete_stop_words, delete_punct)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS_ML + train_folder3, CorpusType.TRAIN, train_folder_3_class,
                 use_lemma, delete_stop_words, delete_punct)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS_ML + train_folder4, CorpusType.TRAIN, train_folder_4_class,
                 use_lemma, delete_stop_words, delete_punct)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS_ML + test_folder1, CorpusType.TEST, test_folder_1_class,
                 use_lemma, delete_stop_words, delete_punct)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS_ML + test_folder2, CorpusType.TEST, test_folder_2_class,
                 use_lemma, delete_stop_words, delete_punct)

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

    global X_train, X_test, X_test_tfidf, X_train_tfidf, Y_train, Y_test, X_train_dict, X_test_tfidf, stop_words, \
        X_test_dict, corpus

    # Convert np arrays to lists
    if type(Y_train) is numpy.ndarray:
        Y_train = Y_train.tolist()
    if type(Y_test) is numpy.ndarray:
        Y_test = Y_test.tolist()

    X_train = []
    X_test = []
    X_test_tfidf = []
    X_train_tfidf = []
    Y_train = []
    Y_test = []
    X_test_dict = []
    X_train_dict = []

    # Set paths for the execution
    utilities.PATH_TO_TAGGED_FILES = "../Tagged_Texts_2/"
    utilities.PATH_FOR_LOG_FILE = PATH_TO_TAGGED_FILES + "logs/"
    utilities.PATH_TO_STATISTIC_FOLDER = PATH_TO_TAGGED_FILES + "Statistic/"
    utilities.PATH_TO_TRAIN_TEST_CORPUS = PATH_TO_TAGGED_FILES + "Train_Test_Corpus/"
    utilities.PATH_TO_TRAIN_TEST_CORPUS_ML = PATH_TO_TAGGED_FILES + "Train_Test_Corpus_ML/"

    stop_words = []
    corpus = []


def collect_stop_words():
    """
    Function to collect stop words from two files and save them to global variable stop_words
    :return: None
    """

    stop_words_1 = open(STOP_WORDS_1, mode="r+", encoding="utf8")
    stop_words_2 = open(STOP_WORDS_2, mode="r+", encoding="utf8")

    lines = stop_words_1.readlines()
    for line in lines:
        stop_words.append(line.strip())

    lines = stop_words_2.readlines()
    for line in lines:
        stop_words.append(line.strip())


def do_machine_learning(train_folder1, train_folder_1_class, train_folder2, train_folder_2_class, train_folder3,
                        train_folder_3_class, train_folder4, train_folder_4_class, test_folder1, test_folder_1_class,
                        test_folder2, test_folder_2_class, n, n_gram_type: str, algorithm: MLAlgorithm, use_lemma,
                        delete_stop_words, delete_punct):
    """
    Function that starts machine learning algorithm.
    :param n_gram_type: words or characters
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
    :param n: typle representing number for n-grams
    :param algorithm: type MLAlgorithm - can be SVM or NaiveBayes
    :param use_lemma: True if lemma of the word should be used
    :param delete_stop_words: True if stop words should not be used
    :param delete_punct: True if PUNCT words should not be used
    :return: average values for precision, recall and f-measure
    """
    print("Start Machine Leaning Module")
    global Y_train, Y_test

    # Clear data from previous execution that are remaining in the variables
    clear_variables()

    # Collect stop words in case they need to be deleted
    if delete_stop_words:
        collect_stop_words()

    # Prepare train and test datasets
    prepare_train_and_test_datasets(train_folder1, train_folder_1_class, train_folder2, train_folder_2_class,
                                    train_folder3, train_folder_3_class, train_folder4, train_folder_4_class,
                                    test_folder1, test_folder_1_class, test_folder2, test_folder_2_class,
                                    use_lemma, delete_stop_words, delete_punct)

    # Encode y part of datasets - from words to 0 and 1
    encode_y_data()

    # Perform TF-IFD on x part of datasets
    word_vectorization(n, n_gram_type)

    # Start algorithm
    if algorithm == MLAlgorithm.SVM:
        start_svm_algorithm(c=1.0, kernel='linear', degree=3, gamma='auto')
    elif algorithm == MLAlgorithm.NaiveBayes:
        start_naive_bayes_algorithm()
    else:
        print("No algorithm specified")
        exit()

    # Return average values for precision, recall and f-measure
    return (P_all, R_all, F_all)
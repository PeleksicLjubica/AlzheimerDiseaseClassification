"""

Module for word embedding and SVM learning

"""
import numpy
import numpy as np

from utilities import *
import pathlib
import re
import pandas as pd
from matplotlib.cbook import flatten
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import naive_bayes, svm
from sklearn.metrics import accuracy_score

X_train = []
X_test = []
Y_train = []
Y_test = []
X_train_tfidf = []
X_test_tfidf = []

corpus = []


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


def start_svm_algorithm(c: float, kernel: str, degree: int, gamma=str):
    # Fit the training set on SVM Classifier
    svm_classifier = svm.SVC(C=c, kernel=kernel, degree=degree, gamma=gamma)
    svm_classifier.fit(X_train_tfidf, Y_train)

    # Predict on the test set
    predictions = svm_classifier.predict(X_test_tfidf)

    # Print accuracy for SVM
    print("SVM Accuracy Score -> ", accuracy_score(predictions, Y_test) * 100)


def word_vectorization(n):
    """
    Function for feature extraction with tf-idf vectorizer.
    :param n: Number of n-grams
    n = (1,2) unigram model - bag of words
    n = (2,2) bigram model
    ...
    :return: None
    """

    global X_train_tfidf, X_test_tfidf

    tf_idf_vec = TfidfVectorizer(use_idf=True, smooth_idf=False, ngram_range=n)
    tf_idf_vec.fit(corpus)

    X_train_tfidf = tf_idf_vec.transform(X_train)
    X_test_tfidf = tf_idf_vec.transform(X_test)


def encode_y_data():
    global Y_train, Y_test

    label_encoder = LabelEncoder()
    Y_train = label_encoder.fit_transform(Y_train)
    Y_test = label_encoder.fit_transform(Y_test)


def extract_data(path_to_data_folder, corpus_type: CorpusType, classif_class: ClassificationClass):

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
                        list_of_words += " "
                        list_of_words += str(match_obj.group(1))
                corpus.append(list_of_words)
                if corpus_type == CorpusType.TRAIN:
                    X_train.append(list_of_words)
                elif corpus_type == CorpusType.TEST:
                    X_test.append(list_of_words)

    if corpus_type == CorpusType.TRAIN:
        if classif_class == ClassificationClass.POSITIVE:
            tmp = ["Positive"] * paths_processed
        elif classif_class == ClassificationClass.NEGATIVE:
            tmp = ["Negative"] * paths_processed
        Y_train.append(tmp)
        Y_train = list(flatten(Y_train))
    elif corpus_type == CorpusType.TEST:
        if classif_class == ClassificationClass.POSITIVE:
            tmp = ["Positive"] * paths_processed
        elif classif_class == ClassificationClass.NEGATIVE:
            tmp = ["Negative"] * paths_processed
        Y_test.append(tmp)
        Y_test = list(flatten(Y_test))


def prepare_train_and_test_datasets(train_folder1, train_folder2, train_folder3, train_folder4, test_folder1,
                                    test_folder2):
    # Prepare Train and test data sets
    extract_data(PATH_TO_TRAIN_TEST_CORPUS_ML + train_folder1, CorpusType.TRAIN, ClassificationClass.POSITIVE)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS_ML + train_folder2, CorpusType.TRAIN, ClassificationClass.POSITIVE)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS_ML + train_folder3, CorpusType.TRAIN, ClassificationClass.NEGATIVE)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS_ML + train_folder4, CorpusType.TRAIN, ClassificationClass.NEGATIVE)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS_ML + test_folder1, CorpusType.TEST, ClassificationClass.POSITIVE)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS_ML + test_folder2, CorpusType.TEST, ClassificationClass.NEGATIVE)


def clear_variables():

    global X_train, X_test, X_test_tfidf, X_train_tfidf, Y_train, Y_test

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


def do_machine_learning(train_folder1, train_folder2, train_folder3, train_folder4, test_folder1, test_folder2, n,
                        algorithm: MLAlgorithm):

    print("Start Machine Leaning Module")
    global Y_train, Y_test

    clear_variables()

    # Prepare train and test datasets
    prepare_train_and_test_datasets(train_folder1, train_folder2, train_folder3, train_folder4, test_folder1,
                                    test_folder2)

    encode_y_data()

    word_vectorization(n)

    if algorithm == MLAlgorithm.SVM:
        start_svm_algorithm(c=1.0, kernel='linear', degree=3, gamma='auto')
    elif algorithm == MLAlgorithm.NaiveBayes:
        start_naive_bayes_algorithm()
    else:
        print("No algorithm specified")
        exit()

    print("End Machine Leaning Module")

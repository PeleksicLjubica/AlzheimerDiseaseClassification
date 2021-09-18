"""
main.py
Main module used to start all other modules
"""

from ml_statistics import do_machine_learning_statistics
from utilities import *
from machine_learning import do_machine_learning
from linguistic_analysis import start_lexical_analysis


def start_machine_learning(n, n_gram_type, use_lemma, delete_stop_words, delete_punct):
    """
    Function that starts machine learning algorithm inside machine_learning module
    :param n_gram_type: words or characters
    :param n: tuple used to define n for n-grams
    :param use_lemma: True if lemma of the word should be used
    :param delete_stop_words: True if stop words should be deleted
    :param delete_punct: True if punctuation should be deleted
    :return: None
    """

    # Start machine learning module for SVM algorithm
    print("TRAIN: FOLDER_1, FOLDER_3 TEST: FOLDER_2")
    p1, r1, f1, = do_machine_learning(POSITIVE_FOLDER_1, ClassificationClass.POSITIVE, POSITIVE_FOLDER_3,
                                      ClassificationClass.POSITIVE, NEGATIVE_FOLDER_1, ClassificationClass.NEGATIVE,
                                      NEGATIVE_FOLDER_3, ClassificationClass.NEGATIVE, POSITIVE_FOLDER_2,
                                      ClassificationClass.POSITIVE, NEGATIVE_FOLDER_2, ClassificationClass.NEGATIVE, n,
                                      n_gram_type,
                                      MLAlgorithm.SVM, use_lemma, delete_stop_words, delete_punct)

    print("TRAIN: FOLDER_1, FOLDER_2 TEST: FOLDER_3")
    p2, r2, f2, = do_machine_learning(POSITIVE_FOLDER_1, ClassificationClass.POSITIVE, POSITIVE_FOLDER_2,
                                      ClassificationClass.POSITIVE, NEGATIVE_FOLDER_1, ClassificationClass.NEGATIVE,
                                      NEGATIVE_FOLDER_2, ClassificationClass.NEGATIVE, POSITIVE_FOLDER_3,
                                      ClassificationClass.POSITIVE, NEGATIVE_FOLDER_3, ClassificationClass.NEGATIVE, n,
                                      n_gram_type,
                                      MLAlgorithm.SVM, use_lemma, delete_stop_words, delete_punct)
    print("TRAIN: FOLDER_2, FOLDER_3 TEST: FOLDER_1")
    p3, r3, f3, = do_machine_learning(POSITIVE_FOLDER_2, ClassificationClass.POSITIVE, POSITIVE_FOLDER_3,
                                      ClassificationClass.POSITIVE, NEGATIVE_FOLDER_2, ClassificationClass.NEGATIVE,
                                      NEGATIVE_FOLDER_3, ClassificationClass.NEGATIVE, POSITIVE_FOLDER_1,
                                      ClassificationClass.POSITIVE, NEGATIVE_FOLDER_1, ClassificationClass.NEGATIVE, n,
                                      n_gram_type,
                                      MLAlgorithm.SVM, use_lemma, delete_stop_words, delete_punct)

    print("Average P: " + str((p1 + p2 + p3) / 3) + " Average R: " + str((r1 + r2 + r3) / 3) + " Average F: " +
          str((f1 + f2 + f3) / 3))


def start_machine_learning_for_linguistic():
    """
    Start machine learning algorithm with linguistic analysis results for all test sets
    :return: None
    """
    print("TRAIN: FOLDER_1, FOLDER_3 TEST: FOLDER_2")
    do_machine_learning_statistics(POSITIVE_FOLDER_1, ClassificationClass.POSITIVE, POSITIVE_FOLDER_3,
                                   ClassificationClass.POSITIVE, NEGATIVE_FOLDER_1, ClassificationClass.NEGATIVE,
                                   NEGATIVE_FOLDER_3, ClassificationClass.NEGATIVE, POSITIVE_FOLDER_2,
                                   ClassificationClass.POSITIVE, NEGATIVE_FOLDER_2, ClassificationClass.NEGATIVE)

    print("TRAIN: FOLDER_1, FOLDER_2 TEST: FOLDER_3")
    do_machine_learning_statistics(POSITIVE_FOLDER_1, ClassificationClass.POSITIVE, POSITIVE_FOLDER_2,
                                   ClassificationClass.POSITIVE, NEGATIVE_FOLDER_1, ClassificationClass.NEGATIVE,
                                   NEGATIVE_FOLDER_2, ClassificationClass.NEGATIVE, POSITIVE_FOLDER_3,
                                   ClassificationClass.POSITIVE, NEGATIVE_FOLDER_3, ClassificationClass.NEGATIVE)

    print("TRAIN: FOLDER_2, FOLDER_3 TEST: FOLDER_1")
    do_machine_learning_statistics(POSITIVE_FOLDER_2, ClassificationClass.POSITIVE, POSITIVE_FOLDER_3,
                                   ClassificationClass.POSITIVE, NEGATIVE_FOLDER_2, ClassificationClass.NEGATIVE,
                                   NEGATIVE_FOLDER_3, ClassificationClass.NEGATIVE, POSITIVE_FOLDER_1,
                                   ClassificationClass.POSITIVE, NEGATIVE_FOLDER_1, ClassificationClass.NEGATIVE)


if __name__ == '__main__':
    """
    Main function used to start different modules - Linguistic analysis, ML module and ML with linguistic analysis 
    results
    """

    print(" Start Classification of Alzheimer Disease ")

    print(" Start Linguistic Analysis ")
    # Read tagged files and create files containing measures for every file
    start_lexical_analysis(draw_graphs=False, statistics=False, should_read_tagged_files=True)
    # Read files containing measures and calculate statistics and draw graphs
    start_lexical_analysis(draw_graphs=True, statistics=True, should_read_tagged_files=False)

    # Start machine learning algorithm for bag of words, character ngrams for n = 1..9
    print("Bag of words")
    start_machine_learning(n=(1, 1), n_gram_type="word", use_lemma=False, delete_stop_words=False,
                           delete_punct=False)
    start_machine_learning(n=(1, 1), n_gram_type="word", use_lemma=True, delete_stop_words=False,
                           delete_punct=False)
    start_machine_learning(n=(1, 1), n_gram_type="word", use_lemma=True, delete_stop_words=True,
                           delete_punct=False)
    start_machine_learning(n=(1, 1), n_gram_type="word", use_lemma=True, delete_stop_words=True,
                           delete_punct=True)

    print("Character n-grams")
    start_machine_learning(n=(1, 1), n_gram_type="character", use_lemma=False, delete_stop_words=False,
                           delete_punct=False)
    start_machine_learning(n=(2, 2), n_gram_type="character", use_lemma=False, delete_stop_words=False,
                           delete_punct=False)
    start_machine_learning(n=(3, 3), n_gram_type="character", use_lemma=False, delete_stop_words=False,
                           delete_punct=False)
    start_machine_learning(n=(4, 4), n_gram_type="character", use_lemma=False, delete_stop_words=False,
                           delete_punct=False)
    start_machine_learning(n=(5, 5), n_gram_type="character", use_lemma=False, delete_stop_words=False,
                           delete_punct=False)
    start_machine_learning(n=(6, 6), n_gram_type="character", use_lemma=False, delete_stop_words=False,
                           delete_punct=False)
    start_machine_learning(n=(7, 7), n_gram_type="character", use_lemma=False, delete_stop_words=False,
                           delete_punct=False)
    start_machine_learning(n=(8, 8), n_gram_type="character", use_lemma=False, delete_stop_words=False,
                           delete_punct=False)
    start_machine_learning(n=(9, 9), n_gram_type="character", use_lemma=False, delete_stop_words=False,
                           delete_punct=False)

    # Start hybrid approach - SVM algorithm with linguistic analysis measures
    print("Hybrid approach")
    start_machine_learning_for_linguistic()

    print(" End Classification of Alzheimer Disease ")

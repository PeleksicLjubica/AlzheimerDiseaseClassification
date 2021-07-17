"""

Module for word embedding and SVM learning

"""

from utilities import *
import pathlib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

X_train = []
X_test = []
Y_train = []
Y_test = []


def start_word_embedding():
    print("Start Word Embedding")

    # Prepare Train and test data sets
    extract_data(PATH_TO_TRAIN_TEST_CORPUS + TRAIN_FOLDER_1, Corpus_Type.TRAIN, Clasiffication_Class.POSITIVE)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS + TRAIN_FOLDER_2, Corpus_Type.TRAIN , Clasiffication_Class.POSITIVE)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS + TRAIN_FOLDER_3, Corpus_Type.TRAIN, Clasiffication_Class.NEGATIVE)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS + TRAIN_FOLDER_4, Corpus_Type.TRAIN, Clasiffication_Class.NEGATIVE)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS + TEST_FOLDER_1, Corpus_Type.TEST, Clasiffication_Class.POSITIVE)
    extract_data(PATH_TO_TRAIN_TEST_CORPUS + TEST_FOLDER_2, Corpus_Type.TEST, Clasiffication_Class.NEGATIVE)


def extract_data(path_to_data_folder, ):
    print("Start Bag of words")

    for path in pathlib.Path(PATH_TO_TAGGED_FILES).iterdir():
        if path.is_file():
            # Open next file in the folder
            file = open(path, mode="r+", encoding="utf8")

            if ".DS_Store" in path.parts:
                break

            lines = file.read()


def generate_n_grams(train, test, n):
    """
    Function for feature extraction with tf-idf vectorizer
    Train and test are data for train and test.
    n = (1,1) -> function creates unigram model (bag-of-words)
    n = (2,2) -> function creates bigram model
    Return value -> arrays of n-grams for train and test
    """
    tf_idf_vec = TfidfVectorizer(use_idf=True, smooth_idf=False, ngram_range=n)
    tf_idf_vec.fit(train)

    train_vector = tf_idf_vec.transform(train)
    test_vector = tf_idf_vec.transform(test)

    #tf_idf_dataframe = pd.DataFrame(tf_idf_data.toarray(), columns=tf_idf_vec.get_feature_names())

    return train_vector.toarray(), test_vector.toarray()

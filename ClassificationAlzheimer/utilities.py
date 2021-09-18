"""

File containing constants and functions that are used from different modules.

"""

from enum import Enum

PATH_TO_TAGGED_FILES = "../Tagged_Texts_2/"
PATH_FOR_LOG_FILE = "../Tagged_Texts_2/logs/"
PATH_TO_TEXTS = "../Texts/"  # used for cosine distance counting

REGEX = r'([0-9a-zA-Z{_`’<–”“„\/\(\)\[\]\-\'":*!.?…},šđčćžŠĐÐČĆŽ]*)	([0-9a-zA-Z{_`<–’„”“\/\(\)\[\]\-\'":*!.?…},' \
        r'šđčćžŠÐĐČĆŽ]*)	([0-9a-zA-Z{_`<–’„”“\/\(\)\[\]\-\'":*!.?…},šđčćžŠĐÐČĆŽ]*)'
REGEX_READ_FILE = r'[a-zA-Z\'_ ]*:?: ([0-9.]+)'
STATISTICS_REGEX = r"[a-zA-Z '/]*: Precision: ([0-9.]*) Recall: ([0-9.]*) F-measure: ([0-9.]*)"

BOW_PREPROCESSING_LEMMA: bool = False
BOW_PREPROCESSING_STOP_WORDS: bool = False

PATH_TO_STATISTIC_FOLDER = PATH_TO_TAGGED_FILES + "Statistic/"
PATH_TO_TRAIN_TEST_CORPUS = PATH_TO_TAGGED_FILES + "Train_Test_Corpus/"
PATH_TO_TRAIN_TEST_CORPUS_ML = PATH_TO_TAGGED_FILES + "Train_Test_Corpus_ML/"
POSITIVE_FOLDER_1 = "P_1"
POSITIVE_FOLDER_2 = "P_2"
NEGATIVE_FOLDER_1 = "N_1"
NEGATIVE_FOLDER_2 = "N_2"
POSITIVE_FOLDER_3 = "P_3"
NEGATIVE_FOLDER_3 = "N_3"

STOP_WORDS_1 = "../StopWords/stopwordsSRB1.txt"
STOP_WORDS_2 = "../StopWords/stopwordsSRB2.txt"

class ClassificationClass(Enum):
    NONE = 0
    POSITIVE = 1
    NEGATIVE = 2


class CorpusType(Enum):
    NONE = 0
    TRAIN = 1
    TEST = 2


class Measurement(Enum):
    TTR = 7
    W = 8
    R = 9
    nouns = 28
    verbs = 29
    adj = 30
    adv = 31
    noun_verb = 43
    pronoun_verb = 44
    pronoun_noun = 45


class MLAlgorithm(Enum):
    SVM = 1,
    NaiveBayes = 2

def average(lst):
    return sum(lst) / len(lst)

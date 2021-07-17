"""

File containing constants and functions that are used from different modules.

"""

from enum import Enum

PATH_TO_TAGGED_FILES = "../TaggedTexts"
PATH_FOR_LOG_FILE = "../TaggedTexts/logs"
PATH_TO_TEXTS = "../"  # used for cosine distance counting

REGEX = r'([0-9a-zA-Z{_`’<–”“„\/\(\)\[\]\-\'":*!.?…},šđčćžŠĐÐČĆŽ]*)	([0-9a-zA-Z{_`<–’„”“\/\(\)\[\]\-\'":*!.?…},' \
        r'šđčćžŠÐĐČĆŽ]*)	([0-9a-zA-Z{_`<–’„”“\/\(\)\[\]\-\'":*!.?…},šđčćžŠĐÐČĆŽ]*)'
REGEX_READ_FILE = r'[a-zA-Z\'_ ]*:?: ([0-9.]+)'

BOW_PREPROCESSING_LEMMA: bool = False
BOW_PREPROCESSING_STOP_WORDS: bool = False

PATH_TO_STATISTIC_FOLDER = "../Statistic"
PATH_TO_TRAIN_TEST_CORPUS = "../Train_Test_Corpus/"
TRAIN_FOLDER_1 = "P_1"
TRAIN_FOLDER_2 = "P_2"
TRAIN_FOLDER_3 = "N_1"
TRAIN_FOLDER_4 = "N_2"
TEST_FOLDER_1 = "P_3"
TEST_FOLDER_2 = "N_3"


class Clasiffication_Class(Enum):
    NONE = 0
    POSITIVE = 1
    NEGATIVE = 2


class Corpus_Type(Enum):
    NONE = 0
    TRAIN = 1
    TEST = 2


class Measurement(Enum):
    TTR = 7
    W = 8
    R = 9
    nouns = 29
    verbs = 29
    adj = 30
    adv = 31
    noun_verb = 43
    pronoun_verb = 44
    pronoun_noun = 45

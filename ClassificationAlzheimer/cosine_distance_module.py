'''



'''

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from utilities import *


def calculate_cosine_distance():

    count_vectorizer = CountVectorizer()

    file_to_be_tested = PATH_TO_TEXTS + "test_file.txt"

    for path in pathlib.Path(PATH_TO_TAGGED_FILES + "/logs/" + path_to_tagged).iterdir():
        if path.is_file():
            # Open next file in the folder
            file = open(path, mode="r", encoding="utf8")
            lines = file.readlines()

            for line in lines:

                # Match regex against the line
                match_obj = re.match(REGEX_READ_FILE, line, re.M | re.I)
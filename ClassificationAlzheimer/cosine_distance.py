"""

Module for calculating the cosine distance

"""
import pandas as pd
import pathlib
import os
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from utilities import *
from sklearn.metrics.pairwise import cosine_similarity


documents = []
document_index = []

def read_files(path_to_folder):

    for path in pathlib.Path(path_to_folder).iterdir():
        if path.is_file():
            # Open next file in the folder
            print(path)
            if ".DS_Store" in path:
                break
            file = open(path, mode="r+", encoding="utf8")
            read_file = file.read()
            documents.append(read_file)

            # Find patient name to make log file with data for that patient
            head, tail = os.path.split(path)
            patient_name = tail.split(".")[0]
            document_index.append(patient_name)


def calculate_cosine_distance():

    read_files(PATH_TO_TEXTS + "P/")
    read_files(PATH_TO_TEXTS + "N/")

    tfidf_vectorizer = TfidfVectorizer()
    sparse_matrix = tfidf_vectorizer.fit_transform(documents)

    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix,
                      columns=tfidf_vectorizer.get_feature_names(),
                      index=document_index)

    with numpy.printoptions(threshold=numpy.inf):
        print(df)

        print(cosine_similarity(df, df))

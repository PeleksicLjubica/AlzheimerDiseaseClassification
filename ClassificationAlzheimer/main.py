'''
main.py
Script used to start framework
'''

import pathlib
import re
import os
import math  # needed for logarithm

import numpy as np

from utilities import *
from cosine_distance import calculate_cosine_distance
from plotting import plot_lexical_analysis_results_one_plot, plot_lexical_analysis_results_two_plots
from machine_learning import do_machine_learning
from statistics import calculate_statistics

GLOBAL_NUMBER_OF_WORDS_P = []
GLOBAL_NUMBER_OF_WORDS_N = []

GLOBAL_AVERAGE_WORDS_LEN_P = []
GLOBAL_AVERAGE_WORDS_LEN_N = []

GLOBAL_VOCABULARY_P = []
GLOBAL_VOCABULARY_N = []

GLOBAL_VOCABULARY_SO_P = []
GLOBAL_VOCABULARY_SO_N = []

GLOBAL_TTR_P = []
GLOBAL_TTR_N = []

GLOBAL_W_P = []
GLOBAL_W_N = []

GLOBAL_R_P = []
GLOBAL_R_N = []

GLOBAL_NUMBER_OF_NOUNS_P = []
GLOBAL_NUMBER_OF_NOUNS_N = []
GLOBAL_NUMBER_OF_NOUNS_NOR_P = []
GLOBAL_NUMBER_OF_NOUNS_NOR_N = []

GLOBAL_NUMBER_OF_VERBS_P = []
GLOBAL_NUMBER_OF_VERBS_N = []
GLOBAL_NUMBER_OF_VERBS_NOR_P = []
GLOBAL_NUMBER_OF_VERBS_NOR_N = []

GLOBAL_NUMBER_OF_ADJECTIVES_P = []
GLOBAL_NUMBER_OF_ADJECTIVES_N = []
GLOBAL_NUMBER_OF_ADJECTIVES_NOR_P = []
GLOBAL_NUMBER_OF_ADJECTIVES_NOR_N = []

GLOBAL_NUMBER_OF_ADVERBS_P = []
GLOBAL_NUMBER_OF_ADVERBS_N = []
GLOBAL_NUMBER_OF_ADVERBS_NOR_P = []
GLOBAL_NUMBER_OF_ADVERBS_NOR_N = []

GLOBAL_NUMBER_OF_CCONJ_P = []
GLOBAL_NUMBER_OF_CCONJ_N = []
GLOBAL_NUMBER_OF_CCONJ_NOR_P = []
GLOBAL_NUMBER_OF_CCONJ_NOR_N = []

GLOBAL_NUMBER_OF_PART_P = []
GLOBAL_NUMBER_OF_PART_N = []
GLOBAL_NUMBER_OF_PART_NOR_P = []
GLOBAL_NUMBER_OF_PART_NOR_N = []

GLOBAL_NUMBER_OF_SCONJ_P = []
GLOBAL_NUMBER_OF_SCONJ_N = []
GLOBAL_NUMBER_OF_SCONJ_NOR_P = []
GLOBAL_NUMBER_OF_SCONJ_NOR_N = []

GLOBAL_NUMBER_OF_PRONN_P = []
GLOBAL_NUMBER_OF_PRONN_N = []
GLOBAL_NUMBER_OF_PRONN_NOR_P = []
GLOBAL_NUMBER_OF_PRONN_NOR_N = []

GLOBAL_NOUN_VERB_P = []
GLOBAL_PRONOUN_VERB_P = []
GLOBAL_PRONOUN_NOUN_P = []
GLOBAL_NOUN_VERB_N = []
GLOBAL_PRONOUN_VERB_N = []
GLOBAL_PRONOUN_NOUN_N = []


def read_tagged_files(path_log_file, path_to_tagged):
    """
    Function to read person_name.tt file, extract statistics and write them to person_name.txt file
    These statistics are later used to draw graphs
    """
    log_file = open(path_log_file, mode="w+", encoding="utf8")

    for path in pathlib.Path(PATH_TO_TAGGED_FILES + "/" + path_to_tagged).iterdir():
        if path.is_file():
            # Open next file in the folder
            file = open(path, mode="r+", encoding="utf8")

            if ".DS_Store" not in path.parts:
                # Find patient name to make log file with data for that patient
                head, tail = os.path.split(path)
                patient_name = tail.split(".")[0]
                patient_log_file = open(PATH_FOR_LOG_FILE + '/' + path_to_tagged + "/" + patient_name + '.txt',
                                        mode="w",
                                        encoding="utf8")
                patient_log_file.write(patient_name + "\n")
                patient_log_file.write("------------------------------------------------\n")

                print("------------------------------------------------")
                print(path)

                log_file.write("------------------------------------------------\n")
                log_file.write(str(path) + "\n")

                lines = file.readlines()

                '''
                Variables containing  all information about one patient
                '''
                # Variable containing number of lines in the file
                count = 0

                # Variables containing all POS word counts
                number_of_nouns = 0
                number_of_verbs = 0
                number_of_adjectives = 0
                number_of_adverbs = 0
                number_of_punct = 0
                number_of_aux = 0
                number_of_det = 0
                number_of_part = 0
                number_of_cconj = 0
                number_of_sconj = 0
                number_of_propn = 0
                number_of_pron = 0
                number_of_x = 0
                number_of_adp = 0
                number_of_num = 0
                number_of_intj = 0

                # Variable containing number of words that appear in the transcript
                number_of_words = 0
                # Variable containing all different words that appear in the transcript - used to count vocabulary
                vocabulary_list = {}
                # Variable containing number of different words spoken
                vocabulary = 0
                # Variable containing number of words spoken only once
                vocabulary_spoken_once = 0
                words_length = 0

                # Explore every line
                for line in lines:
                    count += 1

                    # Match regex against the line
                    match_obj = re.match(REGEX, line, re.M | re.I)
                    if match_obj:

                        words_length += len(str(match_obj.group(1)))

                        # If type of object is not PUNCT - punctuation marks, word should be included in vocabulary
                        if str(match_obj.group(2)) != "PUNCT":
                            word = str(match_obj.group(3)).lower()
                            if word not in vocabulary_list:
                                vocabulary_list[word] = 1
                            else:
                                previous_value = vocabulary_list[word]
                                vocabulary_list[word] = previous_value + 1

                        if str(match_obj.group(2)) == "NOUN":
                            number_of_nouns += 1
                        elif str(match_obj.group(2)) == "VERB":
                            number_of_verbs += 1
                        elif str(match_obj.group(2)) == "ADJ":
                            number_of_adjectives += 1
                        elif str(match_obj.group(2)) == "ADV":
                            number_of_adverbs += 1
                        elif str(match_obj.group(2)) == "PUNCT":
                            number_of_punct += 1
                        elif str(match_obj.group(2)) == "AUX":
                            number_of_aux += 1
                        elif str(match_obj.group(2)) == "DET":
                            number_of_det += 1
                        elif str(match_obj.group(2)) == "PART":
                            number_of_part += 1
                        elif str(match_obj.group(2)) == "CCONJ":
                            number_of_cconj += 1
                        elif str(match_obj.group(2)) == "SCONJ":
                            number_of_sconj += 1
                        elif str(match_obj.group(2)) == "PROPN":
                            number_of_propn += 1
                        elif str(match_obj.group(2)) == "PRON":
                            number_of_pron += 1
                        elif str(match_obj.group(2)) == "X":
                            # print("Line{}: {}".format(count, line.strip()))
                            log_file.write("No match!!")
                            number_of_x += 1
                        elif str(match_obj.group(2)) == "ADP":
                            number_of_adp += 1
                        elif str(match_obj.group(2)) == "NUM":
                            number_of_num += 1
                        elif str(match_obj.group(2)) == "INTJ":
                            number_of_intj += 1
                        else:
                            print("Line{}: {}".format(count, line.strip()))
                            print(str(match_obj.group(2)))
                            log_file.write("\n")
                            log_file.write("Line{}: {}".format(count, line.strip()) + "\n")

                        log_file.write("match_obj.group(1) : " + str(match_obj.group(1)) + "\n")
                        log_file.write("match_obj.group(2) : " + str(match_obj.group(2)) + "\n")
                        log_file.write("match_obj.group(3) : " + str(match_obj.group(3)) + "\n")
                    else:
                        print("No match!! " + line)

                number_of_words = number_of_adjectives + number_of_adp + number_of_adverbs + number_of_aux + \
                                  number_of_cconj + number_of_det + number_of_intj + number_of_nouns + number_of_num + \
                                  number_of_part + number_of_pron + number_of_propn + number_of_sconj + number_of_verbs

                vocabulary = len(vocabulary_list)
                average_words_length = words_length / number_of_words

                # calculate vocabulary spoken once
                for item in vocabulary_list:
                    if vocabulary_list[item] == 1:
                        vocabulary_spoken_once += 1

                TTR = (vocabulary * 1.0) / (number_of_words * 1.0)
                W = number_of_words ** (vocabulary ** (-0.165))

                value = 1 - vocabulary_spoken_once / vocabulary
                if value == 0:
                    value = 0.0001
                R = (100 * math.log(number_of_words)) / value

                patient_log_file.write("number_of_words: " + str(number_of_words) + "\n")
                patient_log_file.write("average_words_length: " + str(average_words_length) + "\n")
                patient_log_file.write("vocabulary: " + str(vocabulary) + "\n")
                patient_log_file.write("vocabulary spoken once: " + str(vocabulary_spoken_once) + "\n")
                patient_log_file.write("------------------------------------------------\n")

                patient_log_file.write("Type token ratio: " + str(TTR) + "\n")
                patient_log_file.write("Brunet's Index: " + str(W) + "\n")
                patient_log_file.write("Honore's Statistic: " + str(R) + "\n")
                patient_log_file.write("------------------------------------------------\n")

                number_of_nouns += number_of_propn
                patient_log_file.write("number of nouns: " + str(number_of_nouns + number_of_propn) + "\n")
                patient_log_file.write("number of verbs: " + str(number_of_verbs) + "\n")
                patient_log_file.write("number of adjectives: " + str(number_of_adjectives) + "\n")
                patient_log_file.write("number of adverbs: " + str(number_of_adverbs) + "\n")
                patient_log_file.write("number of punct: " + str(number_of_punct) + "\n")
                patient_log_file.write("number of aux: " + str(number_of_aux) + "\n")
                patient_log_file.write("number of det: " + str(number_of_det) + "\n")
                patient_log_file.write("number of part: " + str(number_of_part) + "\n")
                patient_log_file.write("number of cconj: " + str(number_of_cconj) + "\n")
                patient_log_file.write("number of sconj: " + str(number_of_sconj) + "\n")
                patient_log_file.write("number of propn: " + str(number_of_propn) + "\n")
                patient_log_file.write("number of pronn: " + str(number_of_pron) + "\n")
                patient_log_file.write("number of x: " + str(number_of_x) + "\n")
                patient_log_file.write("number of adp: " + str(number_of_adp) + "\n")
                patient_log_file.write("number of num: " + str(number_of_num) + "\n")
                patient_log_file.write("number of intj: " + str(number_of_intj) + "\n")

                patient_log_file.write("Normalized ------------------------------------------------\n")
                patient_log_file.write(
                    "number of nouns normalized: " + str((number_of_nouns + number_of_propn) / number_of_words) + "\n")
                patient_log_file.write(
                    "number of verbs normalized:: " + str((number_of_verbs + number_of_aux) / number_of_words) + "\n")
                patient_log_file.write(
                    "number of adjectives normalized: " + str(number_of_adjectives / number_of_words) + "\n")
                patient_log_file.write(
                    "number of adverbs normalized: " + str(number_of_adverbs / number_of_words) + "\n")
                patient_log_file.write("number of aux normalized: " + str(number_of_aux / number_of_words) + "\n")
                patient_log_file.write("number of det normalized: " + str(number_of_det / number_of_words) + "\n")
                patient_log_file.write("number of part normalized: " + str(number_of_part / number_of_words) + "\n")
                patient_log_file.write("number of cconj normalized: " + str(number_of_cconj / number_of_words) + "\n")
                patient_log_file.write("number of sconj normalized: " + str(number_of_sconj / number_of_words) + "\n")
                patient_log_file.write("number of propn normalized: " + str(number_of_propn / number_of_words) + "\n")
                patient_log_file.write("number of pronn normalized: " + str(number_of_pron / number_of_words) + "\n")
                patient_log_file.write("number of adp normalized: " + str(number_of_adp / number_of_words) + "\n")
                patient_log_file.write("number of num normalized: " + str(number_of_num / number_of_words) + "\n")
                patient_log_file.write("number of intj normalized: " + str(number_of_intj / number_of_words) + "\n")

                patient_log_file.write("Ratios ------------------------------------------------\n")
                if number_of_verbs != 0:
                    patient_log_file.write("ratio noun to verb: " + str(
                        (number_of_nouns + number_of_propn) / (number_of_verbs + number_of_aux)) + "\n")
                    patient_log_file.write(
                        "ratio pronoun to verb: " + str(number_of_pron / (number_of_verbs + number_of_aux)) + "\n")
                if number_of_nouns != 0:
                    patient_log_file.write(
                        "ratio pronoun to noun: " + str(number_of_pron / (number_of_nouns + number_of_propn)) + "\n")

                patient_log_file.close()
                file.close()

    log_file.close()


def read_results(patient_class, path_log_file, path_to_tagged) -> None:
    """
        Functions for reading statistics from persion_name.txt statistic file for every person and storing information
        to global variable arrays
    """
    for path in pathlib.Path(PATH_TO_TAGGED_FILES + "/logs/" + path_to_tagged).iterdir():
        if path.is_file():
            # Open next file in the folder
            print(path)
            file = open(path, mode="r", encoding="utf8")
            lines = file.readlines()
            count = 0
            # Explore every line
            for line in lines:

                # Match regex against the line
                match_obj = re.match(REGEX_READ_FILE, line, re.M | re.I)
                if match_obj:
                    # Store in global variable arrays for positive patients
                    if patient_class == "P":
                        if count == 2:
                            GLOBAL_NUMBER_OF_WORDS_P.append(int(match_obj.group(1)))
                        elif count == 3:
                            GLOBAL_AVERAGE_WORDS_LEN_P.append(float(match_obj.group(1)))
                        elif count == 4:
                            GLOBAL_VOCABULARY_P.append(int(match_obj.group(1)))
                        elif count == 5:
                            GLOBAL_VOCABULARY_SO_P.append(int(match_obj.group(1)))
                        elif count == 7:
                            GLOBAL_TTR_P.append(float(match_obj.group(1)))
                        elif count == 8:
                            GLOBAL_W_P.append(float(match_obj.group(1)))
                        elif count == 9:
                            GLOBAL_R_P.append(float(match_obj.group(1)))
                        elif count == 11:
                            GLOBAL_NUMBER_OF_NOUNS_P.append(int(match_obj.group(1)))
                        elif count == 12:
                            GLOBAL_NUMBER_OF_VERBS_P.append(int(match_obj.group(1)))
                        elif count == 13:
                            GLOBAL_NUMBER_OF_ADJECTIVES_P.append(int(match_obj.group(1)))
                        elif count == 14:
                            GLOBAL_NUMBER_OF_ADVERBS_P.append(int(match_obj.group(1)))
                        elif count == 18:
                            GLOBAL_NUMBER_OF_PART_P.append(int(match_obj.group(1)))
                        elif count == 19:
                            GLOBAL_NUMBER_OF_CCONJ_P.append(int(match_obj.group(1)))
                        elif count == 20:
                            GLOBAL_NUMBER_OF_SCONJ_P.append(int(match_obj.group(1)))
                        elif count == 22:
                            GLOBAL_NUMBER_OF_PRONN_P.append(int(match_obj.group(1)))
                        elif count == 28:
                            GLOBAL_NUMBER_OF_NOUNS_NOR_P.append(float(match_obj.group(1)))
                        elif count == 29:
                            GLOBAL_NUMBER_OF_VERBS_NOR_P.append(float(match_obj.group(1)))
                        elif count == 30:
                            GLOBAL_NUMBER_OF_ADJECTIVES_NOR_P.append(float(match_obj.group(1)))
                        elif count == 31:
                            GLOBAL_NUMBER_OF_ADVERBS_NOR_P.append(float(match_obj.group(1)))
                        elif count == 34:
                            GLOBAL_NUMBER_OF_PART_NOR_P.append(float(match_obj.group(1)))
                        elif count == 35:
                            GLOBAL_NUMBER_OF_CCONJ_NOR_P.append(float(match_obj.group(1)))
                        elif count == 36:
                            GLOBAL_NUMBER_OF_SCONJ_NOR_P.append(float(match_obj.group(1)))
                        elif count == 38:
                            GLOBAL_NUMBER_OF_PRONN_NOR_P.append(float(match_obj.group(1)))
                        elif count == 43:
                            GLOBAL_NOUN_VERB_P.append(float(match_obj.group(1)))
                        elif count == 44:
                            GLOBAL_PRONOUN_VERB_P.append(float(match_obj.group(1)))
                        elif count == 45:
                            GLOBAL_PRONOUN_NOUN_P.append(float(match_obj.group(1)))
                    # Store in global variable arrays for negative patients
                    elif patient_class == "N":
                        if count == 2:
                            GLOBAL_NUMBER_OF_WORDS_N.append(int(match_obj.group(1)))
                        elif count == 3:
                            GLOBAL_AVERAGE_WORDS_LEN_N.append(float(match_obj.group(1)))
                        elif count == 4:
                            GLOBAL_VOCABULARY_N.append(int(match_obj.group(1)))
                        elif count == 5:
                            GLOBAL_VOCABULARY_SO_N.append(int(match_obj.group(1)))
                        elif count == 7:
                            GLOBAL_TTR_N.append(float(match_obj.group(1)))
                        elif count == 8:
                            GLOBAL_W_N.append(float(match_obj.group(1)))
                        elif count == 9:
                            GLOBAL_R_N.append(float(match_obj.group(1)))
                        elif count == 11:
                            GLOBAL_NUMBER_OF_NOUNS_N.append(int(match_obj.group(1)))
                        elif count == 12:
                            GLOBAL_NUMBER_OF_VERBS_N.append(int(match_obj.group(1)))
                        elif count == 13:
                            GLOBAL_NUMBER_OF_ADJECTIVES_N.append(int(match_obj.group(1)))
                        elif count == 14:
                            GLOBAL_NUMBER_OF_ADVERBS_N.append(int(match_obj.group(1)))
                        elif count == 18:
                            GLOBAL_NUMBER_OF_PART_N.append(int(match_obj.group(1)))
                        elif count == 19:
                            GLOBAL_NUMBER_OF_CCONJ_N.append(int(match_obj.group(1)))
                        elif count == 20:
                            GLOBAL_NUMBER_OF_SCONJ_N.append(int(match_obj.group(1)))
                        elif count == 22:
                            GLOBAL_NUMBER_OF_PRONN_N.append(int(match_obj.group(1)))
                        elif count == 28:
                            GLOBAL_NUMBER_OF_NOUNS_NOR_N.append(float(match_obj.group(1)))
                        elif count == 29:
                            GLOBAL_NUMBER_OF_VERBS_NOR_N.append(float(match_obj.group(1)))
                        elif count == 30:
                            GLOBAL_NUMBER_OF_ADJECTIVES_NOR_N.append(float(match_obj.group(1)))
                        elif count == 31:
                            GLOBAL_NUMBER_OF_ADVERBS_NOR_N.append(float(match_obj.group(1)))
                        elif count == 34:
                            GLOBAL_NUMBER_OF_PART_NOR_N.append(float(match_obj.group(1)))
                        elif count == 35:
                            GLOBAL_NUMBER_OF_CCONJ_NOR_N.append(float(match_obj.group(1)))
                        elif count == 36:
                            GLOBAL_NUMBER_OF_SCONJ_NOR_N.append(float(match_obj.group(1)))
                        elif count == 38:
                            GLOBAL_NUMBER_OF_PRONN_NOR_N.append(float(match_obj.group(1)))
                        elif count == 43:
                            GLOBAL_NOUN_VERB_N.append(float(match_obj.group(1)))
                        elif count == 44:
                            GLOBAL_PRONOUN_VERB_N.append(float(match_obj.group(1)))
                        elif count == 45:
                            GLOBAL_PRONOUN_NOUN_N.append(float(match_obj.group(1)))
                count += 1


# Function for calling plotting module with needed data
def draw_graphs_for_lexical_statistics():
    v = 1

    plot_lexical_analysis_results_two_plots(GLOBAL_NUMBER_OF_NOUNS_P, 2000, GLOBAL_NUMBER_OF_NOUNS_N, "Number of nouns",
                                            GLOBAL_NUMBER_OF_NOUNS_NOR_P, np.mean(GLOBAL_NUMBER_OF_NOUNS_NOR_P), 1,
                                            GLOBAL_NUMBER_OF_NOUNS_NOR_N, np.mean(GLOBAL_NUMBER_OF_NOUNS_NOR_N),
                                            "Number of nouns normalized by number of words")

    plot_lexical_analysis_results_two_plots(GLOBAL_NUMBER_OF_ADJECTIVES_P, 300, GLOBAL_NUMBER_OF_ADJECTIVES_N,
                                            "Number of adjectives", GLOBAL_NUMBER_OF_ADJECTIVES_NOR_P,
                                            np.mean(GLOBAL_NUMBER_OF_ADJECTIVES_NOR_P), 1,
                                            GLOBAL_NUMBER_OF_ADJECTIVES_NOR_N,
                                            np.mean(GLOBAL_NUMBER_OF_ADJECTIVES_NOR_N),
                                            "Number of adjectives normalized by number of words")

    plot_lexical_analysis_results_two_plots(GLOBAL_NUMBER_OF_ADVERBS_P, 600, GLOBAL_NUMBER_OF_ADVERBS_N,
                                            "Number of adverbs", GLOBAL_NUMBER_OF_ADVERBS_NOR_P,
                                            np.mean(GLOBAL_NUMBER_OF_ADVERBS_NOR_P), 1, GLOBAL_NUMBER_OF_ADVERBS_NOR_N,
                                            np.mean(GLOBAL_NUMBER_OF_ADVERBS_NOR_N),
                                            "Number of adverbs normalized by number of words")

    plot_lexical_analysis_results_two_plots(GLOBAL_NUMBER_OF_CCONJ_P, 400, GLOBAL_NUMBER_OF_CCONJ_N,
                                            "Number of cconj", GLOBAL_NUMBER_OF_CCONJ_NOR_P,
                                            np.mean(GLOBAL_NUMBER_OF_CCONJ_NOR_P), 1, GLOBAL_NUMBER_OF_CCONJ_NOR_N,
                                            np.mean(GLOBAL_NUMBER_OF_CCONJ_NOR_N),
                                            "Number of cconj normalized by number of words")

    plot_lexical_analysis_results_two_plots(GLOBAL_NUMBER_OF_PART_P, 400, GLOBAL_NUMBER_OF_PART_N,
                                            "Number of part", GLOBAL_NUMBER_OF_PART_NOR_P,
                                            np.mean(GLOBAL_NUMBER_OF_PART_NOR_P), 1, GLOBAL_NUMBER_OF_PART_NOR_N,
                                            np.mean(GLOBAL_NUMBER_OF_PART_NOR_N),
                                            "Number of part normalized by number of words")

    plot_lexical_analysis_results_two_plots(GLOBAL_NUMBER_OF_PRONN_P, 100, GLOBAL_NUMBER_OF_PRONN_N,
                                            "Number of pronn",
                                            GLOBAL_NUMBER_OF_PRONN_NOR_P, np.mean(GLOBAL_NUMBER_OF_PRONN_NOR_P), 1,
                                            GLOBAL_NUMBER_OF_PRONN_NOR_N, np.mean(GLOBAL_NUMBER_OF_PRONN_NOR_N),
                                            "Number of pronn normalized by number of words")

    plot_lexical_analysis_results_two_plots(GLOBAL_NUMBER_OF_SCONJ_P, 650, GLOBAL_NUMBER_OF_SCONJ_N,
                                            "Number of sconj",
                                            GLOBAL_NUMBER_OF_SCONJ_NOR_P, np.mean(GLOBAL_NUMBER_OF_SCONJ_NOR_P), 1,
                                            GLOBAL_NUMBER_OF_SCONJ_NOR_N, np.mean(GLOBAL_NUMBER_OF_SCONJ_NOR_N),
                                            "Number of sconj normalized by number of words")

    plot_lexical_analysis_results_two_plots(GLOBAL_NUMBER_OF_VERBS_P, 1500, GLOBAL_NUMBER_OF_VERBS_N,
                                            "Number of verbs",
                                            GLOBAL_NUMBER_OF_VERBS_NOR_P, np.mean(GLOBAL_NUMBER_OF_VERBS_NOR_P), 1,
                                            GLOBAL_NUMBER_OF_VERBS_NOR_N, np.mean(GLOBAL_NUMBER_OF_VERBS_NOR_N),
                                            "Number of verbs normalized by number of words")

    plot_lexical_analysis_results_one_plot(GLOBAL_NUMBER_OF_WORDS_P, np.mean(GLOBAL_NUMBER_OF_WORDS_P), 8000,
                                           GLOBAL_NUMBER_OF_WORDS_N, np.mean(GLOBAL_NUMBER_OF_WORDS_N),
                                           "Number of words")

    plot_lexical_analysis_results_one_plot(GLOBAL_AVERAGE_WORDS_LEN_P, np.mean(GLOBAL_AVERAGE_WORDS_LEN_P), 6,
                                           GLOBAL_AVERAGE_WORDS_LEN_N, np.mean(GLOBAL_AVERAGE_WORDS_LEN_N),
                                           "Average words length")

    plot_lexical_analysis_results_one_plot(GLOBAL_NOUN_VERB_P, np.mean(GLOBAL_NOUN_VERB_P), 4.6, GLOBAL_NOUN_VERB_N,
                                           np.mean(GLOBAL_NOUN_VERB_N), "Noun/Verb ratio")
    plot_lexical_analysis_results_one_plot(GLOBAL_PRONOUN_NOUN_P, np.mean(GLOBAL_PRONOUN_NOUN_P), 1,
                                           GLOBAL_PRONOUN_NOUN_N, np.mean(GLOBAL_PRONOUN_NOUN_N), "Pronoun/Noun ratio")

    plot_lexical_analysis_results_one_plot(GLOBAL_TTR_P, np.mean(GLOBAL_TTR_P), 1.2, GLOBAL_TTR_N,
                                           np.mean(GLOBAL_TTR_N), "Type Token Ratio")
    plot_lexical_analysis_results_one_plot(GLOBAL_R_P, np.mean(GLOBAL_R_P), 2300, GLOBAL_R_N, np.mean(GLOBAL_R_N),
                                           "Brunet's Index")
    plot_lexical_analysis_results_one_plot(GLOBAL_W_P, np.mean(GLOBAL_W_P), 20, GLOBAL_W_N, np.mean(GLOBAL_W_N),
                                           "Honore's Statistic")


def start_lexical_analysis(draw_graphs: bool, statistics: bool):
    """
    Function to start lexical analysis. Reading of tagged files has to be done at least once.
    Results are stored in files, read in read_results functions and those results are
    shown on graphs.
    """
    print(" Starting Lexical Analysis ")

    # read_tagged_files(PATH_FOR_LOG_FILE + '/log_P_UD.txt', 'P_UD')
    print(" Finished reading P files ")

    # read_tagged_files(PATH_FOR_LOG_FILE + '/log_N_UD.txt', 'N_UD')
    print(" Finished reading N files ")

    if draw_graphs:
        read_results("P", PATH_FOR_LOG_FILE + '/log_P_UD.txt', 'P_UD')
        print(" Finished reading P results ")

        read_results("N", PATH_FOR_LOG_FILE + '/log_N_UD.txt', 'N_UD')
        print(" Finished reading N results")

        draw_graphs_for_lexical_statistics()
        print(" Draw graphs finished")

    if statistics:
        calculate_statistics(PATH_TO_TRAIN_TEST_CORPUS + POSITIVE_FOLDER_1, PATH_TO_TRAIN_TEST_CORPUS + POSITIVE_FOLDER_2,
                             PATH_TO_TRAIN_TEST_CORPUS + NEGATIVE_FOLDER_1, PATH_TO_TRAIN_TEST_CORPUS + NEGATIVE_FOLDER_2,
                             PATH_TO_TRAIN_TEST_CORPUS + POSITIVE_FOLDER_3, PATH_TO_TRAIN_TEST_CORPUS + NEGATIVE_FOLDER_3,
                             "train_1_2_test_3.txt")

        calculate_statistics(PATH_TO_TRAIN_TEST_CORPUS + POSITIVE_FOLDER_1, PATH_TO_TRAIN_TEST_CORPUS + POSITIVE_FOLDER_3,
                             PATH_TO_TRAIN_TEST_CORPUS + NEGATIVE_FOLDER_1, PATH_TO_TRAIN_TEST_CORPUS + NEGATIVE_FOLDER_3,
                             PATH_TO_TRAIN_TEST_CORPUS + POSITIVE_FOLDER_2, PATH_TO_TRAIN_TEST_CORPUS + NEGATIVE_FOLDER_2,
                             "train_1_3_test_2.txt")

        calculate_statistics(PATH_TO_TRAIN_TEST_CORPUS + POSITIVE_FOLDER_3, PATH_TO_TRAIN_TEST_CORPUS + POSITIVE_FOLDER_2,
                             PATH_TO_TRAIN_TEST_CORPUS + NEGATIVE_FOLDER_3, PATH_TO_TRAIN_TEST_CORPUS + NEGATIVE_FOLDER_2,
                             PATH_TO_TRAIN_TEST_CORPUS + POSITIVE_FOLDER_1, PATH_TO_TRAIN_TEST_CORPUS + NEGATIVE_FOLDER_1,
                             "train_3_2_test_1.txt")

    print(" Ending Lexical Analysis ")


# Function that starts Machine Learning algorithms
def start_machine_learning():
    print(" Starting Machine Learning Algorithms ")

    do_machine_learning(POSITIVE_FOLDER_1, POSITIVE_FOLDER_3, NEGATIVE_FOLDER_1, NEGATIVE_FOLDER_3, POSITIVE_FOLDER_2,
                        NEGATIVE_FOLDER_2, (1, 1), MLAlgorithm.SVM)

    do_machine_learning(POSITIVE_FOLDER_1, POSITIVE_FOLDER_2, NEGATIVE_FOLDER_1, NEGATIVE_FOLDER_2, POSITIVE_FOLDER_3,
                        NEGATIVE_FOLDER_3, (1, 1), MLAlgorithm.SVM)

    do_machine_learning(POSITIVE_FOLDER_2, POSITIVE_FOLDER_3, NEGATIVE_FOLDER_2, NEGATIVE_FOLDER_3, POSITIVE_FOLDER_1,
                        NEGATIVE_FOLDER_1, (1, 1), MLAlgorithm.SVM)

    do_machine_learning(POSITIVE_FOLDER_1, POSITIVE_FOLDER_2, NEGATIVE_FOLDER_1, NEGATIVE_FOLDER_2, POSITIVE_FOLDER_3,
                        NEGATIVE_FOLDER_3, (1, 1), MLAlgorithm.NaiveBayes)

    do_machine_learning(POSITIVE_FOLDER_1, POSITIVE_FOLDER_3, NEGATIVE_FOLDER_1, NEGATIVE_FOLDER_3, POSITIVE_FOLDER_2,
                        NEGATIVE_FOLDER_2, (1, 1), MLAlgorithm.NaiveBayes)

    do_machine_learning(POSITIVE_FOLDER_2, POSITIVE_FOLDER_3, NEGATIVE_FOLDER_2, NEGATIVE_FOLDER_3, POSITIVE_FOLDER_1,
                        NEGATIVE_FOLDER_1, (1, 1), MLAlgorithm.NaiveBayes)

    print(" Ending Machine Learning Algorithms ")


# Main function
if __name__ == '__main__':
    print(" Start Classification of Alzheimer Disease ")

    start_machine_learning()
    # start_lexical_analysis(False, True)
    # calculate_cosine_distance()

    print(" End Classification of Alzheimer Disease ")

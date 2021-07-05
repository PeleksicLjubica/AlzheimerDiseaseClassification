# main.py
# Script used to start framework
import pathlib
import re
import os
import sys
import math  # needed for logarithm
from utilities import *


def read_tagged_files():
    log_file = open(PATH_FOR_LOG_FILE + '/log_P_UD.txt', mode="w", encoding="utf8")

    for path in pathlib.Path(PATH_TO_TAGGED_FILES + '/P_UD').iterdir():
        if path.is_file():
            # Open next file in the folder
            file = open(path, mode="r", encoding="utf8")

            # Find patient name to make log file with data for that patient
            head, tail = os.path.split(path)
            patient_name = tail.split(".")[0]
            print(tail.split(".")[0])
            patient_log_file = open(PATH_FOR_LOG_FILE + '/' + patient_name + '.txt', mode="w", encoding="utf8")
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

            # Variables containing other mertics
            TTR = 0  # Type token ration
            W = 0  # Brunets Index
            R = 0  # Honores Statistic

            # Explore every line
            for line in lines:
                count += 1

                # Match regex against the line
                match_obj = re.match(REGEX, line, re.M | re.I)
                if match_obj:

                    words_length += len(str(match_obj.group(3)))

                    # If type of object is not PUNCT - punctuation marks, word should be included in vocabulary
                    if str(match_obj.group(2)) != "PUNCT":
                        word = str(match_obj.group(3)).lower()
                        if word not in vocabulary_list:
                            vocabulary_list[word] = 1
                        else:
                            previous_value = vocabulary_list[word] = 1
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
                        # print("Line{}: {}".format(count, line.strip()))
                        print(str(match_obj.group(2)))
                        log_file.write("\n")
                        log_file.write("Line{}: {}".format(count, line.strip()) + "\n")

                    log_file.write("match_obj.group(1) : " + str(match_obj.group(1)) + "\n")
                    log_file.write("match_obj.group(2) : " + str(match_obj.group(2)) + "\n")
                    log_file.write("match_obj.group(3) : " + str(match_obj.group(3)) + "\n")
                else:
                    print("No match!!")

            number_of_words = number_of_adverbs + number_of_det + number_of_pron + number_of_adp + \
                              number_of_propn + number_of_sconj + number_of_punct + number_of_cconj + number_of_part + \
                              number_of_aux + number_of_adjectives + number_of_nouns + number_of_verbs + number_of_intj + \
                              number_of_x
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
                value = 0.000000001
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

            patient_log_file.write("number of nouns: " + str(number_of_nouns) + "\n")
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
            patient_log_file.write("number of nouns normalized: " + str(number_of_nouns / number_of_words) + "\n")
            patient_log_file.write("number of verbs normalized:: " + str(number_of_verbs / number_of_words) + "\n")
            patient_log_file.write(
                "number of adjectives normalized:: " + str(number_of_adjectives / number_of_words) + "\n")
            patient_log_file.write("number of adverbs normalized:: " + str(number_of_adverbs / number_of_words) + "\n")
            patient_log_file.write("number of aux normalized:: " + str(number_of_aux / number_of_words) + "\n")
            patient_log_file.write("number of det normalized:: " + str(number_of_det / number_of_words) + "\n")
            patient_log_file.write("number of part normalized:: " + str(number_of_part / number_of_words) + "\n")
            patient_log_file.write("number of cconj normalized:: " + str(number_of_cconj / number_of_words) + "\n")
            patient_log_file.write("number of sconj normalized:: " + str(number_of_sconj / number_of_words) + "\n")
            patient_log_file.write("number of propn normalized:: " + str(number_of_propn / number_of_words) + "\n")
            patient_log_file.write("number of pronn normalized:: " + str(number_of_pron / number_of_words) + "\n")
            patient_log_file.write("number of adp normalized:: " + str(number_of_adp / number_of_words) + "\n")
            patient_log_file.write("number of num normalized:: " + str(number_of_num / number_of_words) + "\n")
            patient_log_file.write("number of intj normalized:: " + str(number_of_intj / number_of_words) + "\n")

            patient_log_file.write("Ratios ------------------------------------------------\n")
            if number_of_verbs != 0:
                patient_log_file.write("ratio noun to verb: " + str(number_of_nouns / number_of_verbs) + "\n")
                patient_log_file.write("ratio pronoun to verb: " + str(number_of_pron / number_of_verbs) + "\n")
            if number_of_nouns != 0:
                patient_log_file.write("ratio pronoun to noun: " + str(number_of_pron / number_of_nouns) + "\n")

            file.close()
    log_file.close()


def start_lexical_analysis():
    print("*******************************************")
    print(" Starting Lexical Analysis ")

    read_tagged_files()

    print("*******************************************")
    print(" Ending Lexical Analysis ")


# Function that starts Machine Learning algorithms
def start_machine_learning():
    print("*******************************************")
    print(" Starting Machine Learning Algorithms ")

    bag_of_words()

    print("*******************************************")
    print(" Ending Machine Learning Algorithms ")


def bag_of_words():
    print("*******************************************")
    print(" Starting Bag of words ")

    print("*******************************************")
    print(" Ending Bag of words  ")


# Main function
if __name__ == '__main__':
    print("*******************************************")
    print(" Start Classification of Alzheimer Disease ")

    start_machine_learning()
    start_lexical_analysis()

    print("*******************************************")
    print(" End Classification of Alzheimer Disease ")

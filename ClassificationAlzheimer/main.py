# main.py
# Script used to start framework
import pathlib
import re
import os
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

            print("------------------------------------------------")
            print(path)

            log_file.write("------------------------------------------------\n")
            log_file.write(str(path) + "\n")

            lines = file.readlines()
            count = 0
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

            for line in lines:
                count += 1

                #print("Line{}: {}".format(count, line.strip()))
                #log_file.write("\n")
                #log_file.write("Line{}: {}".format(count, line.strip()) + "\n")

                match_obj = re.match(REGEX, line, re.M | re.I)
                if match_obj:

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
                    print("No match!!")
                    print("Line{}: {}".format(count, line.strip()))
                    log_file.write("No match!!")

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

import pathlib
from utilities import *
import re


def find_all_train_statistics(measure: Measurement, train_data, path_to_folder):
    """
    Function to collect specific measurement for all train data in path_to_folder
    :param measure:
    :param train_data:
    :param path_to_folder:
    :return:
    """

    print(" Enter find_all_train_statistics")

    for path in pathlib.Path(path_to_folder).iterdir():
        if path.is_file():
            # Open next file in the folder
            print(path)
            file = open(path, encoding="utf8")
            if ".DS_Store" not in path.parts:
                lines = file.readlines()
                line = lines[measure.value]

                match_obj = re.match(REGEX_READ_FILE, line, re.M | re.I)
                if match_obj:
                    train_data.append(float(match_obj.group(1)))

    print(" Exit find_all_positive_train_statistics")
    return train_data


def find_all_test_statistics(measure: Measurement, class_clasif: Clasiffication_Class, path_to_folder,
                             positive_train_data, negative_train_data):

    """
    Function to collect specific measurement for all test data and calculate True Positives, True Negatives, False
    Positives and False Negatives for every document in test corpus
    :param measure:
    :param class_clasif:
    :param path_to_folder:
    :param positive_train_data:
    :param negative_train_data:
    :return:
    """

    TruePositives = 0
    FalsePositives = 0
    TrueNegatives = 0
    FalseNegatives = 0

    # Iterate over positive training data
    for path in pathlib.Path(path_to_folder).iterdir():

        distance_from_document_to_p = 0
        distance_from_document_to_n = 0

        if path.is_file():
            # Open next file in the folder
            file = open(path, encoding="utf8")
            if ".DS_Store" not in path.parts:
                lines = file.readlines()
                line = lines[measure.value]

                match_obj = re.match(REGEX_READ_FILE, line, re.M | re.I)
                if match_obj:
                    test_document_value = float(match_obj.group(1))

                    for value in positive_train_data:
                        distance_from_document_to_p += abs(test_document_value - value)

                    for value in negative_train_data:
                        distance_from_document_to_n += abs(test_document_value - value)

                    if class_clasif == Clasiffication_Class.POSITIVE:
                        if distance_from_document_to_p < distance_from_document_to_n:
                            TruePositives += 1
                        else:
                            FalseNegatives += 1
                    else:
                        if distance_from_document_to_p < distance_from_document_to_n:
                            FalsePositives += 1
                        else:
                            TrueNegatives += 1

    if class_clasif == Clasiffication_Class.POSITIVE:
        return [TruePositives, FalseNegatives]
    elif class_clasif == Clasiffication_Class.NEGATIVE:
        return [TrueNegatives, FalsePositives]


def calculate_statistics_for_measure(measure: Measurement, path_to_train_1, path_to_train_2, path_to_train_3,
                                     path_to_train_4, path_to_test_1, path_to_test_2):
    """
    Function to collect train statistics and
    :param measure:
    :param path_to_train_1:
    :param path_to_train_2:
    :param path_to_train_3:
    :param path_to_train_4:
    :param path_to_test_1:
    :param path_to_test_2:
    :return:
    """
    print(" Enter calculate_statistics_for_measure")

    # Collect data for specific measure for training corpus
    positive_train_data = []
    negative_train_data =[]
    positive_train_data = find_all_train_statistics(measure, positive_train_data, path_to_train_1)
    positive_train_data = find_all_train_statistics(measure, positive_train_data, path_to_train_2)

    negative_train_data = find_all_train_statistics(measure, negative_train_data, path_to_train_3)
    negative_train_data = find_all_train_statistics(measure, negative_train_data, path_to_train_4)

    [TruePositives, FalseNegatives] = find_all_test_statistics(measure, Clasiffication_Class.POSITIVE,
                                                               path_to_test_1, positive_train_data, negative_train_data)

    [_, FalsePositives] = find_all_test_statistics(measure, Clasiffication_Class.NEGATIVE,
                                                               path_to_test_2, positive_train_data, negative_train_data)

    P = TruePositives / (TruePositives + FalsePositives)
    R = TruePositives / (TruePositives + FalseNegatives)
    F = (2 * P * R) / (P + R)
    print(" Exit calculate_statistics_for_measure")
    return P, R, F


def calculate_statistics(path_to_train_1, path_to_train_2, path_to_train_3, path_to_train_4, path_to_test_1,
                         path_to_test_2, log_file_name):
    """
    Function to call statistic calculation for all measures and save calculations to log files
    :param path_to_train_1:
    :param path_to_train_2:
    :param path_to_train_3:
    :param path_to_train_4:
    :param path_to_test_1:
    :param path_to_test_2:
    :param log_file_name: Path with file name for log file
    :return:
    """


    print(" Enter calculate_statistics")

    P_TTR, R_TTR, F_TTR = calculate_statistics_for_measure(Measurement.TTR, path_to_train_1, path_to_train_2,
                                                           path_to_train_3, path_to_train_4, path_to_test_1,
                                                           path_to_test_2)
    P_R, R_R, F_R = calculate_statistics_for_measure(Measurement.R,path_to_train_1, path_to_train_2, path_to_train_3,
                                                     path_to_train_4, path_to_test_1, path_to_test_2)
    P_W, R_W, F_W = calculate_statistics_for_measure(Measurement.W, path_to_train_1, path_to_train_2, path_to_train_3,
                                                     path_to_train_4, path_to_test_1, path_to_test_2)
    P_nouns, R_nouns, F_nouns = calculate_statistics_for_measure(Measurement.nouns, path_to_train_1, path_to_train_2,
                                                                 path_to_train_3, path_to_train_4, path_to_test_1,
                                                                 path_to_test_2)
    P_verbs, R_verbs, F_verbs = calculate_statistics_for_measure(Measurement.verbs, path_to_train_1, path_to_train_2,
                                                                 path_to_train_3, path_to_train_4, path_to_test_1,
                                                                 path_to_test_2)
    P_adj, R_adj, F_adj = calculate_statistics_for_measure(Measurement.adj, path_to_train_1, path_to_train_2,
                                                           path_to_train_3, path_to_train_4, path_to_test_1,
                                                           path_to_test_2)
    P_adv, R_adv, F_adv = calculate_statistics_for_measure(Measurement.adv, path_to_train_1, path_to_train_2,
                                                           path_to_train_3, path_to_train_4, path_to_test_1,
                                                           path_to_test_2)
    P_noun_verb, R_noun_verb, F_noun_verb = calculate_statistics_for_measure(Measurement.noun_verb, path_to_train_1,
                                                                             path_to_train_2, path_to_train_3,
                                                                             path_to_train_4, path_to_test_1,
                                                                             path_to_test_2)
    P_pronoun_verb, R_pronoun_verb, F_pronoun_verb = calculate_statistics_for_measure(Measurement.pronoun_verb,
                                                                                      path_to_train_1, path_to_train_2,
                                                                                      path_to_train_3, path_to_train_4,
                                                                                      path_to_test_1, path_to_test_2)
    P_pronoun_noun, R_pronoun_noun, F_pronoun_noun = calculate_statistics_for_measure(Measurement.pronoun_noun,
                                                                                      path_to_train_1, path_to_train_2,
                                                                                      path_to_train_3, path_to_train_4,
                                                                                      path_to_test_1, path_to_test_2)

    log_file = open(PATH_TO_STATISTIC_FOLDER + "/" + log_file_name, mode="w+", encoding="utf8")
    log_file.write("Train folders: \n")
    log_file.write(path_to_train_1 + "\n")
    log_file.write(path_to_train_2 + "\n")
    log_file.write(path_to_train_3 + "\n")
    log_file.write(path_to_train_4 + "\n")

    log_file.write("Test folders: \n")
    log_file.write(path_to_test_1 + "\n")
    log_file.write(path_to_test_2 + "\n")

    log_file.write("\n")

    log_file.write("TTR statistic: " + "Precision: " + str(P_TTR) + " Recall: " + str(R_TTR) + " F-measure: "
                   + str(F_TTR) + "\n")
    log_file.write("Brunet's index statistic: " + "Precision: " + str(P_R) + " Recall: " + str(R_R) + " F-measure: "
                   + str(F_R) + "\n")
    log_file.write("Honore's Statistic statistic: " + "Precision: " + str(P_W) + " Recall: " + str(R_W) + " F-measure: "
                   + str(F_W) + "\n")
    log_file.write("Noun measure statistic: " + "Precision: " + str(P_nouns) + " Recall: " + str(R_nouns) +
                   " F-measure: " + str(F_nouns) + "\n")
    log_file.write("Verb measure statistic: " + "Precision: " + str(P_verbs) + " Recall: " + str(R_verbs) +
                   " F-measure: " + str(F_verbs) + "\n")
    log_file.write("Adjective measure statistic: " + "Precision: " + str(P_adj) + " Recall: " + str(R_adj) +
                   " F-measure: " + str(F_adj) + "\n")
    log_file.write("Adverb measure statistic: " + "Precision: " + str(P_adv) + " Recall: " + str(R_adv) +
                   " F-measure: " + str(F_adv) + "\n")
    log_file.write("Noun/Verb Ratio statistic: " + "Precision: " + str(P_noun_verb) + " Recall: " + str(R_noun_verb) +
                   " F-measure: " + str(F_noun_verb) + "\n")
    log_file.write("Pronoun/Verb Ratio statistic: " + "Precision: " + str(R_pronoun_verb) + " Recall: " +
                   str(R_pronoun_verb) + " F-measure: " + str(F_pronoun_verb) + "\n")
    log_file.write("Pronoun/Noun statistic: " + "Precision: " + str(P_pronoun_noun) + " Recall: " + str(R_pronoun_noun)
                   + " F-measure: " + str(F_pronoun_noun) + "\n")


    print(" Exit calculate_statistics")
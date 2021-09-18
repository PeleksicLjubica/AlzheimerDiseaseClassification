import pathlib
from utilities import *
import re


def average(lst):
    return sum(lst) / len(lst)


def find_av_statistic():

    TTR_P = []
    TTR_R = []
    TTR_F = []
    R_P = []
    R_R = []
    R_F = []
    H_P = []
    H_R = []
    H_F = []
    N_P = []
    N_R = []
    N_F = []
    V_P = []
    V_R = []
    V_F = []
    ADJ_P = []
    ADJ_R = []
    ADJ_F = []
    ADV_P = []
    ADV_R = []
    ADV_F = []
    NV_P = []
    NV_R = []
    NV_F = []
    PV_P = []
    PV_R = []
    PV_F = []
    PN_P = []
    PN_R = []
    PN_F = []

    for path in pathlib.Path(PATH_TO_STATISTIC_FOLDER).iterdir():
        if path.is_file():
            # Open next file in the folder
            file = open(path, encoding="utf8")
            if ".DS_Store" not in path.parts:
                lines = file.readlines()

                line = lines[9]
                match_obj = re.match(STATISTICS_REGEX, line, re.M | re.I)
                if match_obj:
                    TTR_P.append(float(match_obj.group(1)))
                    TTR_R.append(float(match_obj.group(2)))
                    TTR_F.append(float(match_obj.group(3)))

                line = lines[10]
                match_obj = re.match(STATISTICS_REGEX, line, re.M | re.I)
                if match_obj:
                    R_P.append(float(match_obj.group(1)))
                    R_R.append(float(match_obj.group(2)))
                    R_F.append(float(match_obj.group(3)))

                line = lines[11]
                match_obj = re.match(STATISTICS_REGEX, line, re.M | re.I)
                if match_obj:
                    H_P.append(float(match_obj.group(1)))
                    H_R.append(float(match_obj.group(2)))
                    H_F.append(float(match_obj.group(3)))

                line = lines[12]
                match_obj = re.match(STATISTICS_REGEX, line, re.M | re.I)
                if match_obj:
                    N_P.append(float(match_obj.group(1)))
                    N_R.append(float(match_obj.group(2)))
                    N_F.append(float(match_obj.group(3)))

                line = lines[13]
                match_obj = re.match(STATISTICS_REGEX, line, re.M | re.I)
                if match_obj:
                    V_P.append(float(match_obj.group(1)))
                    V_R.append(float(match_obj.group(2)))
                    V_F.append(float(match_obj.group(3)))

                line = lines[14]
                match_obj = re.match(STATISTICS_REGEX, line, re.M | re.I)
                if match_obj:
                    ADJ_P.append(float(match_obj.group(1)))
                    ADJ_R.append(float(match_obj.group(2)))
                    ADJ_F.append(float(match_obj.group(3)))

                line = lines[15]
                match_obj = re.match(STATISTICS_REGEX, line, re.M | re.I)
                if match_obj:
                    ADV_P.append(float(match_obj.group(1)))
                    ADV_R.append(float(match_obj.group(2)))
                    ADV_F.append(float(match_obj.group(3)))

                line = lines[16]
                match_obj = re.match(STATISTICS_REGEX, line, re.M | re.I)
                if match_obj:
                    NV_P.append(float(match_obj.group(1)))
                    NV_R.append(float(match_obj.group(2)))
                    NV_F.append(float(match_obj.group(3)))

                line = lines[17]
                match_obj = re.match(STATISTICS_REGEX, line, re.M | re.I)
                if match_obj:
                    PV_P.append(float(match_obj.group(1)))
                    PV_R.append(float(match_obj.group(2)))
                    PV_F.append(float(match_obj.group(3)))

                line = lines[18]
                match_obj = re.match(STATISTICS_REGEX, line, re.M | re.I)
                if match_obj:
                    PN_P.append(float(match_obj.group(1)))
                    PN_R.append(float(match_obj.group(2)))
                    PN_F.append(float(match_obj.group(3)))

    print("TTR P: " + str(average(TTR_P))+ " TTR R: " + str(average(TTR_R))+ " TTR F: " + str(average(TTR_F)))
    print("R P: " + str(average(R_P))+ " R R: " + str(average(R_R))+ " R F: " + str(average(R_F)))
    print("H P: " + str(average(H_P))+ " H R: " + str(average(H_R))+ " H F: " + str(average(H_F)))
    print("N P: " + str(average(N_P))+ " N R: " + str(average(N_R))+ " N F: " + str(average(N_F)))
    print("V P: " + str(average(V_P))+ " V R: " + str(average(V_R))+ " V F: " + str(average(V_F)))
    print("ADJ P: " + str(average(ADJ_P))+ " ADJ R: " + str(average(ADJ_R))+ " ADJ F: " + str(average(ADJ_F)))
    print("ADV P: " + str(average(ADV_P))+ " ADV R: " + str(average(ADV_R))+ " ADV F: " + str(average(ADV_F)))
    print("NV P: " + str(average(NV_P))+ " NV R: " + str(average(NV_R))+ " NV F: " + str(average(NV_F)))
    print("PV P: " + str(average(PV_P))+ " PV R: " + str(average(PV_R))+ " PV F: " + str(average(PV_F)))
    print("PN P: " + str(average(PN_P))+ " PN R: " + str(average(PN_R))+ " PN F: " + str(average(PN_F)))


def find_all_train_statistics(measure: Measurement, train_data, path_to_folder):
    """
    Function to collect specific measurement for all train data in path_to_folder
    :param measure: Type of measurement that are collected data for
    :param train_data: Array where data for measurement is stored
    :param path_to_folder: Path to folder where data is located
    :return: None
    """

    for path in pathlib.Path(path_to_folder).iterdir():
        if path.is_file():
            # Open next file in the folder
            file = open(path, encoding="utf8")
            if ".DS_Store" not in path.parts:
                lines = file.readlines()
                line = lines[measure.value]

                match_obj = re.match(REGEX_READ_FILE, line, re.M | re.I)
                if match_obj:
                    train_data.append(float(match_obj.group(1)))

    return train_data


def find_all_test_statistics(measure: Measurement, class_clasif: ClassificationClass, path_to_folder,
                             positive_train_data, negative_train_data):

    """
    Function to collect specific measurement for all test data and calculate True Positives, True Negatives, False
    Positives and False Negatives for every document in test corpus
    :param measure: Type of measurement that are collected data for
    :param class_clasif: Type of class, POSITIVE or NEGATIVE
    :param path_to_folder: Path to folder where data is located
    :param positive_train_data: Array of positive training data collected
    :param negative_train_data: Array of negative training data collected
    :return: Array containing two values depending on class
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

                    distance_from_document_to_n = distance_from_document_to_n / len(negative_train_data)
                    distance_from_document_to_p = distance_from_document_to_p / len(positive_train_data)

                    if class_clasif == ClassificationClass.POSITIVE:
                        if distance_from_document_to_p < distance_from_document_to_n:
                            TruePositives += 1
                        else:
                            FalseNegatives += 1
                    else:
                        if distance_from_document_to_p < distance_from_document_to_n:
                            FalsePositives += 1
                        else:
                            TrueNegatives += 1

    if class_clasif == ClassificationClass.POSITIVE:
        print("TruePositives: " + str(TruePositives) + " FalseNegatives: " + str(FalseNegatives))
        return [TruePositives, FalseNegatives]
    elif class_clasif == ClassificationClass.NEGATIVE:
        print("TrueNegatives: " + str(TrueNegatives) + " FalsePositives: " + str(FalsePositives))
        return [TrueNegatives, FalsePositives]


def calculate_statistics_for_measure(measure: Measurement, path_to_train_1, path_to_train_2, path_to_train_3,
                                     path_to_train_4, path_to_test_1, path_to_test_2):
    """
    Function to collect train statistics and
    :param measure: Type of measurement that are collected data for
    :param path_to_train_1: Path to training folder
    :param path_to_train_2: Path to training folder
    :param path_to_train_3: Path to training folder
    :param path_to_train_4: Path to training folder
    :param path_to_test_1: Path to test folder
    :param path_to_test_2: Path to test folder
    :return: values for precision, recall and f-measure
    """
    print(" Enter calculate_statistics_for_measure")

    # Collect data for specific measure for training corpus
    positive_train_data = []
    negative_train_data = []

    positive_train_data = find_all_train_statistics(measure, positive_train_data, path_to_train_1)
    positive_train_data = find_all_train_statistics(measure, positive_train_data, path_to_train_2)
    negative_train_data = find_all_train_statistics(measure, negative_train_data, path_to_train_3)
    negative_train_data = find_all_train_statistics(measure, negative_train_data, path_to_train_4)

    [TruePositives, FalseNegatives] = find_all_test_statistics(measure, ClassificationClass.POSITIVE,
                                                               path_to_test_1, positive_train_data, negative_train_data)

    [TrueNegatives, FalsePositives] = find_all_test_statistics(measure, ClassificationClass.NEGATIVE,
                                                   path_to_test_2, positive_train_data, negative_train_data)

    print(str(measure) + " True Positives: " + str(TruePositives) + " FalsePositives: " + str(FalsePositives) + " FalseNegatives: " +
          str(FalseNegatives) + "True Negatives: " + str(TrueNegatives))

    print("True: " + str(TruePositives + TrueNegatives) + " False: " + str(FalsePositives + FalseNegatives) )

    P = 0
    R = 0
    F = 0
    if TruePositives + FalsePositives != 0:
        P = TruePositives / (TruePositives + FalsePositives)
    else:
        x = 0
    if TruePositives + FalseNegatives != 0:
        R = TruePositives / (TruePositives + FalseNegatives)
    else:
        x = 0

    if P+R != 0:
        F = (2 * P * R) / (P + R)
    else:
        x = 0

    print("P: " + str(P) + " R: " + str(R) + " F: " + str(F))

    print(" Exit calculate_statistics_for_measure")
    return P, R, F


def calculate_statistics(path_to_train_1, path_to_train_2, path_to_train_3, path_to_train_4, path_to_test_1,
                         path_to_test_2, path_to_statistic_folder, should_calculate_avg_statistics):
    """
    Function to call statistic calculation for all measures and save calculations to log files
    :param path_to_statistic_folder:
    :param should_calculate_avg_statistics:
    :param path_to_train_1: Path to training folder
    :param path_to_train_2: Path to training folder
    :param path_to_train_3: Path to training folder
    :param path_to_train_4: Path to training folder
    :param path_to_test_1: Path to test folder
    :param path_to_test_2: Path to test folder
    :param log_file_name: Path with file name for log file
    :return: None
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

    log_file = open(path_to_statistic_folder, mode="w+", encoding="utf8")
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

    log_file.close()

    if should_calculate_avg_statistics:
       find_av_statistic()

    print(" Exit calculate_statistics")
"""
Module for plotting.
Containing different functions to call pyplot and draw them
"""

import matplotlib.pyplot as plt


# Function for plotting two plots on the same image
def plot_lexical_analysis_results_two_plots(fp_result_p, fp_range, fp_result_n, fp_desc, sp_result_p,
                                            sp_result_p_mean_value, sp_range, sp_result_n, sp_result_n_mean_value,
                                            sp_desc):
    plt.figure(1)
    plt.subplot(211)
    tmp_list1 = [1] * len(fp_result_p)
    plt.plot(fp_result_p, tmp_list1, "ro", label='Positive')
    tmp_list2 = [2] * len(fp_result_n)
    plt.plot(fp_result_n, tmp_list2, 'bo', label='Negative')
    plt.xlabel(fp_desc)
    plt.axis([0, fp_range, 0, 3])
    plt.yticks([])
    plt.legend()

    plt.subplot(212)
    tmp_list = [1] * len(sp_result_p)
    plt.plot(sp_result_p, tmp_list, "ro", label='Positive')
    plt.plot(sp_result_p_mean_value, [1], "go", label='Positive mean value')

    tmp_list = [2] * len(sp_result_n)
    plt.plot(sp_result_n, tmp_list, "bo", label='Negative')
    plt.plot(sp_result_n_mean_value, [2], 'mo', label='Negative mean value')

    plt.xlabel(sp_desc)
    plt.axis([0, sp_range, 0, 3])
    plt.yticks([])
    plt.legend()

    plt.show()


# Function for plotting one plot in the image
def plot_lexical_analysis_results_one_plot(fp_result_p, fp_mean_p, fp_range, fp_result_n, fp_mean_n, fp_desc):
    tmp_list1 = [1] * len(fp_result_p)
    plt.plot(fp_result_p, tmp_list1, "ro", label='Positive')
    plt.plot(fp_mean_p, [1], "go", label='Positive mean value')

    tmp_list2 = [2] * len(fp_result_n)
    plt.plot(fp_result_n, tmp_list2, 'bo', label='Negative')
    plt.plot(fp_mean_n, [2], 'mo', label='Negative mean value')

    plt.xlabel(fp_desc)
    plt.axis([0, fp_range, 0, 3])
    plt.yticks([])
    plt.legend()
    plt.show()

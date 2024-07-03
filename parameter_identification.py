import os
from typing import Optional, Tuple
import numpy as np
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)


def extract_hyp_parameters(files_ls: list[str]) -> Tuple[np.array, list[str]]:
    """
    Extracts the different hyperparameter values used for testing
    :param files_ls: list of files in testing dir
    :return: As array and also as strings for later labelling
    """
    files_ls = [x for x in files_ls if "spatially_adaptive" not in x]
    files_ls = [x for x in files_ls if "plots" not in x]
    tmp = [i.split("/")[-1] for i in files_ls]
    hyp_i_ls = np.unique(np.array([round(float(i), 10) for i in tmp]))
    return hyp_i_ls, tmp


def extract_dice_scores_from_files(files_ls: list) -> list[float]:
    """
    Extracts the dice scores from the log files
    :param files_ls: list of log files from testing
    :return: list of dice scores across different elasticity parameters
    """

    dice_mean_ls = list()
    for file_i in files_ls:
        mean_file = file_i + "/testing_" + (file_i.split("/")[-1]) + "_mean.txt"
        with open(mean_file) as file:
            info = ""
            for line in file:
                info += line.rstrip()

        dice_i = ((info.split(",")[0]).split(":")[1]).strip()
        dice_mean_ls.append(float(dice_i))
    return dice_mean_ls


def extract_dice_per_class_scores_from_files(files_ls: list) -> list[list[float]]:
    """
    Extracts the dice scores from the log files
    :param files_ls: list of log files from testing
    :return: list of dice scores across different elasticity parameters
    """

    dice_mean_ls = list()
    for file_i in files_ls:
        mean_file = file_i + "/testing_" + (file_i.split("/")[-1]) + "_mean.txt"
        with open(mean_file) as file:
            info = ""
            for line in file:
                info += line.rstrip()

        dice_i = ((info.split(",")[1]).split(":")[1]).strip()[1:-1]
        tmp = [float(x) for x in dice_i.split()]
        dice_mean_ls.append(tmp)
    return dice_mean_ls


def extract_elasticity_parameters(files_ls: list[str]) -> Tuple[np.array, np.array, list[str], list[str]]:
    """
    first two: unique, float. second two: all repeated, str
    """
    lambda_i_ls_str = [(i.split("/")[-1]).split("_")[0] for i in files_ls]
    mu_i_ls_str = [(i.split("/")[-1]).split("_")[1] for i in files_ls]
    lambda_i_ls = np.unique(np.array([round(float(i), 10) for i in lambda_i_ls_str]))
    mu_i_ls = np.unique(np.array([round(float(i), 10) for i in mu_i_ls_str]))
    return lambda_i_ls, mu_i_ls, lambda_i_ls_str, mu_i_ls_str


def extract_tre_from_files(files_ls: list) -> list[float]:
    """
    Extracts the dice scores from the log files
    :param files_ls: list of log files from testing
    :return: list of dice scores across different elasticity parameters
    """

    tre_ls = list()
    for file_i in files_ls:
        mean_file = file_i + "/testing_" + (file_i.split("/")[-1]) + "_mean.txt"
        with open(mean_file) as file:
            info = ""
            for line in file:
                info += line.rstrip()

        tre_i = ((info.split(":")[-1]).split("-")[0]).strip()
        tre_ls.append(float(tre_i))
    return tre_ls


def find_best_hyp_params_global_diffusion(modelpath: str, greater_zero: Optional[bool] = True, metric="DSC") -> tuple[float, float]:
    """
    Finds best suitable hyp param according to metric during testing
    :param modelpath:
    :return: [hyp_class1, hyp_class2,...]
    """
    testing_dir = modelpath + "/testing"
    files_ls = [x[0] for x in sorted(os.walk(testing_dir))][1:]

    # extracting dice scores
    if metric == "TRE":
        dice_mean_ls = extract_tre_from_files(files_ls)
    if metric == "DSC":
        dice_mean_ls = extract_dice_scores_from_files(files_ls)
    hyp_params = extract_hyp_parameters(files_ls)

    # find best hyp_val which leads to highest dice score per class
    dice_mean = np.array(dice_mean_ls)
    if not greater_zero:
        hyp_params_ls = hyp_params[0]
    else:
        dice_mean = dice_mean[1:]  # reg weight = 0 not acceptable, must be positive
        hyp_params_ls = hyp_params[0][1:]

    if metric == "DSC":
        dice_mean_max = np.argmax(dice_mean, axis=0)
        hyp_max = hyp_params_ls[dice_mean_max]
        metric_max = dice_mean[dice_mean_max]
    if metric == "TRE":
        tre_mean_max = np.argmin(dice_mean, axis=0)
        hyp_max = hyp_params_ls[tre_mean_max]
        metric_max = dice_mean[tre_mean_max]
    print("Best hyper parameters extracted: ", hyp_max)
    return hyp_max, metric_max


def find_best_hyp_params_per_class_diffusion(modelpath: str, greater_zero: Optional[bool] = True, metric="DSC", background_reg_val=0) -> tuple[list, list]:
    """
    Finds best suitable hyp param according to dice scores during testing for each class
    :param modelpath:
    :return: [hyp_class1, hyp_class2,...]
    """
    testing_dir = modelpath + "/testing"
    files_ls = [x[0] for x in sorted(os.walk(testing_dir))][1:]

    # extracting dice scores
    if metric == "TRE":
        dice_mean_ls = extract_tre_from_files(files_ls)
    if metric == "DSC":
        dice_mean_ls = extract_dice_per_class_scores_from_files(files_ls)
    hyp_params = extract_hyp_parameters(files_ls)

    # find best hyp_val which leads to highest dice score per class
    dice_mean = np.array(dice_mean_ls)
    if not greater_zero:
        hyp_params_ls = hyp_params[0]
    else:
        dice_mean = dice_mean[1:]  # reg weight = 0 not acceptable, must be positive
        hyp_params_ls = hyp_params[0][1:]

    if metric == "DSC":
        dice_mean_max = np.argmax(dice_mean, axis=0)
        hyp_sa = [hyp_params_ls[i] for i in dice_mean_max]
    if metric == "TRE":
        dice_mean_max = np.argmin(dice_mean, axis=0)
        hyp_sa = [hyp_params_ls[i] for i in dice_mean_max]

    hyp_sa[0] = background_reg_val  # background should be maximally regularized per default (?)

    print("Best hyper parameters extracted: ", hyp_sa)
    return hyp_sa, np.diag(dice_mean[dice_mean_max])


def find_best_hyp_params_per_class_diffusion_sa(modelpath : str, greater_zero:Optional[bool]=True, background_reg_val=0) -> list:
    """
    Finds best suitable hyp param according to dice scores during testing for each class
    :param modelpath:
    :return: [hyp_class1, hyp_class2,...]
    """
    testing_dir = '/'.join(modelpath.split('/')[:-1]) + "/testing"
    files_ls = [x[0] for x in sorted(os.walk(testing_dir))][1:]

    # extracting dice scores
    dice_mean_ls = extract_dice_per_class_scores_from_files(files_ls)
    hyp_params = extract_hyp_parameters(files_ls)

    # find best hyp_val which leads to highest dice score per class
    dice_mean = np.array(dice_mean_ls)
    if not greater_zero:
        dice_mean_max= np.argmax(dice_mean, axis=0)
    else:
        dice_mean_max = np.argmax(dice_mean[1:,:], axis=0) # reg weight = 0 not acceptable, must be positive
        dice_mean_max+=1

    hyp_sa = [hyp_params[0][i] for i in dice_mean_max]
    hyp_sa[0]=background_reg_val # background should be maximally regularized per default (?)

    print("Best hyper parameters extracted: ", hyp_sa)
    return hyp_sa


def find_best_hyp_params_per_class_elastic(modelpath:str, greater_zero:Optional[bool]=True, metric="DSC", background_val=[0,0]):
    testing_dir = modelpath + "/testing"

    files_ls = [x[0] for x in sorted(os.walk(testing_dir))][1:]
    if testing_dir + '/plots' in files_ls: files_ls.remove(testing_dir + '/plots')
    files_ls = [x for x in files_ls if "spatially_adaptive" not in x]

    if metric == "DSC":
        dice_mean_ls = extract_dice_per_class_scores_from_files(files_ls)
        _,_, lambda_i_ls_str, mu_i_ls_str = extract_elasticity_parameters(files_ls)
    if metric == "TRE":
        dice_mean_ls = extract_tre_from_files(files_ls)
        _,_, lambda_i_ls_str, mu_i_ls_str = extract_elasticity_parameters(files_ls)

    if greater_zero:
        zero_idx = [i for i, (x, y) in enumerate(zip(lambda_i_ls_str, mu_i_ls_str)) if x == '0.0' and y == '0.0'][0]
        if metric == "TRE":
            dice_mean_ls[zero_idx] = len(dice_mean_ls[-1])*[100]
        if metric == "DSC":
            dice_mean_ls[zero_idx] = len(dice_mean_ls[-1])*[0]

    dice_arr = np.array(dice_mean_ls)
    hyp_sa=list()
    max_eval=list()

    if metric == "DSC":
        argmax=np.argmax(dice_arr, axis=0)
        for i in argmax:
            hyp_sa.append([float(lambda_i_ls_str[i]), float(mu_i_ls_str[i])])
            max_eval.append(dice_mean_ls[i])
    hyp_sa=np.array(hyp_sa)
    n_classes=hyp_sa.shape[0]
    hyp_sa_lambda=hyp_sa[:,0]
    hyp_sa_mu=hyp_sa[:,1]
    hyp_sa=np.concatenate((hyp_sa_lambda, hyp_sa_mu))

    hyp_sa[0]=background_val[0] # background maximally regularized
    hyp_sa[n_classes]=background_val[1] # background maximally regularized
    return hyp_sa, np.diag(max_eval)


def find_best_hyp_params_global_elastic(modelpath:str, greater_zero:Optional[bool]=True, metric="DSC", background_val=[0,0]):
    testing_dir = modelpath + "/testing"
    files_ls = [x[0] for x in sorted(os.walk(testing_dir))][1:]

    # extracting dice scores
    if metric == "TRE":
        dice_mean_ls = extract_tre_from_files(files_ls)
    if metric == "DSC":
        dice_mean_ls = extract_dice_scores_from_files(files_ls)
    _,_, lambda_i_ls_str, mu_i_ls_str = extract_elasticity_parameters(files_ls)

    dice_arr = np.array(dice_mean_ls)

    if greater_zero:
        zero_idx = [i for i, (x, y) in enumerate(zip(lambda_i_ls_str, mu_i_ls_str)) if x == '0.0' and y == '0.0'][0]
        if metric == "TRE":
            dice_arr[zero_idx]=100
        if metric=="DSC":
            dice_arr[zero_idx]=0

    hyp_sa=list()

    if metric == "DSC":
        argmax=np.argmax(dice_arr, axis=0)
        hyp_sa.append([float(lambda_i_ls_str[argmax]), float(mu_i_ls_str[argmax])])
        max_eval= dice_mean_ls[argmax]
    if metric == "TRE":
        argmax = np.argmin(dice_arr, axis=0)
        hyp_sa.append([float(lambda_i_ls_str[argmax]), float(mu_i_ls_str[argmax])])
        max_eval= dice_mean_ls[argmax]
    return hyp_sa, max_eval


def find_best_hyp_params_per_class_elastic_sa(modelpath : str) -> list:
    """
    Finds best suitable hyp param according to dice scores during testing for each class
    :param modelpath:
    :return: [[mu_class1, mu_class2,...],[lamda_class1, lambda_class2,...]]
    """
    testing_dir = '/'.join(modelpath.split('/')[:-1]) + "/testing"
    files_ls = [x[0] for x in sorted(os.walk(testing_dir))][1:]

    # extracting dice scores
    dice_mean_ls = extract_dice_per_class_scores_from_files(files_ls)
    hyp_params = extract_elasticity_parameters(files_ls)

    # find best hyp_val which leads to highest dice score per class
    dice_mean = np.array(dice_mean_ls)
    dice_mean_max= np.argmax(dice_mean, axis=0)

    hyp_sa = [hyp_params[0][i] for i in dice_mean_max]
    hyp_sa[0]=1

    return hyp_sa

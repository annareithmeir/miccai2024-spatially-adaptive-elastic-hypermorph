import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import voxelmorph as vxm

import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False) # to suppress warnings

import parameter_identification
import datasets
import utils


def inference__with_best_hyp(modelpath: str, test_input: list[np.array], best_params: list[float]):
    """
    perform inference of a test sample from the test scan_to_scan_generator with the trained model and the identified best hyper parameters

    """

    model = vxm.networks.HyperVxmDense.load(modelpath, input_model=None)
    pred = model.predict([test_input[0], test_input[1], best_params])

    warped = pred[0]
    warp = pred[1]
    return warped, warp


def overall_pipeline():

    model_type= "diffsuion" # diffusion/elastic

    global_model_path = "results/LungCTCT/model_final_d"
    local_model_path = "results/LungCTCT/model_final_d_sa_vec"
    background_reg_val = 0

    files_path = "datasets/LungCT"
    test_dataset = datasets.Learn2RegLungCTDataset(files_path, mode="test", seg_mode="LungRibsLiver",
                                                   normalize_mode=True)
    base_generator = utils.scan_to_scan_generator(test_dataset, batch_size=1)
    test_generator = utils.hyp_generator(1, base_generator, test_dataset.return_mode)
    test_input, _ = next(test_generator)

    if model_type == "diffusion":
        print("####### DIFFUSION ########")
        hyp, max_eval = parameter_identification.find_best_hyp_params_per_class_diffusion(global_model_path, greater_zero=True,
                                                                                 metric="DSC", background_reg_val=background_reg_val)
        print("Best class-wise hyps found with global model are: ", hyp, " with DSC=", max_eval)
        print(
            "---------------------------------------------------------------------------------------------------------")
        hyp_global, eval_global = parameter_identification.find_best_hyp_params_global_diffusion(global_model_path,
                                                                                        greater_zero=True, metric="DSC")
        print("Best global hyp found with global model are: ", hyp_global, " with DSC=", eval_global)
        print(
            "---------------------------------------------------------------------------------------------------------")

    elif model_type == "elastic":
        print("####### ELASTIC ########")
        hyp, max_eval = parameter_identification.find_best_hyp_params_per_class_elastic(global_model_path,
                                                                               greater_zero=True, metric="DSC",
                                                                               background_val=background_reg_val)
        print("Best class-wise hyps found with global model are: ", hyp, " with DSC=", max_eval)
        print(
            "---------------------------------------------------------------------------------------------------------")
        hyp_global, max_eval_global = parameter_identification.find_best_hyp_params_global_elastic(global_model_path,
                                                                                          greater_zero=True,
                                                                                          metric="DSC")
        print("Best global hyps [lambda_i, ..., mu_i,...] found with global model are: ", hyp_global, " with DSC=",
              max_eval_global)
        print(
            "---------------------------------------------------------------------------------------------------------")

    else:
        print("Not implemented")

    inference__with_best_hyp(local_model_path, test_input, hyp_global)


# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Evaluation script for the BOP Challenge 2019/2020."""

import os
import time
import argparse
import subprocess
import numpy as np

from bop_toolkit_lib import config
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc

import pickle
# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
p = {
    # Errors to calculate.
    "errors": [
        # {
        #     "n_top": -1,
        #     "type": "ad",
        #     "correct_th": [[th] for th in np.arange(1, 101, 1)],
        # },
        # {
        #     "n_top": -1,
        #     "type": "adi",
        #     "correct_th": [[th] for th in np.arange(1, 101, 1)],
        # },
        # {
        #     "n_top": -1,
        #     "type": "add",
        #     "correct_th": [[th] for th in np.arange(1, 101, 1)],
        # },
        {
            "n_top": -1,
            "type": "te",
            "correct_th": [[th] for th in np.arange(1, 10, 1)],
        },
        {
            "n_top": -1,
            "type": "re",
            "correct_th": [[th] for th in np.arange(1, 10, 1)],
        },
        # {
        #     # "n_top": -1,
        #     "n_top": 0,
        #     "type": "vsd",
        #     "vsd_deltas": {
        #         "hb": 15,
        #         "icbin": 15,
        #         "icmi": 15,
        #         "itodd": 5,
        #         "lm": 15,
        #         "lmo": 15,
        #         "ruapc": 15,
        #         "tless": 15,
        #         "tudl": 15,
        #         "tyol": 15,
        #         "ycbv": 15,
        #         "hope": 15,
        #         "hopeVideo":15,
        #         "SynthStatic":15,
        #         "SynthDynamic":15,
        #         "SynthStaticDummy":15
        #     },
        #     "vsd_taus": list(np.arange(0.05, 0.51, 0.05)),
        #     "vsd_normalized_by_diameter": True,
        #     "correct_th": [[th] for th in np.arange(0.05, 0.51, 0.05)],
        # },
        # {
        #     # "n_top": -1,
        #     "n_top": 0,
        #     "type": "mssd",
        #     "correct_th": [[th] for th in np.arange(0.05, 0.51, 0.05)],
        # },
        # {
        #     # "n_top": -1,
        #     "n_top": 0,
        #     "type": "mspd",
        #     "correct_th": [[th] for th in np.arange(5, 51, 5)],
        # },
    ],
    # Minimum visible surface fraction of a valid GT pose.
    # -1 == k most visible GT poses will be considered, where k is given by
    # the "inst_count" item loaded from "targets_filename".
    "visib_gt_min": -1,
    # See misc.get_symmetry_transformations().
    "max_sym_disc_step": 0.01,
    # Type of the renderer (used for the VSD pose error function).
    "renderer_type": "vispy",  # Options: 'vispy', 'cpp', 'python'.
    # Names of files with results for which to calculate the errors (assumed to be
    # stored in folder p['results_path']). See docs/bop_challenge_2019.md for a
    # description of the format. Example results can be found at:
    # https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019_sample_results.zip
    "result_filenames": [

        # "/media/vojta/Data/HappyPose_Data/bop_datasets/SynthStatic/cosypose_SynthStatic-test.csv",
        # "/media/vojta/Data/HappyPose_Data/bop_datasets/hopeVideo/cosypose_hopeVideo-test.csv",
        # "/media/vojta/Data/HappyPose_Data/bop_datasets/SynthStatic/gtsam_SynthStatic-test.csv",
        # "/media/vojta/Data/HappyPose_Data/bop_datasets/hopeVideo/cosypose_hopeVideo-test.csv",
        # "/media/vojta/Data/HappyPose_Data/bop_datasets/hopeVideo/gtsam_hopeVideo-test.csv",

    ],
    # Folder with results to be evaluated.
    "results_path": config.results_path,
    # Folder for the calculated pose errors and performance scores.
    "eval_path": config.eval_path,
    # File with a list of estimation targets to consider. The file is assumed to
    # be stored in the dataset folder.
    "targets_filename": "test_targets_bop19.json",
}
# DATASET_NAME = "hopeVideo"
# # DATASET_NAME = "SynthStatic"
# results_dir = f"/media/vojta/Data/HappyPose_Data/bop_datasets/{DATASET_NAME}"
# for file_name in os.listdir(results_dir):
#     if f"gtsam_{DATASET_NAME}-test_" in file_name:
#         p["result_filenames"].append(results_dir + "/" + file_name)

################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--renderer_type", default=p["renderer_type"])
parser.add_argument(
    "--result_filenames",
    default=",".join(p["result_filenames"]),
    help="Comma-separated names of files with results.",
)
parser.add_argument("--results_path", default=p["results_path"])
parser.add_argument("--eval_path", default=p["eval_path"])
parser.add_argument("--targets_filename", default=p["targets_filename"])
args = parser.parse_args()

p["renderer_type"] = str(args.renderer_type)
p["result_filenames"] = args.result_filenames.split(",")
p["results_path"] = str(args.results_path)
p["eval_path"] = str(args.eval_path)
p["targets_filename"] = str(args.targets_filename)

with open(os.path.join(p["eval_path"], "all_results.txt"), "w") as file:
    pass
with open(os.path.join(p["eval_path"], "all_results_info.txt"), "w") as file:
    pass

# Evaluation.
# ------------------------------------------------------------------------------
# # precision_dict = {}
# # recall_dict = {}
# for ortt in [1e-4, 1e-3, 1e-2, 1e-1, 1., 2., 3., 5., 10.]:
#     for ortr in [1e-4, 1e-3, 1e-2, 1e-1, 1., 2., 3., 5., 10.]:
#         try:
#             # tsamsearch-parameters_ycbv-test_cosy_synt_real_0.0_threshold_noreject_1_0_1.0_1e-08_1e-07_0.1_0.17453292519943295_2.00E+00_2.00E+00.csv
#             # result_filename = f'/home/ros/kzorina/vojtas/ycbv/ablation/gtsamsearch-parameters_ycbv-test_cosy_synt_real_0.0_threshold_noreject_1_0_1.0_1e-08_1e-07_0.1_0.17453292519943295_{tvt:.2E}_{rvt:.2E}.csv'
#             result_filename = f'/home/ros/kzorina/vojtas/ycbv/ablation/gtsamsearch-parameters2_ycbv-test_cosy_synt_real_0.0_threshold_noreject_1_0_1.0_1e-08_1e-07_{ortt}_{ortr}_1.00E+00_1.00E+00.csv'
for result_filename in p["result_filenames"]:

    # do bop
    misc.log("===========")
    misc.log("EVALUATING: {}".format(result_filename))
    misc.log("===========")

    time_start = time.time()

    # Volume under recall surface (VSD) / area under recall curve (MSSD, MSPD).
    average_recalls = {}
    average_precisions = {}

    # Name of the result and the dataset. (blabla_dataset-blabla)
    result_name = os.path.splitext(os.path.basename(result_filename))[0]
    dataset = str(result_name.split("_")[1].split("-")[0])

    # Calculate the average estimation time per image.
    ests = inout.load_bop_results(
        os.path.join(p["results_path"], result_filename), version="bop19"
    )
    times = {}
    times_available = True
    for est in ests:
        # if est["scene_id"] != 48: continue  # TODO: remove!!
        result_key = "{:06d}_{:06d}".format(est["scene_id"], est["im_id"])
        if est["time"] < 0:
            # All estimation times must be provided.
            times_available = False
            break
        elif result_key in times:
            if abs(times[result_key] - est["time"]) > 0.001:
                raise ValueError(
                    "The running time for scene {} and image {} is not the same for "
                    "all estimates.".format(est["scene_id"], est["im_id"])
                )
        else:
            times[result_key] = est["time"]

    if times_available:
        average_time_per_image = np.mean(list(times.values()))
    else:
        average_time_per_image = -1.0

    # Evaluate the pose estimates.
    for error in p["errors"]:
        # Calculate error of the pose estimates.
        calc_errors_cmd = [
            "python",
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "eval_calc_errors.py"
            ),
            "--n_top={}".format(error["n_top"]),
            "--error_type={}".format(error["type"]),
            "--result_filenames={}".format(result_filename),
            "--renderer_type={}".format(p["renderer_type"]),
            "--results_path={}".format(p["results_path"]),
            "--eval_path={}".format(p["eval_path"]),
            "--targets_filename={}".format(p["targets_filename"]),
            "--max_sym_disc_step={}".format(p["max_sym_disc_step"]),
            "--skip_missing=1",
        ]
        if error["type"] == "vsd":
            vsd_deltas_str = ",".join(
                ["{}:{}".format(k, v) for k, v in error["vsd_deltas"].items()]
            )
            calc_errors_cmd += [
                "--vsd_deltas={}".format(vsd_deltas_str),
                "--vsd_taus={}".format(",".join(map(str, error["vsd_taus"]))),
                "--vsd_normalized_by_diameter={}".format(
                    error["vsd_normalized_by_diameter"]
                ),
            ]

        misc.log("Running: " + " ".join(calc_errors_cmd))
        if subprocess.call(calc_errors_cmd) != 0:
            raise RuntimeError("Calculation of pose errors failed.")

        # Paths (rel. to p['eval_path']) to folders with calculated pose errors.
        # For VSD, there is one path for each setting of tau. For the other pose
        # error functions, there is only one path.
        error_dir_paths = {}
        if error["type"] == "vsd":
            for vsd_tau in error["vsd_taus"]:
                error_sign = misc.get_error_signature(
                    error["type"],
                    error["n_top"],
                    vsd_delta=error["vsd_deltas"][dataset],
                    vsd_tau=vsd_tau,
                )
                error_dir_paths[error_sign] = os.path.join(result_name, error_sign)
        else:
            error_sign = misc.get_error_signature(error["type"], error["n_top"])
            error_dir_paths[error_sign] = os.path.join(result_name, error_sign)

        # Recall scores for all settings of the threshold of correctness (and also
        # of the misalignment tolerance tau in the case of VSD).
        recalls = []
        precisions = []

        # Calculate performance scores.
        for error_sign, error_dir_path in error_dir_paths.items():
            for correct_th in error["correct_th"]:
                calc_scores_cmd = [
                    "python",
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "eval_calc_scores.py",
                    ),
                    "--error_dir_paths={}".format(error_dir_path),
                    "--eval_path={}".format(p["eval_path"]),
                    "--targets_filename={}".format(p["targets_filename"]),
                    "--visib_gt_min={}".format(p["visib_gt_min"]),
                ]

                calc_scores_cmd += [
                    "--correct_th_{}={}".format(
                        error["type"], ",".join(map(str, correct_th))
                    )
                ]

                misc.log("Running: " + " ".join(calc_scores_cmd))
                if subprocess.call(calc_scores_cmd) != 0:
                    raise RuntimeError("Calculation of scores failed.")

                # Path to file with calculated scores.
                score_sign = misc.get_score_signature(correct_th, p["visib_gt_min"])

                scores_filename = "scores_{}.json".format(score_sign)
                scores_path = os.path.join(
                    p["eval_path"], result_name, error_sign, scores_filename
                )

                # Load the scores.
                misc.log("Loading calculated scores from: {}".format(scores_path))
                scores = inout.load_json(scores_path)
                recalls.append(scores["recall"])
                precisions.append(scores["precision"])

        average_recalls[error["type"]] = np.mean(recalls)
        average_precisions[error["type"]] = np.mean(precisions)

        misc.log("Recall scores: {}".format(" ".join(map(str, recalls))))
        misc.log("Average recall: {}".format(average_recalls[error["type"]]))
        misc.log("precision scores: {}".format(" ".join(map(str, precisions))))
        misc.log("Average precision: {}".format(average_precisions[error["type"]]))

    time_total = time.time() - time_start
    misc.log("Evaluation of {} took {}s.".format(result_filename, time_total))

    # Calculate the final scores.
    final_scores = {}
    for error in p["errors"]:
        final_scores["bop19_average_recall_{}".format(error["type"])] = average_recalls[
            error["type"]
        ]
        final_scores["bop19_average_precision_{}".format(error["type"])] = average_precisions[
            error["type"]
        ]
    final_scores["bop19_average_recall"] = 0.
    final_scores["bop19_average_precision"] = 0.

    # Final score for the given dataset.
    # final_scores["bop19_average_recall"] = np.mean(
    #     # [average_recalls["ad"], average_recalls["adi"], average_recalls["add"]]
    #     [average_recalls["ad"], average_recalls["adi"]]
    # )
    #
    # final_scores["bop19_average_precision"] = np.mean(
    #     # [average_precisions["ad"], average_precisions["adi"], average_precisions["add"]]
    #     [average_precisions["ad"], average_precisions["adi"]]
    # )
    # final_scores["bop19_average_recall"] = np.mean(
    #     [average_recalls["vsd"], average_recalls["mssd"], average_recalls["mspd"]]
    # )
    #
    # final_scores["bop19_average_precision"] = np.mean(
    #     [average_precisions["vsd"], average_precisions["mssd"], average_precisions["mspd"]]
    # )

    # Average estimation time per image.
    final_scores["bop19_average_time_per_image"] = average_time_per_image

    # Save the final scores.
    final_scores_path = os.path.join(p["eval_path"], result_name, "scores_bop19.json")
    inout.save_json(final_scores_path, final_scores)
    with open(os.path.join(p["eval_path"], "all_results.txt"), "a") as file:
        lst = result_filename.split("_")
        # ort = lst[-4]
        # tvt = lst[-3]
        # Rvt = lst[-2]
        # file.write(f"{result_filename}:\n")
        # file.write("outlier_treshold,translation_validity_treshold,Rotation_validity_treshold," + str(list(final_scores.keys()))[1:-1].translate({ord('\''): None, ord(' '): None}) + "\n")
        # file.write(f"{ort},{tvt},{Rvt}," + str(list(final_scores.values()))[1:-1].translate({ord('\''): None, ord(' '): None}) + "\n")
        file.write(time.ctime() + str(list(final_scores.values()))[1:-1].translate({ord('\''): None, ord(' '): None}) + "\n")
    with open(os.path.join(p["eval_path"], "all_results_info.txt"), "a") as file:
        lst = result_filename.split("_")
        # ort = lst[-4]
        # tvt = lst[-3]
        # Rvt = lst[-2]
        file.write(f"{result_filename}:\n")
        file.write("outlier_treshold,translation_validity_treshold,Rotation_validity_treshold," + str(list(final_scores.keys()))[1:-1].translate({ord('\''): None, ord(' '): None}) + f"{result_filename}" +"\n")
    # Print the final scores.
    misc.log("FINAL SCORES:")
    for score_name, score_value in final_scores.items():
        misc.log("- {}: {}".format(score_name, score_value))
    # precision_dict[(tvt, rvt)] = final_scores["bop19_average_precision"]
    # recall_dict[(tvt, rvt)] = final_scores["bop19_average_recall"]
    # pickle.dump(precision_dict, open('/home/ros/kzorina/vojtas/ycbv/ablation_kz_precision.p', 'wb'))
    # pickle.dump(recall_dict, open('/home/ros/kzorina/vojtas/ycbv/ablation_kz_recall.p', 'wb'))
# except Exception as e:
#     print(f"Skipping file {result_filename} because of exception:")
#     print(e)

misc.log("Done.")

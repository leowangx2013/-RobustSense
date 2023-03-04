import numpy as np
import argparse
from config_generation_utils import *
from plot_utils import *
import subprocess
import os
import time
import glob
import sys
from pathlib import Path
import yaml
import pandas as pd
import shutil

from tensorboard.backend.event_processing import event_accumulator

GPUS = [0, 1, 2, 3]

# VEHICLE_TYPES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
VEHICLE_TYPES = [1, 2, 3, 4, 5, 6, 7, 8, 9] # Exclude no vehicle 0
TERRAIN_TYPES = [0, 1, 2]
SPEED_TYPES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0]
DISTANCE_TYPES = [0.0, 1.0, 2.0, 3.0, 4.0, -1.0]

DOWNSTREAM_CONFIG_PATH = "/home/tianshi/FoundationSense/src/data"
PT_INDEXES_PATH = "/home/tianshi/GAN_Vehicle/pt_indexes"
# sys.path.append("../vae")

# Read args
parser = argparse.ArgumentParser()
parser.add_argument("--mask_con_n", type=int, default=1, help="number of masked condition for each run")
parser.add_argument("--beta", type=int, default=10, help="beta for KL dievergence loss weight")
parser.add_argument("--config_output_path", type=str, default="../vae/cvae_configs", help="output path of the generated cVAE config files")
parser.add_argument("--batch_name", type=str, default="mask_one_terrain", help="name for the batch")
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--n_process", type=int, default=8, help="number of processes to run in parallel")
parser.add_argument("--stage", type=str, default="train_vae", help="stage of the batch evaluation: train_vae, gen_vae, train_downstream")
parser.add_argument("--n_gen", type=int, default=25, help="number of generated images for each condition")

opt = parser.parse_args()

if opt.stage == "train_vae":
    if not os.path.exists(os.path.join(opt.config_output_path, opt.batch_name)):
        Path(os.path.join(opt.config_output_path, opt.batch_name)).mkdir(parents=True, exist_ok=True)

    for f in os.listdir(os.path.join(opt.config_output_path, opt.batch_name)):
        os.remove(os.path.join(opt.config_output_path, opt.batch_name, f))

    if not os.path.exists(f"../vae/weights/{opt.batch_name}"):
        Path(f"../vae/weights/{opt.batch_name}").mkdir(parents=True, exist_ok=True)

    for f in os.listdir(f"../vae/weights/{opt.batch_name}"):
        os.remove(os.path.join(f"../vae/weights/{opt.batch_name}", f))

elif opt.stage == "train_downstream":
    if not os.path.exists(os.path.join(DOWNSTREAM_CONFIG_PATH, opt.batch_name)):
        Path(os.path.join(DOWNSTREAM_CONFIG_PATH, opt.batch_name)).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(os.path.join(DOWNSTREAM_CONFIG_PATH, opt.batch_name)):
        os.remove(os.path.join(DOWNSTREAM_CONFIG_PATH, opt.batch_name, f))

# Generate VAE config files
def iterate_one_mask_on_terrain(terrain_types_by_vehicle_types):
    for v in VEHICLE_TYPES:
        if len(terrain_types_by_vehicle_types[v]) <= 1:
            continue
        for t in terrain_types_by_vehicle_types[v]:
            generate_vae_config_file([v], [t], output_path=os.path.join(opt.config_output_path, opt.batch_name))

def randomly_mask_condition(n_vehicle, n_terrain, n_runs, terrain_types_by_vehicle_types):
    mask_set = set()
    run_counter = 0
    while run_counter < n_runs:
        vs = np.random.choice(VEHICLE_TYPES, n_vehicle, replace=False)
        ts = np.random.choice(TERRAIN_TYPES, n_terrain, replace=False)
        
        if str(sorted(vs))+str(sorted(ts)) in mask_set:
            continue
        else:
            mask_set.add(str(sorted(vs))+str(sorted(ts)))
            run_counter += 1
            print("vs: ", list(vs))
            generate_vae_config_file(list(vs), list(ts), output_path=os.path.join(opt.config_output_path, opt.batch_name))

terrain_types_by_vehicle_types = get_terrain_types_by_vehicle_types()

if opt.stage == "train_vae":
    if opt.mask_con_n == 1:
        iterate_one_mask_on_terrain(terrain_types_by_vehicle_types)
    else:
        randomly_mask_condition(opt.mask_con_n, 1, 20, terrain_types_by_vehicle_types)

    # Run train_cVAE in batch
    processes = set()
    max_processes = opt.n_process

    for n, cvae_config_file in enumerate(glob.glob(os.path.join(opt.config_output_path, opt.batch_name, "*.yaml"))):
        run_name = os.path.basename(cvae_config_file).split(".")[0].split("_")[-1]
        print("run_name: ", run_name)
        weight_output_path = os.path.join("./weights/", opt.batch_name)
        command = f"python train_cVAE.py --cvae_config_file {cvae_config_file} --beta {opt.beta} --run {run_name}_beta{opt.beta} --gpu {GPUS[n % len(GPUS)]} --n_epochs {opt.n_epochs} --weight_output_path {weight_output_path}" 
        processes.add(subprocess.Popen([command], shell=True, cwd="../vae"))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([
                p for p in processes if p.poll() is not None])

# Run gen_cVAE in batch
elif opt.stage == "gen_vae":
    # Run gen_cVAE in batch
    processes = set()
    max_processes = opt.n_process

    # Clean up pt_files and pt_indexes
    for f in os.listdir(os.path.join("../pt_files", opt.batch_name)):
        shutil.rmtree(os.path.join("../pt_files", opt.batch_name, f))

    for f in os.listdir(os.path.join("../pt_indexes", opt.batch_name)):
        shutil.rmtree(os.path.join("../pt_indexes", opt.batch_name, f))

    for n, cvae_config_file in enumerate(glob.glob(os.path.join(opt.config_output_path, opt.batch_name, "*.yaml"))):
        run_name = os.path.basename(cvae_config_file).split(".")[0].split("_")[-1]
        # print("run_name: ", run_name)
        weight_output_path = os.path.join("./weights/", opt.batch_name)
        command = f"python gen_cVAE.py --run {run_name}_beta{opt.beta} --checkpoint {opt.batch_name}/cVAE_{run_name}_beta{opt.beta}.pt --gpu {GPUS[n % len(GPUS)]} --output_folder {opt.batch_name}" 
        processes.add(subprocess.Popen([command], shell=True, cwd="../vae"))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([
                p for p in processes if p.poll() is not None])

# Train downstream models in batch and record results
elif opt.stage == "train_downstream":
    # Generate downstream config files
    with open(os.path.join(DOWNSTREAM_CONFIG_PATH, "ACIDS_template.yaml"), "r") as file:
        config_template = yaml.safe_load(file)

    for index_path in glob.glob(os.path.join(PT_INDEXES_PATH, opt.batch_name, "*")):
        # Create aug config file
        new_config = config_template.copy()
        new_config["vehicle_classification"]["train_index_file"] = os.path.join(PT_INDEXES_PATH, opt.batch_name, os.path.basename(index_path), "train_index_aug.txt")
        new_config["vehicle_classification"]["val_index_file"] = os.path.join(PT_INDEXES_PATH, opt.batch_name, os.path.basename(index_path), "val_index.txt")
        new_config["vehicle_classification"]["test_index_file"] = os.path.join(PT_INDEXES_PATH, opt.batch_name, os.path.basename(index_path), "test_index.txt")

        with open(os.path.join(DOWNSTREAM_CONFIG_PATH, opt.batch_name, f"ACIDS_{os.path.basename(index_path)}_aug.yaml"), "w") as file:
            yaml.dump(new_config, file)

        # Create no-aug config file
        new_config["vehicle_classification"]["train_index_file"] = os.path.join(PT_INDEXES_PATH, opt.batch_name, os.path.basename(index_path), "train_index_no-aug.txt")
        new_config["vehicle_classification"]["val_index_file"] = os.path.join(PT_INDEXES_PATH, opt.batch_name, os.path.basename(index_path), "val_index.txt")
        new_config["vehicle_classification"]["test_index_file"] = os.path.join(PT_INDEXES_PATH, opt.batch_name, os.path.basename(index_path), "test_index.txt")

        with open(os.path.join(DOWNSTREAM_CONFIG_PATH, opt.batch_name, f"ACIDS_{os.path.basename(index_path)}_no-aug.yaml"), "w") as file:
            yaml.dump(new_config, file)

    # Run train_downstream in batch
    processes = set()
    max_processes = opt.n_process

    for n, downstream_config_file in enumerate(glob.glob(os.path.join(DOWNSTREAM_CONFIG_PATH, opt.batch_name, "*.yaml"))):
        run_name = os.path.basename(downstream_config_file).split(".")[0][6:]
        
        dataset = downstream_config_file.split(".")[0]
        config_path = os.path.join("./data", opt.batch_name)

        command = f"python train.py -dataset ACIDS_{run_name} -config_path {config_path} -model DeepSense -gpu {GPUS[n % len(GPUS)]}" 
        print(command)
        processes.add(subprocess.Popen([command], shell=True, cwd="/home/tianshi/FoundationSense/src"))

        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([
                p for p in processes if p.poll() is not None])
    
    # Summarize results
    runs = []
    maxes = []
    mins = []
    means = []
    lasts = []

    weight_output_path  = "/home/tianshi/FoundationSense/weights"
    downstream_config_files = sorted( glob.glob(os.path.join(DOWNSTREAM_CONFIG_PATH, opt.batch_name, "*.yaml")) )
    for n, downstream_config_file in enumerate(downstream_config_files):
        run_name = os.path.basename(downstream_config_file).split(".")[0][6:]
        weight_folder = f"ACIDS_{run_name}_DeepSense"
        latest_exp = max(glob.glob(os.path.join(weight_output_path, weight_folder, "exp*")), key=os.path.getmtime)   
        print("latest_exp: ", latest_exp)
        event_file = glob.glob(os.path.join(latest_exp, "train_events", "events*"))[0]
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        test_accuracy = pd.DataFrame(ea.Scalars("Evaluation/Test accuracy"))
        max_accuracy = test_accuracy["value"].max()
        min_accuracy = test_accuracy["value"].min()
        mean_accuracy = test_accuracy["value"].mean()
        last_accuracy = test_accuracy["value"].iloc[-1]
        
        print(f"{run_name}, max: {max_accuracy: .4f}, min: {min_accuracy: .4f}, mean: {mean_accuracy: .4f}, last: {last_accuracy: .4f}")

        runs.append(run_name)
        maxes.append(max_accuracy)
        mins.append(min_accuracy)
        means.append(mean_accuracy)
        lasts.append(last_accuracy)
    
    if not os.path.exists(f"./results/{opt.batch_name}"):
        Path(f"./results/{opt.batch_name}").mkdir(parents=True, exist_ok=True)

    with open(f"./results/{opt.batch_name}/results.txt", "w") as file:
        file.write("run, max, min, mean, last\n")
        for (run, max, min, mean, last) in zip(runs, maxes, mins, means, lasts):
            file.write(f"{run}, {max: .4f}, {min: .4f}, {mean: .4f}, {last: .4f}\n")

    plot_results(runs, maxes, opt.batch_name+"_max", f"./results/{opt.batch_name}")

    
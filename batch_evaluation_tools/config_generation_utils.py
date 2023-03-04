import os
import numpy as np

CLASS_NUM = 9
TERRAIN = {"Desert": 0, "Arctic": 1, "Normal": 2}

def get_terrain_types_by_vehicle_types():
    terrain_types_by_vehicle_types = {}
    with open("/home/tianshi/GAN_Vehicle/acids_dataset_utils/ACIDS_meta.csv", "r") as file:
        lines = file.readlines()
        for line in lines:
            filename, vehicle_type, _, terrain_type, _ = line.split(",")
            vehicle_type = int(vehicle_type.split(" ")[1])
            terrain_type = TERRAIN[terrain_type]
            if vehicle_type not in terrain_types_by_vehicle_types:
                terrain_types_by_vehicle_types[vehicle_type] = set()

            # print("terrain_type: ", terrain_type)
            terrain_types_by_vehicle_types[vehicle_type].add(terrain_type)
            
    return terrain_types_by_vehicle_types

def generate_vae_config_file(masked_vehicle_types, masked_terrain_types, output_path="./"):

    v = "".join([str(i) for i in masked_vehicle_types])
    t = "".join([str(i) for i in masked_terrain_types])
    config_file_name = f"cVAE_config_v{v}t{t}.yaml"

    with open(os.path.join(output_path, config_file_name), "w") as file:
        file.write("masked_vehicle_types: " + str(masked_vehicle_types) + "\n")
        file.write("masked_terrain_types: " + str(masked_terrain_types) + "\n")
        file.write("speed_types: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0]" + "\n")
        file.write("distance_types: [0.0, 1.0, 2.0, 3.0, 4.0, -1.0]" + "\n")
    
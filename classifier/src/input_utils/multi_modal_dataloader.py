import torch

from torch.utils.data import DataLoader
from input_utils.multi_modal_dataset import MultiModalDataset, TripletMultiModalDataset, ACIDS_Dataset
from input_utils.yaml_utils import load_yaml

def create_dataloader(option, args, dataset="Parkland", batch_size=64, workers=5):
    """create the dataloader for the given data path.

    Args:
        data_path (_type_): _description_
        workers (_type_): _description_
    """
    # select the index file
    if option == "train":
        index_file = args.dataset_config["train_index_file"]
    elif option == "val":
        index_file = args.dataset_config["val_index_file"]
    else:
        index_file = args.dataset_config["test_index_file"]

    if "Parkland" in dataset:
        triplet_flag = False
        dataset = MultiModalDataset(
            index_file,
            args.dataset_config["base_path"],
        )
    elif "ACIDS" in dataset:
        triplet_flag = False
        dataset = ACIDS_Dataset(
            index_file,
            args.dataset_config["base_path"],
            label_key=args.dataset_config["label_key"],
        )
    batch_size = min(batch_size, len(dataset))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(option == "train"), num_workers=workers)

    return dataloader, triplet_flag


def preprocess_triplet_batch(data, labels):
    """Preprocess the triplet batch by concatenating all elements within the tuple (anchor, pos, neg)

    Args:
        flag (_type_): _description_
        data (_type_): _description_
        labels (_type_): _description_
    """
    # cat data
    anchor_data = data[0]
    pos_data = data[1]
    neg_data = data[2]
    out_data = dict()
    for loc in anchor_data:
        out_data[loc] = dict()
        for mod in anchor_data[loc]:
            out_data[loc][mod] = torch.cat([anchor_data[loc][mod], pos_data[loc][mod], neg_data[loc][mod]], dim=0)

    # cat labels
    out_labels = torch.cat(labels, dim=0)

    return out_data, out_labels


if __name__ == "__main__":
    option = "train"
    config_file = "/home/sl29/AutoCuration/src/data_configs/ExtraSensory.yaml"

    dataloader = create_dataloader(option, config_file)

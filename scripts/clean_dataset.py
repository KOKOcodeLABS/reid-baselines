import argparse
import os
import shutil
import json


import logging
from datetime import datetime

def setup_logger(dataset_name):
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{dataset_name}_clean_{timestamp}.log"

    logger = logging.getLogger(dataset_name)
    logger.setLevel(logging.INFO)

    # Clear old handlers (important when re-running in same session)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger



def parse_market_filename(filename):
    name = filename.replace(".jpg", "")
    parts = name.split("_")
    pid = parts[0]
    cam = parts[1]
    return pid, cam

def clean_market():
    logger = setup_logger("market1501")

    raw_root = "data/raw/Market-1501-v15.09.15"
    clean_root = "data/clean/market1501"

    splits = {
        "train": "bounding_box_train",
        "query": "query",
        "gallery": "bounding_box_test"
    }

    stats = {
        "dataset": "market1501",
        "num_identities": 0,
        "train_images": 0,
        "query_images": 0,
        "gallery_images": 0,
        "images_dropped": 0
    }

    identity_set = set()

    logger.info("Starting Market-1501 cleaning")

    for split_name, raw_folder in splits.items():
        raw_path = os.path.join(raw_root, raw_folder)

        logger.info(f"Processing split: {split_name}")

        for filename in os.listdir(raw_path):
            if not filename.endswith(".jpg"):
                continue

            pid, cam = parse_market_filename(filename)

            if pid == "-1":
                stats["images_dropped"] += 1
                continue

            identity_set.add(pid)

            dst_dir = os.path.join(clean_root, split_name, pid)
            os.makedirs(dst_dir, exist_ok=True)

            src = os.path.join(raw_path, filename)
            dst = os.path.join(dst_dir, filename)

            shutil.copy2(src, dst)

            stats[f"{split_name}_images"] += 1

    stats["num_identities"] = len(identity_set)

    os.makedirs(clean_root, exist_ok=True)

    metadata_path = os.path.join(clean_root, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(stats, f, indent=4)

    logger.info("Cleaning complete")
    logger.info(json.dumps(stats, indent=4))



def parse_duke_filename(filename):
    name = filename.replace(".jpg", "")
    parts = name.split("_")

    pid = parts[0]
    cam = parts[1]

    return pid, cam

def clean_duke():
    logger = setup_logger("duke")

    raw_root = "data/raw/DukeMTMC-reID"
    clean_root = "data/clean/duke"

    splits = {
        "train": "bounding_box_train",
        "query": "query",
        "gallery": "bounding_box_test"
    }

    stats = {
        "dataset": "duke",
        "num_identities": 0,
        "train_images": 0,
        "query_images": 0,
        "gallery_images": 0,
        "images_dropped": 0
    }

    identity_set = set()

    logger.info("Starting DukeMTMC-reID cleaning")

    for split_name, raw_folder in splits.items():
        raw_path = os.path.join(raw_root, raw_folder)

        logger.info(f"Processing split: {split_name}")

        for filename in os.listdir(raw_path):
            if not filename.endswith(".jpg"):
                continue

            pid, cam = parse_duke_filename(filename)

            if pid == "-1":
                stats["images_dropped"] += 1
                continue

            identity_set.add(pid)

            dst_dir = os.path.join(clean_root, split_name, pid)
            os.makedirs(dst_dir, exist_ok=True)

            src = os.path.join(raw_path, filename)
            dst = os.path.join(dst_dir, filename)

            shutil.copy2(src, dst)

            stats[f"{split_name}_images"] += 1

    stats["num_identities"] = len(identity_set)

    os.makedirs(clean_root, exist_ok=True)

    with open(os.path.join(clean_root, "metadata.json"), "w") as f:
        json.dump(stats, f, indent=4)

    logger.info("Cleaning complete")
    logger.info(json.dumps(stats, indent=4))



def clean_cuhk03():
    logger = setup_logger("cuhk03")

    raw_root = "data/raw/CUHK03"
    clean_root = "data/clean/cuhk03"

    image_root = os.path.join(raw_root, "images_detected")
    split_file = os.path.join(raw_root, "splits_new_detected.json")

    logger.info("Starting CUHK03 cleaning (new protocol, detected)")

    with open(split_file, "r") as f:
        splits = json.load(f)

    split = splits[0]

    stats = {
        "dataset": "cuhk03",
        "num_identities": 0,
        "train_images": 0,
        "query_images": 0,
        "gallery_images": 0,
        "images_dropped": 0
    }

    identity_set = set()

    for split_name in ["train", "query", "gallery"]:
        logger.info(f"Processing split: {split_name}")

        for entry in split[split_name]:
            rel_path, pid, camid = entry

            rel_path = rel_path.replace("\\", "/")
            filename = os.path.basename(rel_path)
            src = os.path.join(image_root, filename)

            if not os.path.exists(src):
                logger.warning(f"Missing file: {src}")
                stats["images_dropped"] += 1
                continue

            pid = str(pid)
            identity_set.add(pid)

            dst_dir = os.path.join(clean_root, split_name, pid)
            os.makedirs(dst_dir, exist_ok=True)

            dst = os.path.join(dst_dir, filename)
            shutil.copy2(src, dst)

            stats[f"{split_name}_images"] += 1

    stats["num_identities"] = len(identity_set)

    os.makedirs(clean_root, exist_ok=True)

    with open(os.path.join(clean_root, "metadata.json"), "w") as f:
        json.dump(stats, f, indent=4)

    logger.info("Cleaning complete")
    logger.info(json.dumps(stats, indent=4))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    if args.dataset == "market1501":
        clean_market()
    elif args.dataset == "duke":
        clean_duke()
    elif args.dataset == "cuhk03":
        clean_cuhk03()
    else:
        raise ValueError("Unknown dataset")





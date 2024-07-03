import argparse
import os
import logging
import shutil
from sklearn.model_selection import train_test_split
import json
import operator

def write_user_sets(output_dir, train_fnames, test_fnames):
    trees_directory = "produced_data/trees"
    train_user_ids = set()
    for fname in train_fnames:
        path = os.path.join(trees_directory, fname)
        with open(path) as f:
           tree_json = json.load(f)
        train_user_ids.update([node["user_id"] for node in tree_json["nodes"]])

    test_user_ids = set()
    for fname in test_fnames:
        path = os.path.join(trees_directory, fname)
        with open(path) as f:
           tree_json = json.load(f)
        test_user_ids.update([node["user_id"] for node in tree_json["nodes"]])

    path = os.path.join(output_dir, "train_user_ids.json")
    with open(path, "w") as f:
        json.dump({
            "user_ids": list(train_user_ids)
        },
        f)

    path = os.path.join(output_dir, "test_user_ids.json")
    with open(path, "w") as f:
        json.dump({
            "user_ids": list(test_user_ids)
        },
        f)


def run(args):
    logging.info("Split dataset to train and test")
    logging.info("Test size:%s" % (args.test_size))

    trees = os.listdir("produced_data/trees")
    root_dir = os.path.join("produced_data", "datasets")
    train_trees, test_trees = train_test_split(trees, test_size=args.test_size, shuffle=False, random_state=42)
    print(len(trees))

    output_dir = os.path.join(root_dir, 'dataset')
    logging.info("Writing files in %s directory" % (output_dir))
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    train_path = os.path.join(output_dir, "train")
    test_path = os.path.join(output_dir, "test")
    os.makedirs(train_path)
    os.makedirs(test_path)

    logging.info("The dataset has %s train trees" % (len(train_trees)))
    logging.info("The dataset has %s test trees" % (len(test_trees)))
    for fname in train_trees:
        shutil.copy("produced_data/trees/%s" % (fname), os.path.join(train_path, fname))
    for fname in test_trees:
        shutil.copy("produced_data/trees/%s" % (fname), os.path.join(test_path, fname))
    logging.info("Writing sets of user ids for train, test, validation datasets")
    write_user_sets(output_dir, train_trees, test_trees)

    metadata = {
        'number_of_train_trees': len(train_trees),
        'number_of_test_trees': len(test_trees),
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(epilog="Example: python tweets_to_trees.py")
    parser.add_argument(
        "--test-size",
        help="Test size",
        dest="test_size",
        type=float,
        default=0.33,
    )

    args = parser.parse_args()
    run(args)

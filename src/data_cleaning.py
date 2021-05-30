import os
import glob
import pandas as pd
import json

from config import data_root

def mapping_labels(filename="labels_map.txt", path_data_dic="data_dic.txt", root=data_root):
    """
    Mapping the chinese label to numeric.
    Create labels_map.txt if it is not exist.
    If labels_map.txt exist then read labels_map.txt
    return ["orginal label"]
    """
    save_file = os.path.join(root, filename)
    labels_map = {}
    if os.path.exists(save_file):
        # read mapping info from file
        print(f"{filename} exist, read from file...")
        labels_map = json.load(open(save_file, encoding="utf-8"))
        labels_map = [labels_map[str(i)] for i in range(len(labels_map))]
    else:
        # create mapping info and save it
        print(f"{filename} not found, creating...")
        data_dic = get_data_dic()
        data_dic.append("isnull")
        for idx, label in enumerate(data_dic):
            labels_map[str(idx)] = label
        with open(save_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(labels_map, ensure_ascii=False))
        labels_map = [labels_map[str(i)] for i in range(len(labels_map))]
    
    return labels_map

def create_csv(path, root=data_root):
    """
    Input [train / test / validate] folder,
    create a csv file 
    return pandas file.
    """
    dir = os.path.join(root, path)

    # mapping labels
    labels_map = mapping_labels()
    print(labels_map[:5])
    
    # dataframe is used to create csv
    tmp_df = pd.DataFrame(columns=["filename", "label"])
    tmp_df["filename"] = os.listdir(dir)

    for idx, f in enumerate(os.listdir(dir)):
        filename = os.path.splitext(f)[0].encode("utf-8", errors="surrogateescape").decode("utf-8")
        
        # get label
        start = filename.find('_') + 1
        end = filename.find('.')
        label = filename[start:]

        try:
            label = labels_map.index(label)
        except:
            label = len(labels_map) - 1

        tmp_df["label"][idx] = label
    
    print(len(os.listdir(dir)))
    
    # Save dataframe as csv file
    csv_path = str(os.path.join(root, (path + ".csv")))
    print(csv_path)
    print("-------")
    tmp_df.to_csv(csv_path, encoding='utf-8', errors='surrogateescape', index=False)

def load_csv(path, root=data_root, encoding='utf-8'):
    df = pd.read_csv(path, encoding=encoding)
    print("total length: ", len(df))
    print(df.head())
    print("---------")
    return df

def get_data_dic(path="data_dic.txt", root=data_root):
    """
    Read file data dic .txt
    """
    path = os.path.join(root, path)
    data_dic = []
    with open(os.path.join(root, "data_dic.txt"), "r", encoding="utf-8") as f:
        data_dic = f.read().splitlines()

    return data_dic


if __name__ == "__main__":
    create_csv("test")
    # create_csv("val")
    # create_csv("train")

    # train_df = load_csv("train.csv")
    # test_df = load_csv("test.csv")
    # val_df = load_csv("val.csv")

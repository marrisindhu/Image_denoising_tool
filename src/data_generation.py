import warnings
warnings.filterwarnings("ignore")

import pathlib
import numpy as np
import cv2
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split

def get_images_paths(root_dir_ssid):
    # Getting SSID dataset images
    root = pathlib.Path(root_dir_ssid)
    img_paths = list(root.rglob("*.PNG*"))
    img_paths_lst = [str(path) for path in img_paths]

    gt_lst = []
    noisy_lst= []
    for p in img_paths_lst:
        img_type = p.split("/")[-1].split('_')[-3]
        if img_type=="NOISY":
            noisy_lst.append(p)
        elif img_type=="GT":
            gt_lst.append(p)

    noisy_array = np.asarray(noisy_lst)
    gt_array = np.asarray(gt_lst)
    return noisy_array, gt_array

def get_images_in_mem(images_paths):
    images_lst = []
    for img_path in tqdm(images_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        images_lst.append(img)
    return np.array(images_lst)


def load_data():
    noisy_array_paths, gt_array_paths = get_images_paths("data/SSID_dataset/")
    noisy_train_paths, noisy_test_paths, gt_train_paths, gt_test_paths = train_test_split(noisy_array_paths, gt_array_paths, test_size=0.20, random_state=42)

    noisy_train_images = get_images_in_mem(noisy_train_paths)
    noisy_test_images = get_images_in_mem(noisy_test_paths)

    gt_train_images = get_images_in_mem(gt_train_paths)
    gt_test_images = get_images_in_mem(gt_test_paths)

    return noisy_train_images, noisy_test_images, gt_train_images, gt_test_images

if __name__ == "__main__":
    load_data()


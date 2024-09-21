import pickle as pk
import os

dataset_path = os.path.join(r'datasets/ETT')

with open(dataset_path + "/scaler.pkl", "rb") as fb:
    data = pk.load(fb)
print("val data shape: ", data.shape)
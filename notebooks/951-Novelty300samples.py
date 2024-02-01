import os
import time
from pymongo import MongoClient
from rich import print
import matplotlib.pyplot as plt
import numpy as np
import src

src.visualization.set_matplotlib_params() # set matplotlib parameters to generate plots for the thesis

# novelty detection index for training with 300 samples

# script settings
URI = "mongodb://localhost:27017/"
db = "BACKUP"   
collection = "Train_300samples_test1"
threshold = 25 # threshold for novelty detection [%]

data = MongoClient(URI)[db][collection]
document = data.find_one({"_id": "Kmeans cluster novelty indicator"})
timestamps = document["timestamp"]
if not all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps)-1)): raise ValueError("Timestamps are not in order")
novelty_metric = np.multiply(document["values"],100)


fig, ax = plt.subplots()
ax.plot(timestamps,novelty_metric)
ax.hlines(threshold, timestamps[0], timestamps[-1], colors='r', linestyles='dashed')

ax.set_xlabel("Sample")
ax.set_ylabel("Novelty metric [%]")

plt.show()
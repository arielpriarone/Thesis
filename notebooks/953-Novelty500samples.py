import datetime
import os
import time
from pymongo import MongoClient
from rich import print
import matplotlib.pyplot as plt
import numpy as np
import src
import datetime as dt

src.visualization.set_matplotlib_params() # set matplotlib parameters to generate plots for the thesis

# novelty detection index for training with 500 samples configuration all features for only bearing 3 x

# script settings
URI = "mongodb://localhost:27017/"
db = "BACKUP"   
collection = "500_samples_OnlyBearing3x_allfeatures"
threshold = 50 # threshold for novelty detection [%]

data = MongoClient(URI)[db][collection]
document = data.find_one({"_id": "Kmeans cluster novelty indicator"})
timestamps = document["timestamp"]
if not all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps)-1)): raise ValueError("Timestamps are not in order")
novelty_metric = np.multiply(document["values"],100)


fig, ax = plt.subplots()
ax.plot(timestamps,novelty_metric,'k', label="Novelty metric")
ax.hlines(threshold, timestamps[0], timestamps[-1], colors='r', linestyles='dashed', label="Threshold")
ax.annotate("Novel behaviour\n2003-11-16 07:46", (dt.datetime.fromisoformat("2003-11-16T07:46"), threshold), textcoords="offset points", xytext=(-100,100), ha='center', fontsize=10, color='k', arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.set_xlabel("Sample")
ax.set_ylabel("Novelty metric [%]")
ax.legend()

plt.show()

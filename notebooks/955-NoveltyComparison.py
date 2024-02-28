from gettext import find
import os
from pydoc import doc
import time
from pymongo import MongoClient
from pyparsing import col
from rich import print
import matplotlib.pyplot as plt
import numpy as np
import src
import matplotlib.dates as mdates
import datetime as dt

src.visualization.set_matplotlib_params() # set matplotlib parameters to generate plots for the thesis

# RMS
def rms(x):
    return np.sqrt(np.mean(np.array(x)**2))


# script settings
URI = "mongodb://localhost:27017/"
db = "BACKUP"   
sensors = ["Bearing 1 x", "Bearing 1 y", "Bearing 2 x", "Bearing 2 y", "Bearing 3 x", "Bearing 3 y", "Bearing 4 x", "Bearing 4 y"]
threshold = 25 # threshold for novelty detection [%]
collection = "Train_300samples_test1"
data = MongoClient(URI)[db][collection]
document = data.find_one({"_id": "Kmeans cluster novelty indicator"})
timestamps = document["timestamp"]
if not all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps)-1)): raise ValueError("Timestamps are not in order")
novelty_metric = np.multiply(document["values"],100)


fig, ax = plt.subplots()

# # plot novelty metric - 300 samples
# collection = "Train_300samples_test1"
# data = MongoClient(URI)[db][collection]
# document = data.find_one({"_id": "Kmeans cluster novelty indicator"})
# timestamps = document["timestamp"]
# if not all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps)-1)): raise ValueError("Timestamps are not in order")
# novelty_metric = np.multiply(document["values"],100)
# ax.plot(timestamps,novelty_metric, label="trained with 300 samples")

# plot novelty metric - 600 samples
collection = "Train_600samples_test1"
data = MongoClient(URI)[db][collection]
document = data.find_one({"_id": "Kmeans cluster novelty indicator"})
timestamps = document["timestamp"]
if not all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps)-1)): raise ValueError("Timestamps are not in order")
novelty_metric = np.multiply(document["values"],100)
ax.scatter(timestamps,novelty_metric, label="trained with 600 samples", color='k', marker='.', s=2)
ax.annotate("Novel behaviour\n2003-11-20 23:44", (dt.datetime.fromisoformat("2003-11-20 23:44"), threshold), xytext=(dt.datetime.fromisoformat("2003-11-15 23:44"),650), ha='center', fontsize=10, color='k', arrowprops=dict(facecolor='black', arrowstyle='->'))

ax.hlines(threshold, timestamps[0], timestamps[-1], colors='k', linestyles='dashed', label="Threshold")

ax.set_xlabel("Sample")
ax.set_ylabel("Novelty metric [%]")
ax.legend()
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

# plot rms 
collection = "RawData_1st_test_IMSBearing"
documents = MongoClient(URI)[db][collection].find()
timestamps = [] 
RMS = {}
for sensor in sensors:
    RMS[sensor] = np.array([])
for i, document in enumerate(documents):
    timestamps.append(document["timestamp"])
    print("Reading document", i+1)
    for sensor in sensors:
        RMS[sensor] = np.append(RMS[sensor], rms(document[sensor]))

max_rms = np.zeros(len(timestamps))
for sensor in sensors:
    max_rms = np.maximum(max_rms, RMS[sensor])

fig, ax = plt.subplots()
# for sensor in sensors:
#     ax.plot(timestamps,RMS[sensor], label="Acc. " + str(sensor)[-3:])

ax.scatter(timestamps, max_rms, label="Max RMS (all sensors)", color='k', marker='.', s=2)
# ax.hlines(0.250, timestamps[0], timestamps[-1], colors='r', linestyles='dashed', label="Th. 1")
# ax.hlines(0.200, timestamps[0], timestamps[-1], colors='k', linestyles='dashed', label="Threshold")
ax.set_xlabel("Timestamp")
ax.set_ylabel("RMS")
ax.legend()
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


# # plot rms  - zoomed
# fig, ax = plt.subplots()
# for sensor in sensors:
#     ax.plot(timestamps,RMS[sensor], label="Acc. " + str(sensor)[-3:])

# ax.plot(timestamps, max_rms, label="Max RMS")
# ax.hlines(0.250, timestamps[865], timestamps[-1], colors='r', linestyles='dashed', label="Th. 1")
# ax.hlines(0.200, timestamps[865], timestamps[-1], colors='magenta', linestyles='dashed', label="Th. 2")
# ax.set_xlabel("Timestamp")
# ax.set_ylabel("RMS")
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.xticks(rotation=45)


plt.show()

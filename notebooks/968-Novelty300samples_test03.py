import datetime
import os
import time
import matplotlib as mpl
from matplotlib.lines import lineStyles
from pymongo import MongoClient
from rich import print
import matplotlib.pyplot as plt
import numpy as np
import src
import datetime as dt
import matplotlib.dates as mdates

src.visualization.set_matplotlib_params() # set matplotlib parameters to generate plots for the thesis

# novelty detection index for training with 500 samples configuration all features for only bearing 3 x

# script settings
URI = "mongodb://localhost:27017/"
db = "BACKUP"   
collection = "Train_300samples_test3"
threshold = 10 # threshold for novelty detection [%]

data = MongoClient(URI)[db][collection]
document = data.find_one({"_id": "Kmeans cluster novelty indicator"})
timestamps = document["timestamp"]
if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)): raise ValueError("Timestamps are not in order")
novelty_metric = np.multiply(document["values"],100)

# fig 1 - novelty detection
from bisect import bisect_right

timestamps_float = [ts.timestamp() for ts in timestamps]
offset = timestamps_float[0]  # offset to set the first timestamp to 0
timestamps_float = [ts - offset for ts in timestamps_float]  # set the first timestamp to 0
print(f"type(timestamps_float): {type(timestamps_float)}")
print(f"type(timestamps_float[0]): {type(timestamps_float[0])}")
PRED_INSTANTS = []
fig, ax = plt.subplots()
plt.subplots_adjust(top=0.936,
bottom=0.19,
left=0.112,
right=0.989,
hspace=0.2,
wspace=0.2)
ax.scatter(timestamps,novelty_metric,c='k',marker='.', s=2, label='Novelty metric')

timestamp_plot = [timestamp_float for timestamp_float in np.linspace(timestamps_float[0],timestamps_float[-1]+86400*2,500)]

ax.axhline(threshold, color='k', linestyle='dotted',label= 'Threshold',linewidth=1)
#ax.axhline(7000, color='k', linestyle='dashed',label= 'RUL threshold',linewidth=1)
ax.set_ylim(-300,7500)  
ax.set_xlabel("Timestamp")
ax.set_ylabel("Novelty metric [%]")
ax.set_yscale('symlog')
ax.annotate('Novel behaviour\n2004-04-12 19:21', xy=(dt.datetime(2004,4,12,19,21),10), xytext=(dt.datetime(2004,4,4,12,0),200), arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.legend()


# figure 2 - RUL prediction

PRED_INSTANTS = ["2004-04-12 19:21", "2004-04-17 00:13"]
colormap = [mpl.colors.to_hex(plt.cm.tab10(i)) for i in range(len(PRED_INSTANTS))]
fig, ax = plt.subplots()
ax.scatter(timestamps,novelty_metric,c='k',marker='.', s=2, label='Novelty metric')
for ind, instant in enumerate([dt.datetime.fromisoformat(aux).timestamp()-offset for aux in PRED_INSTANTS]):
    #time_of_prediction = dt.datetime.fromisoformat("2003-11-21T00:00").timestamp() - offset  # time of prediction "2003-11-16T07:46"
    time_of_prediction = instant
    windowing = 150 # windowing for the prediction

    print(f"type(time_of_prediction): {type(time_of_prediction)}")
    print(f"time_of_prediction: {time_of_prediction}")
    print(f"timestamps_float[0]: {timestamps_float[0]}")
    index = bisect_right(timestamps_float, time_of_prediction)
    print(f"Index of prediction: {index}")

    a,b,c = src.ExpRegressor(timestamps_float[index-windowing:index], novelty_metric[index-windowing:index]) #Fits the function a*exp(b*x)+c to the given data points.
    print (f"a: {a}, b: {b}, c: {c}")
    
    predictions = a * np.exp(b * np.array(timestamp_plot)) + c

    ax.axvline(timestamps[index], color=colormap[ind], linestyle='dashed',linewidth=1)
    ax.plot([dt.datetime.fromtimestamp(timestamp_float+offset) for timestamp_float in timestamp_plot],
            predictions, label=PRED_INSTANTS[ind], color=colormap[ind], linewidth=1)
    
ax.axhline(threshold, color='k', linestyle='dotted',label= 'Threshold',linewidth=1)
ax.axhline(7000, color='k', linestyle='dashed',label= 'RUL threshold',linewidth=1)
ax.set_ylim(-300,7500)  
ax.set_xlabel("Timestamp")
ax.set_ylabel("Novelty metric [%]")
ax.set_xlim(timestamps[0],timestamps[-1])
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.legend()

# figure 3 - Fault detection
threshold = -25 # threshold for fault detection [%]
document = data.find_one({"_id": "Kmeans cluster fault indicator"})
timestamps = document["timestamp"]
if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)): raise ValueError("Timestamps are not in order")
fault_metric = np.multiply(document["values"],100)
timestamps_float = [ts.timestamp() for ts in timestamps]
offset = timestamps_float[0]  # offset to set the first timestamp to 0
timestamps_float = [ts - offset for ts in timestamps_float]  # set the first timestamp to 0
print(f"type(timestamps_float): {type(timestamps_float)}")
print(f"type(timestamps_float[0]): {type(timestamps_float[0])}")
PRED_INSTANTS = []
fig, ax = plt.subplots()
plt.subplots_adjust(top=0.936,
bottom=0.19,
left=0.112,
right=0.989,
hspace=0.2,
wspace=0.2)
ax.scatter(timestamps,fault_metric,c='k',marker='.', s=2, label='Fault metric')

timestamp_plot = [timestamp_float for timestamp_float in np.linspace(timestamps_float[0],timestamps_float[-1]+86400*2,500)]

ax.axhline(threshold, color='k', linestyle='dotted',label= 'Threshold',linewidth=1)
#ax.axhline(7000, color='k', linestyle='dashed',label= 'RUL threshold',linewidth=1)
# ax.set_ylim(-300,7500)  
#ax.set_xlabel("Timestamp")
#ax.set_ylabel("Fault metric [%]")
# ax.set_yscale('symlog')
ax.annotate('Faulty behaviour\n2004-04-17 17:33', xy=(dt.datetime(2004,4,17,17,33),threshold), xytext=(dt.datetime(2004,4,5,12,0),25), arrowprops=dict(facecolor='black', arrowstyle='->'))
#ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
#ax.legend()


# rul calculation
from bisect import bisect_right

timestamps_float = [ts.timestamp() for ts in timestamps]
offset = timestamps_float[0]  # offset to set the first timestamp to 0
timestamps_float = [ts - offset for ts in timestamps_float]  # set the first timestamp to 0
print(f"type(timestamps_float): {type(timestamps_float)}")
print(f"type(timestamps_float[0]): {type(timestamps_float[0])}")
PRED_INSTANTS = ["2004-04-17 17:33"]#, "2004-02-18 18:13
#fig, ax = plt.subplots()
#ax.scatter(timestamps,fault_metric,c='k',marker='.', s=2, label='Fault metric')
colormap = [mpl.colors.to_hex(plt.cm.tab10(i)) for i in range(len(PRED_INSTANTS))]
timestamp_plot = [timestamp_float for timestamp_float in np.linspace(timestamps_float[0],timestamps_float[-1]+86400*2,500)]
for ind, instant in enumerate([dt.datetime.fromisoformat(aux).timestamp()-offset for aux in PRED_INSTANTS]):
    #time_of_prediction = dt.datetime.fromisoformat("2003-11-21T00:00").timestamp() - offset  # time of prediction "2003-11-16T07:46"
    time_of_prediction = instant
    windowing = 250 # windowing for the prediction

    print(f"type(time_of_prediction): {type(time_of_prediction)}")
    print(f"time_of_prediction: {time_of_prediction}")
    print(f"timestamps_float[0]: {timestamps_float[0]}")
    index = bisect_right(timestamps_float, time_of_prediction)
    print(f"Index of prediction: {index}")

    a,b,c = src.ExpRegressor(timestamps_float[index-windowing:index], fault_metric[index-windowing:index]) #Fits the function a*exp(b*x)+c to the given data points.
    print (f"a: {a}, b: {b}, c: {c}")
    
    predictions = a * np.exp(b * np.array(timestamp_plot)) + c

    ax.axvline(timestamps[index], color=colormap[ind], linestyle='dashed',linewidth=1)
    ax.plot([dt.datetime.fromtimestamp(timestamp_float+offset) for timestamp_float in timestamp_plot],
            predictions, label=PRED_INSTANTS[ind], color=colormap[ind], linewidth=1)
    
ax.axhline(500, color='k', linestyle='dashed',label= 'RUL threshold',linewidth=1)
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set_ylim(-100,600)  
ax.set_xlabel("Timestamp")
ax.set_ylabel("Fault metric [%]")  
ax.legend()

plt.show()
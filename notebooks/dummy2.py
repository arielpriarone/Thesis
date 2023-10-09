import matplotlib.pyplot as plt
import numpy as np

# Sample data for two groups with shared features
shared_labels = ['Feature A', 'Feature B', 'Feature C']
group1_values = [10, 15, 8]
group2_values = [18, 6]

# Define colors for each group
group1_color = 'b'
group2_color = 'g'

# Calculate the width for each bar group
bar_width = 0.35

# Create an array for the x-axis positions
x = np.arange(len(shared_labels))

# Create the figure and axes
fig, ax = plt.subplots()

# Plot the bars for the first group
bar1 = ax.bar(x - bar_width/2, group1_values, bar_width, label='Group 1', color=group1_color)

# Plot the bars for the second group
bar2 = ax.bar(x + bar_width/2, group2_values, bar_width, label='Group 2', color=group2_color)

# Set the x-axis labels and their positions
ax.set_xticks(x)
ax.set_xticklabels(shared_labels)

# Add a legend
ax.legend()

# Add labels and title
ax.set_xlabel('Features')
ax.set_ylabel('Values')
ax.set_title('Bar Plot with Different Groups')

# Show the plot
plt.show()

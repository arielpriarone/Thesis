
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

labels = ['Label_{}'.format(i) for i in range(1, 65)]


def custom_tick_locator(width):
    # Function to select the subset of tick locations based on the width of the plot
    num_labels = len(labels)
    num_ticks = 4  # Adjust the divisor based on your preference

    tick_step = max(num_labels // num_ticks, 1)
    return range(0, num_labels, tick_step)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 5))  # Adjust figsize as needed

# Plot your data (replace 'data' with your actual data)
data = [i**2 for i in range(1, 65)]
ax.plot(data)

# Get the current figure width
fig_width = fig.get_figwidth()

# Set the x-axis tick locator
ax.xaxis.set_major_locator(ticker.FixedLocator(custom_tick_locator(fig_width)))

# Set the tick labels using the subset of labels
ax.set_xticklabels([labels[i] for i in custom_tick_locator(fig_width)])

# Rotate the tick labels if needed
plt.xticks(rotation=45)

# Show the plot
plt.show()

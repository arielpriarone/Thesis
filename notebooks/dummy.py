import matplotlib.pyplot as plt

# Create a plot
fig, ax = plt.subplots()

# Add a legend with bbox_to_anchor
legend = fig.legend(["Data"], bbox_to_anchor=(0.8,0.7,0.2,0.3))

# Show the plot
plt.show()

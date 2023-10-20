# Sample dictionary with lists
data_dict = {
    'list1': [1, 2, 3, 4, 5],
    'list2': ['A', 'B', 'C', 'D', 'E'],
    'list3': [10.1, 20.2, 30.3, 40.4, 50.5]
}

# Sample filter list
filter_list = [True, False, True, False, True]

# Use a list comprehension to select elements based on the filter
filtered_dict = {key: [value for value, select in zip(data_dict[key], filter_list) if select] for key in data_dict}

# Print the resulting filtered dictionary
print(filtered_dict)

from itertools import chain

list_of_lists = [[1, 2, 4], [4, 5, 6], [7, 8, 9]]
flat_list = list(chain.from_iterable(list_of_lists))
print(flat_list)
import os
import json

#This is basically for the my_cpu case to load part of the data.

def create_subset(data_dir, dataset_file_name, split):

    max_limit = 5000
    _subset = dict()

    print('Creating subset.')

    with open(os.path.join(data_dir, dataset_file_name), 'r') as f:
        n2n_data = json.load(f)

    for i in range(max_limit):
        _subset[str(i)] = n2n_data[str(i)]

    with open(os.path.join(data_dir, 'subset_'+dataset_file_name.split('.')[0]+'.json'), 'w') as f:
        json.dump(_subset, f)

    print('Done')

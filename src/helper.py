import os
from shutil import rmtree
import numpy as np
import pandas as pd

def create_directory_if_not_defined(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def delete_files_in_directory(dir,recursive=False):
    for the_file in os.listdir(dir):
        file_path = os.path.join(dir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path) and recursive: rmtree(file_path)
        except Exception as e:
            print(e)

def setup_clean_directory(dir):
    create_directory_if_not_defined(dir)
    delete_files_in_directory(dir,recursive=True)

def get_files_of_type(path, type='jpg'):
    return np.array([x for x in sorted(os.listdir(path)) if x.lower().endswith(type.lower())])

def get_files_filtered(path, regex):
    import re
    matches = []
    pattern = re.compile(regex)
    for file in get_files_of_type(path,''):
        if pattern.match(file):
            matches.append(file)
    return np.array(matches)


def get_subdirectories(path):
    return os.walk(path).__next__()[1]

def array_is_in_array(arr1,arr2):
    rtn = np.zeros(arr1.shape, dtype='bool')
    for e in arr2:
        rtn = np.logical_or(rtn,arr1 == e)
    return rtn



def save_csv(data, path):
    pd.DataFrame(data).to_csv(path, index=False, header=False)

def read_csv(path):
    return pd.read_csv(path, header=None).to_numpy()





if __name__ == '__main__':
    print(get_files_filtered('.','^ac'))
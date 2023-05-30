import os
import copy
import numpy as np


def search_file_list(rootname, file_name):
    file_list = []
    for root, dirs, files in os.walk(rootname):
        if file_name in files:
            file_list.append(os.path.join(root, file_name))
    return file_list

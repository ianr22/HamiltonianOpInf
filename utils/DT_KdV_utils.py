# IMPORTS
# LIBRARIES
import numpy as np

# function to return POD base
def get_POD_basis(UU, n=150):
    return UU[:,:n]

# helper function to save paths
def set_path_name_list(dir, subdirectory: str, names: list):
    return [f"{dir}/{subdirectory}/{name}" for name in names]

# IMPORTS
# LIBRARIES
import numpy as np

def get_POD_basis(UU, n=150):
    return UU[:,:n]
# IMPORTS
# LIBRARIES
import sys
import numpy as np
import matplotlib.pyplot as plt
# UTILS
from ..utils.KdV_BBM_utils import *
from ..utils.OpInf_utils import *
from ..utils.ROM_utils import *

# VARIABLES
# PARAMETERS
N = 500
Nt = 1000
T = 20
# RANGES
xTrain = np.linspace(-20, 20, N)
dx     = xTrain[1]-xTrain[0]
tTrain = np.linspace(0, T, Nt)
# INITIAL CONDITIONS (FOM)
ic = KdV_soliton_IC(xTrain)
#KdV ICs
KdV_params = {
    "a" : -6,
    "p" : 0,
    "v" : -1
} # These are the same as in the paper, but I figure I'd put them here for ease

print("running")

# A function to generate a POD Bases
# PARAMETERS:
# RETURNS: 
def POD_basis():
    print("POD generating")
    
    # Compute FOM snapshots
    A, B, E      = build_KdV_mats(N, [-20,20])
    X1, Xd1, gH1 = integrate_KdV_v1_FOM(tTrain, ic, A, B, **KdV_params)
    X2, Xd2, gH2 = integrate_KdV_v2_FOM(tTrain, ic, A, E, **KdV_params)

    print("snapshots calculated")

    # SVD of Snapshot Matrices
    UU1mc, SS1mc = np.linalg.svd(X1-ic.reshape(-1,1))[:2]
    UU1, SS1     = np.linalg.svd(X1)[:2]
    UU2mc, SS2mc = np.linalg.svd(X2-ic.reshape(-1,1))[:2]
    UU2, SS2     = np.linalg.svd(X2)[:2]

    print("SVD of snapshots calculated")

    # Building operators for the ROMs
    # Mean-centered operators
    (OpListMC1, OpListMC1G, 
    OpListMC2, OpListMC2G) = build_KdV_ROM_Ops([UU1mc,UU2mc], A,
                                                    B, E, ic, 
                                                    **KdV_params, MC=True)
    
    print("Mean centered operators constructed")

    # No mean-centering
    (OpList1, OpList1G,
    OpList2, OpList2G)     = build_KdV_ROM_Ops([UU1,UU2], A, 
                                                    B, E, ic, 
                                                    **KdV_params, MC=False)
    
    print("Non-mean-centered operators constructed")
    
    return("swagtastic")

# main function, handles script 
def main():
    print(POD_basis())

if __name__ == '__main__':
    main()
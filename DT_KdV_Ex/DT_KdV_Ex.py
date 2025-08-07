# IMPORTS
# LIBRARIES
from os import path
import numpy as np
import matplotlib.pyplot as plt
# UTILS
from ..utils.KdV_BBM_utils import *
from ..utils.OpInf_utils import *
from ..utils.ROM_utils import *
from ..utils.DT_KdV_utils import *

# VARIABLES
# FILE
dir = path.dirname(__file__)

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

# FUNCTIONS
# A function to generate a POD Bases
# PARAMETERS:
# RETURNS: 
def POD_basis(A, B, E):
    print("POD generating")

    X1, Xd1, gH1 = integrate_KdV_v1_FOM(tTrain, ic, A, B, **KdV_params)
    X2, Xd2, gH2 = integrate_KdV_v2_FOM(tTrain, ic, A, E, **KdV_params)

    # SVD of Snapshot Matrices
    UU1mc, SS1mc = np.linalg.svd(X1-ic.reshape(-1,1))[:2]
    UU1, SS1     = np.linalg.svd(X1)[:2]
    UU2mc, SS2mc = np.linalg.svd(X2-ic.reshape(-1,1))[:2]
    UU2, SS2     = np.linalg.svd(X2)[:2]

    print("SVD of snapshots calculated")

    bases = [get_POD_basis(U) for U in [UU1, UU2, UU1mc, UU2mc]]
    
    return bases

# A function to build reduced operators for hamiltonian operator inference
def hamiltonian_reduced_operators(UList, A, B, E, ic, n=150, a=-6, p=0, v=-1, MC=False):
    ic      = ic.flatten()
    U1, U2  = UList[0], UList[1]
    N       = U1.shape[0]

    LHat    = U1.T @ A @ U1
    cVecV1  = np.zeros(n)
    cVecV2  = np.zeros(n)
    CmatV1  = U1.T @ (p*identity(N)+v*B) @ U1
    CmatV2  = U2.T @ (p*A + v*E) @ U2
    temp1   = np.einsum('ia,ib->iab', U1, U1)
    TtensV1 = a/2 * np.einsum('ia,ibc', U1, temp1)
    temp2   = np.einsum('ia,ib->iab', U2, U2)
    temp2   = temp2.transpose(1,2,0) @ (A @ U2)
    TtensV2 = a/3 * (temp2.transpose(0,2,1)-temp2.transpose(2,1,0))

    # Extra terms in case of mean-centering
    if MC:
        # For Hamiltonian
        ich = U2.T @ ic
        cVecV1 += U1.T @ (a/2 * (ic**2) + (p*identity(N)+v*B) @ ic)
        cVecV2 += (U2.T @ C(ic, A, a) @ U2 + CmatV2) @ ich
        CmatV1 += a * U1.T @ (ic.reshape(-1,1) * U1)
        CmatV2 += U2.T @ C(ic, A, a) @ U2 + TtensV2.transpose(0,2,1) @ ich

    return ( [cVecV1, CmatV1, TtensV1, LHat], 
             [cVecV2, CmatV2, TtensV2] )

# A function to build reduced operators for Galerkin ROM
def galerkin_reduced_operators(UList, A, B, E, ic, n=150, a=-6, p=0, v=-1, MC=False):
    ic      = ic.flatten()
    U1, U2  = UList[0], UList[1]
    N       = U1.shape[0]

    temp1    = np.einsum('ia,ib->iab', U1, U1)
    temp2    = np.einsum('ia,ib->iab', U2, U2)
    temp2    = temp2.transpose(1,2,0) @ (A @ U2)
    TtensV2  = a/3 * (temp2.transpose(0,2,1)-temp2.transpose(2,1,0))
    cVecV1G  = np.zeros(n)
    cVecV2G  = np.zeros(n)
    CmatV1G  = U1.T @ A @ (p*identity(N)+v*B) @ U1
    CmatV2G  = U2.T @ (p*A + v*E) @ U2
    TtensV1G = a/2 * np.einsum('aj,jbc', U1.T @ A, temp1)
    TtensV2 = a/3 * (temp2.transpose(0,2,1)-temp2.transpose(2,1,0))

    # Extra terms in case of mean-centering
    if MC:
        cVecV1G += U1.T @ A @ (a/2 * ic**2 + (p*identity(N)+v*B) @ ic)
        CmatV1G += U1.T @ A @ (a * ic.reshape(-1,1) * U1)

        cVecV2G += U2.T @ (C(ic, A, a) + p*A + v*E) @ ic
        temp2   = np.einsum('ia,ib->abi', U2, U2)
        # TV2p1   = temp2 @ A.todense()
        TV2p1   = np.einsum('abi,ij', temp2, A.todense())
        TV2p2   = np.einsum('aj,jb->abj', U2.T@A, U2)
        TpartV2 = a/3 * (TV2p1 + TV2p2)
        CmatV2G += TpartV2 @ ic + U2.T @ C(ic, A, a) @ U2

    return ([cVecV1G, CmatV1G, TtensV1G],
            [cVecV2G, CmatV2G, TtensV2] )

def main():
    # Full order model
    print("running")
    print("FOM snapshots calculating")
    A, B, E = build_KdV_mats(N, [-20,20])
    print("snapshots calculated")

    # Reduced bases
    bases = POD_basis(A, B, E)
    filenames = ["U1.npy", "U2.npy", "U1mc.npy", "U2mc.npy"]
    paths = set_path_name_list(dir, "Bases", filenames)
    print("Bases generated")
    for basis in zip(paths, bases):
        np.save(*basis)

    # Hamiltonian operators
    print("calculating Hamiltonian ROM operators")
    H_OpInf_ROM_ops = hamiltonian_reduced_operators(bases[0:2], A, B, E, ic, **KdV_params)
    print("calculating mean centered oparators")
    H_OpInf_ROM_ops_MC = hamiltonian_reduced_operators(bases[2:], A, B, E, ic, **KdV_params, MC=True)

    # Galerkin operators
    print("calculating Galerkin ROM operators")
    Galerkin_ROM_ops = galerkin_reduced_operators(bases[0:2], A, B, E, ic, **KdV_params)
    print("calculating mean centered oparators")
    Galerkin_ROM_ops_MC = galerkin_reduced_operators(bases[2:], A, B, E, ic, **KdV_params, MC=True)

    # Seems like a weird way to do this, but here we go
    # Saving operators to disk
    # Hamiltonian
    HOpInfV1FileNames = ["cVecV1.npy", "CmatV1.npy", "TtensV1.npy", "LHat.npy"]
    HOpInfV2FileNames = ["cVecV2G.npy", "CmatV2G.npy", "TtensV2.npy"]
    HOpInfV1paths = set_path_name_list(dir, "Operators/Hamiltonian", HOpInfV1FileNames)
    HOpInfV2paths = set_path_name_list(dir, "Operators/Hamiltonian", HOpInfV2FileNames)  
    for operator in zip(HOpInfV1paths, H_OpInf_ROM_ops[0]):
        np.save(*operator)
    for operator in zip(HOpInfV2paths, H_OpInf_ROM_ops[1]):
        np.save(*operator)

    HOpInfV1MCFileNames = ["cVecV1.npy", "CmatV1.npy", "TtensV1.npy", "LHat.npy"]
    HOpInfV2MCFileNames = ["cVecV2G.npy", "CmatV2G.npy", "TtensV2.npy"]
    HOpInfV1MCpaths = set_path_name_list(dir, "Operators/Hamiltonian/MC", HOpInfV1MCFileNames)
    HOpInfV2MCpaths = set_path_name_list(dir, "Operators/Hamiltonian/MC", HOpInfV2MCFileNames)  
    for operator in zip(HOpInfV1MCpaths, H_OpInf_ROM_ops_MC[0]):
        np.save(*operator)
    for operator in zip(HOpInfV2MCpaths, H_OpInf_ROM_ops_MC[1]):
        np.save(*operator)

    # Galerkin
    GOpInfV1FileNames = ["cVecV1.npy", "CmatV1.npy", "TtensV1.npy", "LHat.npy"]
    GOpInfV2FileNames = ["cVecV2G.npy", "CmatV2G.npy", "TtensV2.npy"]
    GOpInfV1paths = set_path_name_list(dir, "Operators/Galerkin", GOpInfV1FileNames)
    GOpInfV2paths = set_path_name_list(dir, "Operators/Galerkin", GOpInfV2FileNames)  
    for operator in zip(GOpInfV1paths, Galerkin_ROM_ops[0]):
        np.save(*operator)
    for operator in zip(GOpInfV2paths, Galerkin_ROM_ops[1]):
        np.save(*operator)

    GOpInfV1MCFileNames = ["cVecV1.npy", "CmatV1.npy", "TtensV1.npy", "LHat.npy"]
    GOpInfV2MCFileNames = ["cVecV2G.npy", "CmatV2G.npy", "TtensV2.npy"]
    GOpInfV1MCpaths = set_path_name_list(dir, "Operators/Galerkin/MC", GOpInfV1MCFileNames)
    GOpInfV2MCpaths = set_path_name_list(dir, "Operators/Galerkin/MC", GOpInfV2MCFileNames)  
    for operator in zip(GOpInfV1MCpaths, Galerkin_ROM_ops_MC[0]):
        np.save(*operator)
    for operator in zip(GOpInfV2MCpaths, Galerkin_ROM_ops_MC[1]):
        np.save(*operator)

if __name__ == '__main__':
    main()
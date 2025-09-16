# IMPORTS
# LIBRARIES
from os import path
import numpy as np
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
# UTILS
from ..utils.KdV_BBM_utils import *
from ..utils.OpInf_utils import *
from ..utils.ROM_utils import *
from ..utils.DT_KdV_utils import *

# VARIABLES
dir = path.dirname(__file__)

N = 1000
T = 20
NtTest = 1000
tTest = np.linspace(0, T, NtTest)
xTrain = np.linspace(-20, 20, N)
ic = KdV_soliton_IC(xTrain)
dt = tTest[1] - tTest[0]

basesFilenames = ["U1.npy", "U2.npy", "U1mc.npy", "U2mc.npy", "X.npy"]
Bases = [np.load(f"{dir}/Bases/{file}") for file in basesFilenames]

FOMBasesFilenames = ["A.npz", "B.npz", "E.npz"]
FOMBases = [load_npz(f"{dir}/Bases/{file}") for file in FOMBasesFilenames]

opListFilenamesFOM = ["gH1.npy", "Xd1.npy"]
opListFOM = [np.load(f"{dir}/Operators/FOM/{file}") for file in opListFilenamesFOM]

opListFilenames = ["cVecV1.npy", "CmatV1.npy", "TtensV1.npy", "LHat.npy"]
OpListMC = [np.load(f"{dir}/Operators/Hamiltonian/MC/{file}") for file in opListFilenames]
OpList = [np.load(f"{dir}/Operators/Hamiltonian/{file}") for file in opListFilenames]

UU1 = Bases[0]
UU2 = Bases[1]
gH1 = opListFOM[0]
Xd1 = opListFOM[1]
A = FOMBases[0]
print("A shape:", np.shape(A), "dtype:", type(A))
B = FOMBases[1]
E = FOMBases[2]
X1 = Bases[4]
Xtest = integrate_KdV_v1_FOM(tTest, ic, A, B)[0]

(OpList1, OpList1G,
 OpList2, OpList2G)     = build_KdV_ROM_Ops([UU1,UU2], A, 
                                                B, E, ic, MC=False)

gradHhat = UU1[:,:150].T @ gH1
XdotHat  = UU1[:,:150].T @ Xd1
rhsG     = gradHhat @ XdotHat.T
ghghT    = gradHhat @ gradHhat.T

nList = [4*(i+1) for i in range(18)]
eOp     = np.zeros(len(nList))
eIntG   = np.zeros(len(nList))
eIntH   = np.zeros(len(nList))
eHOp    = np.zeros(len(nList))

XrecIntG   = np.zeros((len(nList), N, NtTest))
XrecIntH   = np.zeros((len(nList), N, NtTest))
XrecHOp    = np.zeros((len(nList), N, NtTest))
XrecOp     = np.zeros((len(nList), N, NtTest))

OpInfLists  = build_OpInf_stuff(UU1, X1, Xd1, gH1, A, nList[-1])

LhatHOpFull   = NC_H_OpInf(OpInfLists[0], nList[-1], eps=0.0e-12)

DhatOpFull   = G_OpInf([ghghT, rhsG], nList[-1], eps=0e-10)

for i,n in enumerate(nList):

    LhatHOp   = LhatHOpFull[:n,:n]
    
    DhatOp   = DhatOpFull[:n,:n]
    # DhatOpFull  = ou.G_OpInf([ghghT, rhsG], n, eps=1e-12)

    OpList[-1]   = LhatHOp

    XrecIntG[i]   = integrate_KdV_v1_ROM(tTest, OpList1G, ic, UU1, n, 
                                            Hamiltonian=False, MC=False, Newton=True)
    XrecIntH[i]   = integrate_KdV_v1_ROM(tTest, OpList1, ic, UU1, n, 
                                            Hamiltonian=True, MC=False, Newton=True)
    XrecHOp[i]    = integrate_KdV_v1_ROM(tTest, OpList, ic, UU1, n,
                                            Hamiltonian=True, MC=False, Newton=True)
    # XrecOp    = ru.integrate_OpInf_ROM(tTest, DhatOp, ic, UU1)
    OpList[-1]   = DhatOp
    XrecOp[i]    = integrate_KdV_v1_ROM(tTest, OpList, ic, UU1, n,
                                           Hamiltonian=True, MC=False, Newton=True)

    eIntG[i]   = relError(Xtest, XrecIntG[i])
    eIntH[i]   = relError(Xtest, XrecIntH[i])
    eHOp[i]    = relError(Xtest, XrecHOp[i])
    eOp[i]     = relError(Xtest, XrecOp[i])

# Print error magnitudes
print(f'the relative L2 errors for intrusive G-ROM (no MC) are {eIntG}')
print(f'the relative L2 errors for intrusive H-ROM (no MC) are {eIntH}')
print(f'the relative L2 errors for NC-H-OpInf (no MC) are {eHOp}')
print(f'the relative L2 errors for G-OpInf (no MC) are {eOp}')

name = "tab10"
cmap = get_cmap(name)

plt.semilogy(nList, eIntG, label='Intrusive G-ROM (no MC)',
             marker='o', linestyle='--', color=cmap.colors[0], linewidth=0.5, markersize=5)
plt.semilogy(nList, eIntH, label='Intrusive H-ROM (no MC)',
             marker='s', linestyle='--', color=cmap.colors[1], linewidth=0.5, markersize=5)
plt.semilogy(nList, eOp,  label='G-OpInf (no MC)',
             marker='*', linestyle='--', color=cmap.colors[2], linewidth=0.5, markersize=6)
plt.semilogy(nList, eHOp, label='NC-H-OpInf (no MC)',
             marker='v', linestyle='--', color=cmap.colors[3], linewidth=0.5, markersize=5)
plt.ylabel('relative $L^2$ error')
plt.xlabel('basis size $n$')
plt.title('KdV ROM Errors (Reproductive)')
plt.ylim([10**-5,10])
plt.legend(loc=3)

plt.tight_layout()
# plt.savefig(f'KdVPlotT{T}', transparent=True)
plt.show()
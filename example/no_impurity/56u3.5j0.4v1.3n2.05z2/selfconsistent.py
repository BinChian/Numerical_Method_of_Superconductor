import numpy as np
import torch

import h5py

import nmsldos

setting = nmsldos.Basic_Setting(filename = '56u3.5j0.4v1.3n2.05z2m0')
setting.size_parameter(nx = 56, ny = 56, nz = 2)
setting.hopping_parameter(t1 = 1.00, t2 = -0.08, t3 = -1.35, t4 = 0.12, t5 = -0.09, t6 = -0.25, orbit_rotate = 90)
# setting.potential_parameter(mu = 3.680, u  = 3.500, jh = 0.400, v = 1.300) # normal
setting.potential_parameter(mu = -0.8, u  = 0, jh = 0, v = 1.300) # m0

lattice = nmsldos.Lattice(setting)
lattice.lattice_square()

readfile = nmsldos.Read_File(setting.filename)
pair, S_pz, S_mz = readfile.spm_wave(setting, lattice, file_exist = False)

selfconsistent = nmsldos.Selfconsistent()
selfconsistent_times = 0
while (selfconsistent.converge_total == False):

    selfconsistent_times += 1

    H = nmsldos.Hamiltonian.selfconsistent(setting, lattice, pair, S_pz, S_mz)
    H = torch.tensor(H, dtype = torch.float32).cuda()
    eig_val, eig_vec = torch.linalg.eigh(H)
    eig_val = np.array(eig_val.cpu())
    eig_vec = np.array(eig_vec.cpu())
    
    S_pz, S_mz = selfconsistent.electrons(setting, eig_vec, eig_val, S_pz, S_mz)
    pair, pair_pxpy, pair_pxmy = selfconsistent.spm_wave_pair(setting, lattice, eig_vec, eig_val, pair)

    n_i = selfconsistent.electrons_

    print('selfconsistent_times = ' + str(selfconsistent_times) + ', n_i = ' + str(np.round_(n_i, decimals = 4)))

    if selfconsistent.converge_spin == True and selfconsistent.converge_pair == True:
        selfconsistent.converge_total = True

with h5py.File(setting.filename + '.h5', 'w') as f:
    f.create_dataset('pair_pxpy', data = pair_pxpy)
    f.create_dataset("pair_pxmy", data = pair_pxmy)
    f.create_dataset("S_pz", data = S_pz)
    f.create_dataset("S_mz", data = S_mz)

print('self-consistent done. selfconsistent_times = ' + str(selfconsistent_times) + ', n_i = ' + str(np.round_(n_i, decimals = 4)))
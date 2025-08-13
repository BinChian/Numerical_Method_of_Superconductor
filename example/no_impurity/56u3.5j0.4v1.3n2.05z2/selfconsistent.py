import numpy as np
import torch

import h5py

import nmsldos

setting = nmsldos.parameter(filename = '56u3.5j0.4v1.3n2.05z2m0')
setting.size_parameter(nx = 56, ny = 56, nz = 2)
setting.hopping_parameter(t1 = 1.00, t2 = -0.08, t3 = -1.35, t4 = 0.12, t5 = -0.09, t6 = -0.25, orbit_rotate = 90)
setting.potential_parameter(mu = 3.680, u  = 3.500, jh = 0.400, v = 1.300)

lattice = nmsldos.lattice(setting)
lattice.lattice_square()

readfile = nmsldos.readfile(setting.filename)
pair, S_pz, S_mz = readfile.spm_wave(setting, lattice, file_exist = False)

selfconsistent = nmsldos.selfconsistent()
selfconsistent_times = 0
while (selfconsistent.converge_total == False):

    selfconsistent_times += 1

    H = nmsldos.hamiltonian.selfconsistent(setting, lattice, pair, S_pz, S_mz)
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

print('self-consistent done. selfconsistent_times = ' + str(selfconsistent_times) + ', n_i = ' + str(np.round_(n_i, decimals = 4)))

with h5py.File(setting.filename + '.h5', 'a') as f:
    # save parameter
    if 'parameter' in f:
        del f['parameter']
        grp_parameter = f.create_group('parameter')
    else:
        grp_parameter = f.create_group('parameter')
        
    for key, value in setting.__dict__.items():
        # if isinstance(value, str):
        #     dt = h5py.string_dtype(encoding = 'utf-8')
        #     grp_parameter.create_dataset(key, data = value, dtype = dt)
        # else:
        #     grp_parameter.create_dataset(key, data = value)
        grp_parameter.create_dataset(key, data = value)
    
    # save pair 
    if 'pair' in f:
        del f['pair']
        grp_pair = f.create_group('pair')
    else:
        grp_pair = f.create_group('pair')
    
    grp_pair.create_dataset('pair_pxpy', data = pair_pxpy)
    grp_pair.create_dataset('pair_pxmy', data = pair_pxmy)
    
    # save density wave
    if 'density wave' in f:
        del f['density wave']
        grp_dw = f.create_group('density wave')
    else:
        grp_dw = f.create_group('density wave')
    
    grp_dw.create_dataset('S_pz', data = S_pz)
    grp_dw.create_dataset('S_mz', data = S_mz)

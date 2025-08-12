import numpy as np
import torch

import h5py

from tqdm import tqdm
from time import sleep

import nmsldos

setting = nmsldos.parameter(filename = '56u3.5j0.4v1.3n2.05z2')
setting.size_parameter(nx = 56, ny = 56, nz = 2)
setting.hopping_parameter(t1 = 1.00, t2 = -0.08, t3 = -1.35, t4 = 0.12, t5 = -0.09, t6 = -0.25, orbit_rotate = 90)
setting.potential_parameter(mu = 3.680, u  = 3.500, jh = 0.400, v = 1.300)
# setting.potential_parameter(mu = -0.8, u  = 0, jh = 0, v = 1.300) # m0

lattice = nmsldos.lattice(setting)
lattice.lattice_square()

readfile = nmsldos.readfile(setting.filename)
pair, S_pz, S_mz = readfile.spm_wave(setting, lattice, file_exist = False)

T = 0.001
mx = 20; my = 20; msite = mx*my
resolution = 2001; ubound =  1.0; lbound = -1.0
E = np.linspace(lbound, ubound, resolution)

psite = 1; probe = [lattice.index[0, 0, 0, 0]]
rho = np.zeros((psite, resolution))

supercell_times = 0
progress = tqdm(total = mx*my , desc = "Supercell: ", ncols = 100, ascii = True)
for kx in range(mx):
    for ky in range(my):

        pfx = np.exp(1j*2.0*np.pi*kx/mx/setting.nx)
        pfy = np.exp(1j*2.0*np.pi*ky/my/setting.ny)

        H = nmsldos.hamiltonian.supercell(setting, lattice, pair, S_pz, S_mz, pfx, pfy)
        H = torch.tensor(H, dtype = torch.complex64).cuda()
        eig_val, eig_vec = torch.linalg.eigh(H)
        eig_val = np.array(eig_val.cpu())
        eig_vec = np.array(eig_vec.cpu())
        
        energy_minus = np.transpose(1.0 - (np.tanh(0.5*(eig_val - E.reshape(-1, 1))/T))**2)
        energy_plus  = np.transpose(1.0 - (np.tanh(0.5*(eig_val + E.reshape(-1, 1))/T))**2)
        vec_up = np.abs(eig_vec[probe[0], :])**2
        vec_down = np.abs(eig_vec[probe[0] + setting.tsite, :])**2
        
        rho = rho + (vec_up).dot(energy_minus) + (vec_down).dot(energy_plus)
        
        supercell_times += 1
        progress.update(1)
        sleep(0.01)
progress.close()
rho = rho/msite*(0.25/T)

with h5py.File('LDOS_' + setting.filename + '_T=' + str(T) + '.h5', 'w') as f:
    f.create_dataset('rho', data = rho)
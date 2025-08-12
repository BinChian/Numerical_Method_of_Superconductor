import numpy as np
import h5py

class readfile():
    def __init__(self, filename):
        self.filename = filename
        
    def s_wave(self, setting, lattice, file_exist = False):
        if file_exist:
            df = h5py.File(self.filename + ".h5", "r")

            pair_digArray = np.array(df['pair_dig'])
            S_pz = np.array(df['S_pz'])
            S_mz = np.array(df['S_mz'])

            pair = np.diag(pair_digArray)
        else:
            S_pz = np.zeros(setting.tsite)
            S_mz = np.zeros(setting.tsite)
            for i in range(setting.tsite):
                S_pz[i] = 0.5 + 0.1*(-1)**(lattice.ix[i])
                S_mz[i] = 0.5 - 0.1*(-1)**(lattice.ix[i])

            pair_digArray = np.linspace(0.1, 0.1, setting.tsite)
            pair = np.diag(pair_digArray)
        return pair, S_pz, S_mz

    def d_wave(self, setting, lattice, file_exist = False):
        if file_exist:
            df = h5py.File(self.filename + ".h5", "r")

            pair_px = np.array(df['pair_px'])
            pair_py = np.array(df['pair_py'])
            S_pz = np.array(df['S_pz'])
            S_mz = np.array(df['S_mz'])

            pair = np.zeros((setting.tsite, setting.tsite))
            for i in range(setting.tsite):
                pair[i, lattice.ipx[i]] = pair_px[i]
                pair[i, lattice.ipy[i]] = pair_py[i]
                pair[lattice.ipx[i], i] = pair[i, lattice.ipx[i]]
                pair[lattice.ipy[i], i] = pair[i, lattice.ipy[i]]
        else:
            S_pz = np.zeros(setting.tsite)
            S_mz = np.zeros(setting.tsite)
            for i in range(setting.tsite):
                S_pz[i] = 0.5 + 0.1*(-1)**(lattice.ix[i])
                S_mz[i] = 0.5 - 0.1*(-1)**(lattice.ix[i])

            pair_px = np.linspace(0.1, 0.1, setting.tsite)
            pair_py = np.linspace(0.1, 0.1, setting.tsite)
            pair = np.zeros((setting.tsite, setting.tsite))
            for i in range(setting.tsite):
                pair[i, lattice.ipx[i]] = pair_px[i]
                pair[i, lattice.ipy[i]] = pair_py[i]
                pair[lattice.ipx[i], i] = pair[i, lattice.ipx[i]]
                pair[lattice.ipy[i], i] = pair[i, lattice.ipy[i]]
        return pair, S_pz, S_mz

    def spm_wave(self, setting, lattice, file_exist = False):
        if file_exist:
            df = h5py.File(self.filename + ".h5", "r")

            pair_pxpy = np.array(df['pair_pxpy'])
            pair_pxmy = np.array(df['pair_pxmy'])
            S_pz = np.array(df['S_pz'])
            S_mz = np.array(df['S_mz'])

            pair = np.zeros((setting.tsite, setting.tsite))
            for i in range(setting.tsite):
                pair[i, lattice.ipxpy[i]] = pair_pxpy[i]
                pair[i, lattice.ipxmy[i]] = pair_pxmy[i]
                pair[lattice.ipxpy[i], i] = pair[i, lattice.ipxpy[i]]
                pair[lattice.ipxmy[i], i] = pair[i, lattice.ipxmy[i]]
        else:
            S_pz = np.zeros(setting.tsite)
            S_mz = np.zeros(setting.tsite)
            for i in range(setting.tsite):
                S_pz[i] = 0.5 + 0.1*(-1)**(lattice.ix[i])
                S_mz[i] = 0.5 - 0.1*(-1)**(lattice.ix[i])

            pair_pxpy = np.linspace(0.1, 0.1, setting.tsite)
            pair_pxmy = np.linspace(0.1, 0.1, setting.tsite)
            pair = np.zeros((setting.tsite, setting.tsite))
            for i in range(setting.tsite):
                pair[i, lattice.ipxpy[i]] = pair_pxpy[i]
                pair[i, lattice.ipxmy[i]] = pair_pxmy[i]
                pair[lattice.ipxpy[i], i] = pair[i, lattice.ipxpy[i]]
                pair[lattice.ipxmy[i], i] = pair[i, lattice.ipxmy[i]]
        return pair, S_pz, S_mz
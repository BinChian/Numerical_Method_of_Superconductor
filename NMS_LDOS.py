import numpy as np
import sys
import pandas as pd
import torch

def __init__():
    pass

def print_hamiltonian(H):
    np.savetxt('H_print.csv', H, delimiter = ',')
    
class Basic_Setting():
    def __init__(self, filename):
        self.imp_numbers = 0
        self.filename = filename
        
    def size_parameter(self, nx, ny, nz = 1, ns = 1):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ns = ns

        self.tsite = nx*ny*nz*ns
        
    def hopping_parameter(self, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, orbit_rotate = 0):
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.t5 = t5
        self.t6 = t6
        self.orbit_rotate = orbit_rotate

    def potential_parameter(self, mu = 0, u  = 0, jh = 0, v = 0):
        self.mu = mu
        self.u = u
        self.jh = jh
        self.u1 = u - 2.0*jh
        self.v = v

    def impurity_parameter(self, lattice, *args, imp_numbers, imp_value, d = 0):
        self.imp_numbers = imp_numbers
        self.imp_value = imp_value

        if imp_numbers == 1:
            if len(args) == 0:
                self.csite = lattice.index[self.nx//2 - 1, self.ny//2 - 1, 0, 0]
            elif len(args) == 1:
                self.csite = args[0]
            else:
                sys.exit("Error: Wrong impurity number")
            self.filename = self.filename + 'imp' + str(imp_value)
            
        elif imp_numbers == 2:
            self.d = d
            if len(args) == 0:
                self.csite1 = lattice.index[self.nx//2 - self.d//2 - 1  , self.ny//2 - 1, 0, 0]
                self.csite2 = lattice.index[self.nx//2 + (self.d - self.d//2), self.ny//2 - 1, 0, 0]
            elif len(args) == 2:
                self.csite1 = args[0]
                self.csite2 = args[1]
            else:
                sys.exit("Error: Wrong impurity number")
            self.filename = self.filename + 'imp' + str(imp_value) + 'd' + str(self.d)
            
class Lattice:
    def __init__(self, setting):
        self.nx = setting.nx
        self.ny = setting.ny
        self.nz = setting.nz
        self.ns = setting.ns
        tsite = setting.tsite

        self.i0 = np.zeros(tsite, dtype = int)
        self.ix = np.zeros(tsite, dtype = int)
        self.iy = np.zeros(tsite, dtype = int)
        self.iz = np.zeros(tsite, dtype = int)
        self.isp = np.zeros(tsite, dtype = int)
        self.ipx = np.zeros(tsite, dtype = int)
        self.imx = np.zeros(tsite, dtype = int)
        self.ipy = np.zeros(tsite, dtype = int)
        self.imy = np.zeros(tsite, dtype = int)
        self.ipxpy = np.zeros(tsite, dtype = int)
        self.imxpy = np.zeros(tsite, dtype = int)
        self.ipxmy = np.zeros(tsite, dtype = int)
        self.imxmy = np.zeros(tsite, dtype = int)
        self.ipx2 = np.zeros(tsite, dtype = int)
        self.imx2 = np.zeros(tsite, dtype = int)
        self.ipy2 = np.zeros(tsite, dtype = int)
        self.imy2 = np.zeros(tsite, dtype = int)
        self.index = np.zeros((self.nx, self.ny, self.nz, self.ns), dtype = int)

    def lattice_square(self):
            nx = self.nx
            ny = self.ny
            nz = self.nz
            ns = self.ns

            i = 0
            for iy_ in range(ny):
                for ix_ in range(nx):
                    for isp_ in range(ns):
                        for iz_ in range(nz):
                            self.i0[i] = 1
                            self.ix[i] = ix_
                            self.iy[i] = iy_
                            self.iz[i] = iz_
                            self.isp[i] = isp_
                            self.index[ix_,iy_,iz_,isp_] = i
                            
                            px = ix_ + 1
                            py = iy_ + 1
                            if ix_ == nx - 1:
                                px = 0
                            if iy_ == ny - 1:
                                py = 0
                            self.ipx[i] = (iy_)*nx*nz*ns + (px)*nz*ns + (isp_)*nz + iz_
                            self.ipy[i] = (py)*nx*nz*ns + (ix_)*nz*ns + (isp_)*nz + iz_
                            
                            mx = ix_ - 1
                            my = iy_ - 1
                            if ix_ == 0:
                                mx = nx - 1
                            if iy_ == 0:
                                my = ny - 1
                            self.imx[i] = (iy_)*nx*nz + (mx)*nz*ns + (isp_)*nz + iz_
                            self.imy[i] = (my)*nx*nz + (ix_)*nz*ns + (isp_)*nz + iz_
                            
                            self.ipxpy[i] = (py)*nx*nz*ns + (px)*nz*ns + (isp_)*nz + iz_
                            self.imxpy[i] = (py)*nx*nz*ns + (mx)*nz*ns + (isp_)*nz + iz_
                            self.imxmy[i] = (my)*nx*nz*ns + (mx)*nz*ns + (isp_)*nz + iz_
                            self.ipxmy[i] = (my)*nx*nz*ns + (px)*nz*ns + (isp_)*nz + iz_
                            
                            px2 = ix_ + 2
                            py2 = iy_ + 2
                            if ix_ == nx - 1:
                                px2 = 1
                            if iy_ == ny - 1:
                                py2 = 1
                            if ix_ == nx - 2:
                                px2 = 0
                            if iy_ == ny - 2:
                                py2 = 0
                            self.ipx2[i] = (iy_)*nx*nz*ns + (px2)*nz*ns + (isp_)*nz + iz_
                            self.ipy2[i] = (py2)*nx*nz*ns + (ix_)*nz*ns + (isp_)*nz + iz_
                            
                            mx2 = ix_ - 2
                            my2 = iy_ - 2
                            if ix_ == 1:
                                mx2 = nx - 1
                            if iy_ == 1:
                                my2 = ny - 1
                            if ix_ == 0:
                                mx2 = nx - 2
                            if iy_ == 0:
                                my2 = ny - 2
                            self.imx2[i] = (iy_)*nx*nz + (mx2)*nz*ns + (isp_)*nz + iz_
                            self.imy2[i] = (my2)*nx*nz + (ix_)*nz*ns + (isp_)*nz + iz_
                            
                            if nx == 1:
                                self.ipx[i] = 0
                            if nx == 1:
                                self.imx[i] = 0
                            if ny == 1:
                                self.ipy[i] = 0
                            if ny == 1:
                                self.imy[i] = 0
                            
                            if nx == 1 or nx == 2:
                                self.ipx2[i] = 0
                            if nx == 1 or nx == 2:
                                self.imx2[i] = 0
                            if ny == 1 or ny == 2:
                                self.ipy2[i] = 0
                            if ny == 1 or ny == 2:
                                self.imy2[i] = 0
                            
                            if nx == 1 or ny == 1:
                                self.ipxpy[i] = 0
                            if nx == 1 or ny == 1:
                                self.imxpy[i] = 0
                            if nx == 1 or ny == 1:
                                self.imxmy[i] = 0
                            if nx == 1 or ny == 1:
                                self.ipxmy[i] = 0
                            i = i + 1
    
    def lattice_vacancy(self, i):
        self.i0[i] = 0

        self.ipx[self.imx[i]] = 0
        self.imx[self.ipx[i]] = 0
        self.ipy[self.imy[i]] = 0
        self.imy[self.ipy[i]] = 0

        self.ipxpy[self.imxmy[i]] = 0
        self.imxpy[self.ipxmy[i]] = 0
        self.ipxmy[self.imxpy[i]] = 0
        self.imxmy[self.ipxpy[i]] = 0

        self.ipx[i] = 0
        self.imx[i] = 0
        self.ipy[i] = 0
        self.imy[i] = 0

        self.ipxpy[i] = 0
        self.imxpy[i] = 0
        self.ipxmy[i] = 0
        self.imxmy[i] = 0

        self.ipx2[i] = 0
        self.imx2[i] = 0
        self.ipy2[i] = 0
        self.imy2[i] = 0

    def lattice_qnummap():
        pass

    def lattice_print():
        pass

    def lattice_vacancy_qnummap():
        pass

    def lattice_vacancy_qnummap_print():
        pass
    
class Read_File():
    def __init__(self, filename):
        self.filename = filename
        
    def s_wave(self, setting, lattice, file_exist = False):
        if file_exist:
            df = pd.read_excel(self.filename + '.xlsx')

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
            df = pd.read_excel(self.filename + '.xlsx')

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
            df = pd.read_excel(self.filename + '.xlsx')

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
    
    def specify_spm_wave_file(self, filename, setting, lattice):
        df = pd.read_excel(filename + '.xlsx')

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
        return pair, S_pz, S_mz

class Selfconsistent:
    def __init__(self, T = 0.001, converge_spin = False, converge_pair = False):

        self.electrons_ = 0.0

        self.converge_spin = converge_spin
        self.alpha_spin = 0.5
        self.tol_spin = 5.0E-4

        self.converge_pair = converge_pair
        self.alpha_pair = 0.5
        self.tol_pair = 5.0E-4

        self.converge_total = self.converge_spin and self.converge_pair

        self.T = T

    # electrons
    def electrons(self, setting, eig_vec, eig_val, *args):
        if len(args) == 0:
            self.calculating_electrons(self, setting, eig_vec, eig_val)
        if len(args) == 2:
            spin_up, spin_dn = self.spin(setting, eig_vec, eig_val, args[0], args[1])
            return spin_up, spin_dn
   
    def spin(self, setting, eig_vec, eig_val, spin_up, spin_dn):
        ff = 0.5*(1-np.tanh(0.5*eig_val/self.T))
        eig_vec_up = np.split(eig_vec, 2)[0]
        eig_vec_dn = np.split(eig_vec, 2)[1]

        new_spin_up = (np.abs(eig_vec_up)**2).dot(ff)
        new_spin_dn = (np.abs(eig_vec_dn)**2).dot((1.0 - ff))
        # new_spin_up = torch.einsum('ij, j->i', torch.abs(eig_vec_up)**2, ff)
        # new_spin_dn = torch.einsum('ij, j->i', torch.abs(eig_vec_up)**2, (1.0 - ff))
            
        electrons_ = np.average(new_spin_up + new_spin_dn)
        old_electrons_ = np.average(spin_up + spin_dn)
        tol = self.tol_spin/setting.tsite
        if (np.abs(old_electrons_ - electrons_) > tol):
            self.converge_spin = False
            spin_up = spin_up*(1.0 - self.alpha_spin) + new_spin_up*self.alpha_spin
            spin_dn = spin_dn*(1.0 - self.alpha_spin) + new_spin_dn*self.alpha_spin
        else:
            self.converge_spin = True
        self.electrons_ = electrons_*setting.nz
        return spin_up, spin_dn
    
    def calculating_electrons(self, setting, eig_vec, eig_val):
        ff = 0.5*(1-np.tanh(0.5*eig_val/self.T))
        eig_vec_up = np.split(eig_vec, 2)[0]
        eig_vec_dn = np.split(eig_vec, 2)[1]

        electrons_ = np.average((np.abs(eig_vec_up)**2).dot(ff) + (np.abs(eig_vec_dn)**2).dot(1.0-ff))
        self.electrons_ = electrons_*setting.nz
    
    # s_wave_pair
    def s_wave_pair(self, lattice, tsite, eig_vec, eig_val, T, V, pair):
        if np.isreal(eig_vec).all() and np.isreal(pair).all():
            pair, converge_pair, new_pair = self.s_wave_real(lattice, tsite, eig_vec, eig_val, T, V, pair)
        else :
            pair, converge_pair, new_pair = self.s_wave_cplx(lattice, tsite, eig_vec, eig_val, T, V, pair)
        return pair, converge_pair, new_pair
    
    def s_wave_real(self, lattice, tsite, eig_vec, eig_val, T, V, pair):
        old_pair = np.diag(pair)
        new_pair = np.zeros(tsite)

        for i in range(tsite):
            new_pair[i] = (np.multiply(eig_vec[i,:],eig_vec[i+tsite,:]).dot(np.tanh(0.5*eig_val/T)))*V*0.5

        s_wave = np.sum(np.diag(pair))/tsite
        new_s_wave = np.sum(new_pair)/tsite
        tol = self.tol_pair/tsite
        if (np.abs(s_wave - new_s_wave) > tol):
            converge_pair = False
            new_pair = old_pair*(1.0-self.alpha_pair) + new_pair*self.alpha_pair
            for i in range(tsite):
                pair[i, i] = new_pair[i]
            
        else:
            converge_pair = True
        return pair, converge_pair, new_pair 

    def s_wave_cplx(self, lattice, tsite, eig_vec, eig_val, T, V, pair):
        old_pair = np.diag(pair)
        new_pair = np.zeros(tsite, dtype = np.csingle)

        for i in range(tsite):
            new_pair[i] = (np.multiply(eig_vec[i,:],np.conj(eig_vec[i+tsite,:])).dot(np.tanh(0.5*eig_val/T)))*V*0.5
        
        s_wave = np.sum(np.diag(pair))/tsite
        new_s_wave = np.sum(new_pair)/tsite
        tol = self.tol_pair/tsite
        if (np.abs(s_wave - new_s_wave) > tol):
            converge_pair = False
            new_pair = old_pair*(1.0-self.alpha_pair) + new_pair*self.alpha_pair
            for i in range(tsite):
                pair[i, i] = new_pair[i]
            
        else:
            converge_pair = True
        return pair, converge_pair, new_pair     
    
    # d_wave_pair
    def d_wave_pair(self, lattice, tsite, eig_vec, eig_val, T, V, pair):
        if np.isreal(eig_vec).all() and np.isreal(pair).all():
            pair, converge_pair, new_pair_px, new_pair_py = self.d_wave_real(lattice, tsite, eig_vec, eig_val, T, V, pair)
        else :
            pair, converge_pair, new_pair_px, new_pair_py = self.d_wave_cplx(lattice, tsite, eig_vec, eig_val, T, V, pair)
        return pair, converge_pair, new_pair_px, new_pair_py

    def d_wave_real(self, lattice, tsite, eig_vec, eig_val, T, V, pair):
        new_pair_px = np.zeros(tsite)
        new_pair_py = np.zeros(tsite)


        for i in range(tsite):
            new_pair_px[i] = (np.multiply((eig_vec[i,:], eig_vec[lattice.ipx[i]+tsite,:]))).dot(np.tanh(0.5*eig_val/T))*V*0.5
            new_pair_py[i] = (np.multiply((eig_vec[i,:], eig_vec[lattice.ipy[i]+tsite,:]))).dot(np.tanh(0.5*eig_val/T))*V*0.5
        
        pair_px = np.zeros(tsite)
        pair_py = np.zeros(tsite)
        for i in range(tsite):
                pair_px[i] = pair[i, lattice.ipx[i]]
                pair_py[i] = pair[i, lattice.ipy[i]]
        
        d_wave = np.sum(pair_px + pair_py)/tsite
        new_d_wave = np.sum(new_pair_px + new_pair_py)/tsite
        tol = self.tol_pair/tsite
        if (np.abs(d_wave - new_d_wave) > tol):
            converge_pair = False
            new_pair_px = pair_px*(1.0-self.alpha_pair) + new_pair_px*self.alpha_pair
            new_pair_py = pair_py*(1.0-self.alpha_pair) + new_pair_py*self.alpha_pair
            for i in range(tsite):
                pair[i, lattice.ipx[i]] = new_pair_px[i]
                pair[i, lattice.ipy[i]] = new_pair_py[i]
                pair[lattice.ipx[i], i] = pair[i, lattice.ipx[i]]
                pair[lattice.ipy[i], i] = pair[i, lattice.ipy[i]]
            
        else:
            converge_pair = True
        return pair, converge_pair, new_pair_px, new_pair_py

    def d_wave_cplx(self, lattice, tsite, eig_vec, eig_val, T, V, pair):
        new_pair_px = np.zeros(tsite, dtype = np.csingle)
        new_pair_py = np.zeros(tsite, dtype = np.csingle)


        for i in range(tsite):
            new_pair_px[i] = (np.multiply((eig_vec[i,:], np.conj(eig_vec[lattice.ipx[i]+tsite,:])))).dot(np.tanh(0.5*eig_val/T))*V*0.5
            new_pair_py[i] = (np.multiply((eig_vec[i,:], np.conj(eig_vec[lattice.ipy[i]+tsite,:])))).dot(np.tanh(0.5*eig_val/T))*V*0.5
        
        pair_px = np.zeros(tsite, dtype = np.csingle)
        pair_py = np.zeros(tsite, dtype = np.csingle)
        for i in range(tsite):
                pair_px[i] = pair[i, lattice.ipx[i]]
                pair_py[i] = pair[i, lattice.ipy[i]]
        
        d_wave = np.sum(pair_px + pair_py)/tsite
        new_d_wave = np.sum(new_pair_px + new_pair_py)/tsite
        tol = self.tol_pair/tsite
        if (np.abs(d_wave - new_d_wave) > tol):
            converge_pair = False
            new_pair_px = pair_px*(1.0-self.alpha_pair) + new_pair_px*self.alpha_pair
            new_pair_py = pair_py*(1.0-self.alpha_pair) + new_pair_py*self.alpha_pair
            for i in range(tsite):
                pair[i, lattice.ipx[i]] = new_pair_px[i]
                pair[i, lattice.ipy[i]] = new_pair_py[i]
                pair[lattice.ipx[i], i] = pair[i, lattice.ipx[i]]
                pair[lattice.ipy[i], i] = pair[i, lattice.ipy[i]]
            
        else:
            converge_pair = True
        return pair, converge_pair, new_pair_px, new_pair_py
    
    # spm_wave

    def spm_wave_pair(self, setting, lattice, eig_vec, eig_val, pair):
            
            vec_up = eig_vec[0:setting.tsite, :]
            vec_down_ipxpy = eig_vec[lattice.ipxpy + setting.tsite, :]
            vec_down_ipxmy = eig_vec[lattice.ipxmy + setting.tsite, :]
            
            new_pair_pxpy = np.multiply(vec_up, vec_down_ipxpy).dot(np.tanh(0.5*eig_val/self.T))*setting.v*0.5
            new_pair_pxmy = np.multiply(vec_up, vec_down_ipxmy).dot(np.tanh(0.5*eig_val/self.T))*setting.v*0.5

            pair_pxpy = pair[np.arange(pair.shape[0]), lattice.ipxpy]
            pair_pxmy = pair[np.arange(pair.shape[0]), lattice.ipxmy]

            spm_wave = np.average(pair_pxpy + pair_pxmy)
            new_spm_wave = np.average(new_pair_pxpy + new_pair_pxmy)
            tol = self.tol_pair/setting.tsite
            if (np.abs(spm_wave - new_spm_wave) > tol):
                self.converge_pair = False
                new_pair_pxpy = pair_pxpy*(1.0 - self.alpha_pair) + new_pair_pxpy*self.alpha_pair
                new_pair_pxmy = pair_pxmy*(1.0 - self.alpha_pair) + new_pair_pxmy*self.alpha_pair

                pair[np.arange(pair.shape[0]), lattice.ipxpy] = new_pair_pxpy
                pair[np.arange(pair.shape[0]), lattice.ipxmy] = new_pair_pxmy
                pair[lattice.ipxpy, np.arange(pair.shape[1])] = pair[np.arange(pair.shape[0]), lattice.ipxpy]
                pair[lattice.ipxmy, np.arange(pair.shape[1])] = pair[np.arange(pair.shape[0]), lattice.ipxmy]
            
            else:
                self.converge_pair = True
            return pair, new_pair_pxpy, new_pair_pxmy

class Hamiltonian():
    def selfconsistent(setting, lattice, pair, S_pz, S_mz):
        tsite = setting.tsite
        t1 = setting.t1
        t2 = setting.t2
        t3 = setting.t3
        t4 = setting.t4
        t5 = setting.t5
        t6 = setting.t6

        mu = setting.mu
        u  = setting.u
        jh = setting.jh
        u1 = setting.u1

        H = np.zeros((tsite*2, tsite*2), dtype = np.float32)
        iz = np.array([1, -1]*(tsite//2))
        i = np.arange(tsite)
        
        H[np.arange(tsite)      , np.arange(tsite)      ] =   -mu + u*S_mz + u1*S_mz[i+iz] + (u1-jh)*S_pz[i+iz]
        H[np.arange(tsite)+tsite, np.arange(tsite)+tsite] = -(-mu + u*S_pz + u1*S_pz[i+iz] + (u1-jh)*S_mz[i+iz])
        
        if setting.imp_numbers == 1:
            imp_site = np.array([setting.csite, setting.csite + 1])
            H[imp_site,       imp_site]       = H[imp_site,       imp_site]       + setting.imp_value
            H[imp_site+tsite, imp_site+tsite] = H[imp_site+tsite, imp_site+tsite] - setting.imp_value
        elif setting.imp_numbers == 2:
            imp_site = np.array([setting.csite1, setting.csite1 + 1, setting.csite2, setting.csite2 + 1])
            H[imp_site,       imp_site]       = H[imp_site,       imp_site]       + setting.imp_value
            H[imp_site+tsite, imp_site+tsite] = H[imp_site+tsite, imp_site+tsite] - setting.imp_value
        
        H[i, lattice.ipx] = -t1
        H[i, lattice.imx] = -t1
        H[i, lattice.ipy] = -t1
        H[i, lattice.imy] = -t1
        
        logical_array = np.logical_or(np.logical_and((lattice.ix + lattice.iy) % 2 == 0, iz == 1), np.logical_and((lattice.ix + lattice.iy) % 2 == 1, iz == -1))
        t2p = np.full(tsite, t2)
        t3p = np.full(tsite, t3)
        
        t2p[np.where(logical_array == False)[0]] = t3
        t3p[np.where(logical_array == False)[0]] = t2
        
        H[i, lattice.ipxpy] = -t2p
        H[i, lattice.imxpy] = -t3p
        H[i, lattice.imxmy] = -t2p
        H[i, lattice.ipxmy] = -t3p
        
        
        H[i, lattice.ipxpy+iz] = -t4
        H[i, lattice.imxpy+iz] = -t4
        H[i, lattice.imxmy+iz] = -t4
        H[i, lattice.ipxmy+iz] = -t4
        
        H[i, lattice.ipx+iz] = -t5
        H[i, lattice.ipy+iz] = -t5
        H[i, lattice.imx+iz] = -t5
        H[i, lattice.imy+iz] = -t5

        H[i, lattice.ipx2] = -t6
        H[i, lattice.ipy2] = -t6
        H[i, lattice.imx2] = -t6
        H[i, lattice.imy2] = -t6

        H[i+tsite, lattice.ipx+tsite] = -np.conj(H[i, lattice.ipx])
        H[i+tsite, lattice.imx+tsite] = -np.conj(H[i, lattice.imx])
        H[i+tsite, lattice.ipy+tsite] = -np.conj(H[i, lattice.ipy])
        H[i+tsite, lattice.imy+tsite] = -np.conj(H[i, lattice.imy])

        H[i+tsite, lattice.ipxpy+tsite] = -np.conj(H[i, lattice.ipxpy])
        H[i+tsite, lattice.imxpy+tsite] = -np.conj(H[i, lattice.imxpy])
        H[i+tsite, lattice.imxmy+tsite] = -np.conj(H[i, lattice.imxmy])
        H[i+tsite, lattice.ipxmy+tsite] = -np.conj(H[i, lattice.ipxmy])

        H[i+tsite, lattice.ipxpy+tsite+iz] = -np.conj(H[i, lattice.ipxpy+iz])
        H[i+tsite, lattice.imxpy+tsite+iz] = -np.conj(H[i, lattice.imxpy+iz])
        H[i+tsite, lattice.imxmy+tsite+iz] = -np.conj(H[i, lattice.imxmy+iz])
        H[i+tsite, lattice.ipxmy+tsite+iz] = -np.conj(H[i, lattice.ipxmy+iz])

        H[i+tsite, lattice.ipx+tsite+iz] = -np.conj(H[i, lattice.ipx+iz])
        H[i+tsite, lattice.ipy+tsite+iz] = -np.conj(H[i, lattice.ipy+iz])
        H[i+tsite, lattice.imx+tsite+iz] = -np.conj(H[i, lattice.imx+iz])
        H[i+tsite, lattice.imy+tsite+iz] = -np.conj(H[i, lattice.imy+iz])

        H[i+tsite, lattice.ipx2+tsite] = -np.conj(H[i, lattice.ipx2])
        H[i+tsite, lattice.ipy2+tsite] = -np.conj(H[i, lattice.ipy2])
        H[i+tsite, lattice.imx2+tsite] = -np.conj(H[i, lattice.imx2])
        H[i+tsite, lattice.imy2+tsite] = -np.conj(H[i, lattice.imy2])

        H[i, lattice.ipxpy+tsite] = pair[i, lattice.ipxpy]
        H[i, lattice.imxpy+tsite] = pair[i, lattice.imxpy]
        H[i, lattice.ipxmy+tsite] = pair[i, lattice.ipxmy]
        H[i, lattice.imxmy+tsite] = pair[i, lattice.imxmy]

        H[i+tsite, lattice.ipxpy] = pair[lattice.ipxpy, i]
        H[i+tsite, lattice.imxpy] = pair[lattice.imxpy, i]
        H[i+tsite, lattice.ipxmy] = pair[lattice.ipxmy, i]
        H[i+tsite, lattice.imxmy] = pair[lattice.imxmy, i]
        return H

    def supercell(setting, lattice, pair, S_pz, S_mz, pfx, pfy):
        t1 = setting.t1
        t2 = setting.t2
        t3 = setting.t3
        t4 = setting.t4
        t5 = setting.t5
        t6 = setting.t6

        mu = setting.mu
        u  = setting.u
        jh = setting.jh
        u1 = setting.u1

        tsite = setting.tsite

        H = np.zeros((tsite + tsite, tsite + tsite), dtype = np.complex64)
        for i in range(tsite):

            if lattice.iz[i] == 0:
                iz = 1
            elif lattice.iz[i] == 1:
                iz = -1

            H[i,      i]       =   -mu + u*S_mz[i] + u1*S_mz[i+iz] + (u1-jh)*S_pz[i+iz]
            H[i+tsite,i+tsite] = -(-mu + u*S_pz[i] + u1*S_pz[i+iz] + (u1-jh)*S_mz[i+iz])

            if setting.imp_numbers == 1:
                if (i == setting.csite) or (i == setting.csite + 1):
                    H[i,      i]       = H[i,      i]       + setting.imp_value
                    H[i+tsite,i+tsite] = H[i+tsite,i+tsite] - setting.imp_value

            elif setting.imp_numbers == 2:
                if (i == setting.csite1) or (i == setting.csite1 + 1) or (i == setting.csite2) or (i == setting.csite2 + 1):
                    H[i,      i]       = H[i,      i]       + setting.imp_value
                    H[i+tsite,i+tsite] = H[i+tsite,i+tsite] - setting.imp_value

            H[i, lattice.ipx[i]] = -t1
            H[i, lattice.imx[i]] = -t1
            H[i, lattice.ipy[i]] = -t1
            H[i, lattice.imy[i]] = -t1

            if setting.orbit_rotate == 0:
                t2p = t2
                t3p = t3

            elif setting.orbit_rotate == 90:
                if (((lattice.ix[i] + lattice.iy[i]) % 2 == 0) and (iz == 1)) or (((lattice.ix[i] + lattice.iy[i]) % 2 == 1) and (iz == -1)):
                    t2p = t2
                    t3p = t3
                else:
                    t2p = t3
                    t3p = t2

            H[i, lattice.ipxpy[i]] = -t2p
            H[i, lattice.imxpy[i]] = -t3p
            H[i, lattice.imxmy[i]] = -t2p
            H[i, lattice.ipxmy[i]] = -t3p

            H[i, lattice.ipxpy[i]+iz] = -t4
            H[i, lattice.imxpy[i]+iz] = -t4
            H[i, lattice.imxmy[i]+iz] = -t4
            H[i, lattice.ipxmy[i]+iz] = -t4
            H[i+iz, lattice.ipxpy[i]] = -t4
            H[i+iz, lattice.imxpy[i]] = -t4
            H[i+iz, lattice.imxmy[i]] = -t4
            H[i+iz, lattice.ipxmy[i]] = -t4

            H[i, lattice.ipx[i]+iz] = -t5
            H[i, lattice.ipy[i]+iz] = -t5
            H[i, lattice.imx[i]+iz] = -t5
            H[i, lattice.imy[i]+iz] = -t5
            H[i+iz, lattice.ipx[i]] = -t5
            H[i+iz, lattice.ipy[i]] = -t5
            H[i+iz, lattice.imx[i]] = -t5
            H[i+iz, lattice.imy[i]] = -t5

            H[i, lattice.ipx2[i]] = -t6
            H[i, lattice.ipy2[i]] = -t6
            H[i, lattice.imx2[i]] = -t6
            H[i, lattice.imy2[i]] = -t6

            H[i+tsite, lattice.ipx[i]+tsite] = -np.conj(H[i, lattice.ipx[i]])*pfx
            H[i+tsite, lattice.imx[i]+tsite] = -np.conj(H[i, lattice.imx[i]])*np.conj(pfx)
            H[i+tsite, lattice.ipy[i]+tsite] = -np.conj(H[i, lattice.ipy[i]])*pfy
            H[i+tsite, lattice.imy[i]+tsite] = -np.conj(H[i, lattice.imy[i]])*np.conj(pfy)

            H[i+tsite, lattice.ipxpy[i]+tsite] = -np.conj(H[i, lattice.ipxpy[i]])*pfx*pfy
            H[i+tsite, lattice.imxpy[i]+tsite] = -np.conj(H[i, lattice.imxpy[i]])*np.conj(pfx)*pfy
            H[i+tsite, lattice.imxmy[i]+tsite] = -np.conj(H[i, lattice.imxmy[i]])*np.conj(pfx)*np.conj(pfy)
            H[i+tsite, lattice.ipxmy[i]+tsite] = -np.conj(H[i, lattice.ipxmy[i]])*pfx*np.conj(pfy)

            H[i+tsite, lattice.ipxpy[i]+tsite+iz] = -np.conj(H[i, lattice.ipxpy[i]+iz])*pfx*pfy
            H[i+tsite, lattice.imxpy[i]+tsite+iz] = -np.conj(H[i, lattice.imxpy[i]+iz])*np.conj(pfx)*pfy
            H[i+tsite, lattice.imxmy[i]+tsite+iz] = -np.conj(H[i, lattice.imxmy[i]+iz])*np.conj(pfx)*np.conj(pfy)
            H[i+tsite, lattice.ipxmy[i]+tsite+iz] = -np.conj(H[i, lattice.ipxmy[i]+iz])*pfx*np.conj(pfy)
            H[i+tsite+iz, lattice.ipxpy[i]+tsite] = -np.conj(H[i+iz, lattice.ipxpy[i]])*pfx*pfy
            H[i+tsite+iz, lattice.imxpy[i]+tsite] = -np.conj(H[i+iz, lattice.imxpy[i]])*np.conj(pfx)*pfy
            H[i+tsite+iz, lattice.imxmy[i]+tsite] = -np.conj(H[i+iz, lattice.imxmy[i]])*np.conj(pfx)*np.conj(pfy)
            H[i+tsite+iz, lattice.ipxmy[i]+tsite] = -np.conj(H[i+iz, lattice.ipxmy[i]])*pfx*np.conj(pfy)

            H[i+tsite, lattice.ipx[i]+tsite+iz] = -np.conj(H[i, lattice.ipx[i]+iz])*pfx
            H[i+tsite, lattice.ipy[i]+tsite+iz] = -np.conj(H[i, lattice.ipy[i]+iz])*pfy
            H[i+tsite, lattice.imx[i]+tsite+iz] = -np.conj(H[i, lattice.imx[i]+iz])*np.conj(pfx)
            H[i+tsite, lattice.imy[i]+tsite+iz] = -np.conj(H[i, lattice.imy[i]+iz])*np.conj(pfy)
            H[i+tsite+iz, lattice.ipx[i]+tsite] = -np.conj(H[i+iz, lattice.ipx[i]])*pfx
            H[i+tsite+iz, lattice.ipy[i]+tsite] = -np.conj(H[i+iz, lattice.ipy[i]])*pfy
            H[i+tsite+iz, lattice.imx[i]+tsite] = -np.conj(H[i+iz, lattice.imx[i]])*np.conj(pfx)
            H[i+tsite+iz, lattice.imy[i]+tsite] = -np.conj(H[i+iz, lattice.imy[i]])*np.conj(pfy)

            H[i+tsite, lattice.ipx2[i]+tsite] = -np.conj(H[i, lattice.ipx2[i]])*pfx*pfx
            H[i+tsite, lattice.ipy2[i]+tsite] = -np.conj(H[i, lattice.ipy2[i]])*pfy*pfy
            H[i+tsite, lattice.imx2[i]+tsite] = -np.conj(H[i, lattice.imx2[i]])*np.conj(pfx)*np.conj(pfx)
            H[i+tsite, lattice.imy2[i]+tsite] = -np.conj(H[i, lattice.imy2[i]])*np.conj(pfy)*np.conj(pfy)

            H[i, lattice.ipx[i]] = H[i, lattice.ipx[i]]*pfx
            H[i, lattice.imx[i]] = H[i, lattice.imx[i]]*np.conj(pfx)
            H[i, lattice.ipy[i]] = H[i, lattice.ipy[i]]*pfy
            H[i, lattice.imy[i]] = H[i, lattice.imy[i]]*np.conj(pfy)

            H[i, lattice.ipxpy[i]] = H[i, lattice.ipxpy[i]]*pfx*pfy
            H[i, lattice.imxpy[i]] = H[i, lattice.imxpy[i]]*np.conj(pfx)*pfy
            H[i, lattice.imxmy[i]] = H[i, lattice.imxmy[i]]*np.conj(pfx)*np.conj(pfy)
            H[i, lattice.ipxmy[i]] = H[i, lattice.ipxmy[i]]*pfx*np.conj(pfy)

            H[i, lattice.ipxpy[i]+iz] = H[i, lattice.ipxpy[i]+iz]*pfx*pfy
            H[i, lattice.imxpy[i]+iz] = H[i, lattice.imxpy[i]+iz]*np.conj(pfx)*pfy
            H[i, lattice.imxmy[i]+iz] = H[i, lattice.imxmy[i]+iz]*np.conj(pfx)*np.conj(pfy)
            H[i, lattice.ipxmy[i]+iz] = H[i, lattice.ipxmy[i]+iz]*pfx*np.conj(pfy)
            H[i+iz, lattice.ipxpy[i]] = H[i+iz, lattice.ipxpy[i]]*pfx*pfy
            H[i+iz, lattice.imxpy[i]] = H[i+iz, lattice.imxpy[i]]*np.conj(pfx)*pfy
            H[i+iz, lattice.imxmy[i]] = H[i+iz, lattice.imxmy[i]]*np.conj(pfx)*np.conj(pfy)
            H[i+iz, lattice.ipxmy[i]] = H[i+iz, lattice.ipxmy[i]]*pfx*np.conj(pfy)

            H[i, lattice.ipx[i]+iz] = H[i, lattice.ipx[i]+iz]*pfx
            H[i, lattice.ipy[i]+iz] = H[i, lattice.ipy[i]+iz]*pfy
            H[i, lattice.imx[i]+iz] = H[i, lattice.imx[i]+iz]*np.conj(pfx)
            H[i, lattice.imy[i]+iz] = H[i, lattice.imy[i]+iz]*np.conj(pfy)
            H[i+iz, lattice.ipx[i]] = H[i+iz, lattice.ipx[i]]*pfx
            H[i+iz, lattice.ipy[i]] = H[i+iz, lattice.ipy[i]]*pfy
            H[i+iz, lattice.imx[i]] = H[i+iz, lattice.imx[i]]*np.conj(pfx)
            H[i+iz, lattice.imy[i]] = H[i+iz, lattice.imy[i]]*np.conj(pfy)

            H[i, lattice.ipx2[i]] = H[i, lattice.ipx2[i]]*pfx*pfx
            H[i, lattice.ipy2[i]] = H[i, lattice.ipy2[i]]*pfy*pfy
            H[i, lattice.imx2[i]] = H[i, lattice.imx2[i]]*np.conj(pfx)*np.conj(pfx)
            H[i, lattice.imy2[i]] = H[i, lattice.imy2[i]]*np.conj(pfy)*np.conj(pfy)

            H[i, lattice.ipxpy[i]+tsite] = pair[i, lattice.ipxpy[i]]*pfx*pfy
            H[i, lattice.imxpy[i]+tsite] = pair[i, lattice.imxpy[i]]*np.conj(pfx)*pfy
            H[i, lattice.ipxmy[i]+tsite] = pair[i, lattice.ipxmy[i]]*pfx*np.conj(pfy)
            H[i, lattice.imxmy[i]+tsite] = pair[i, lattice.imxmy[i]]*np.conj(pfx)*np.conj(pfy)
            H[i+tsite, lattice.ipxpy[i]] = pair[lattice.ipxpy[i], i]*pfx*pfy
            H[i+tsite, lattice.imxpy[i]] = pair[lattice.imxpy[i], i]*np.conj(pfx)*pfy
            H[i+tsite, lattice.ipxmy[i]] = pair[lattice.ipxmy[i], i]*pfx*np.conj(pfy)
            H[i+tsite, lattice.imxmy[i]] = pair[lattice.imxmy[i], i]*np.conj(pfx)*np.conj(pfy)
        return H
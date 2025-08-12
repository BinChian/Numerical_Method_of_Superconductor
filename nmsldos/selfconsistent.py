import numpy as np

class selfconsistent:
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
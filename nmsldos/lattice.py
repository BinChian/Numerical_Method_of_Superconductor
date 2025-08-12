import numpy as np

class lattice:
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
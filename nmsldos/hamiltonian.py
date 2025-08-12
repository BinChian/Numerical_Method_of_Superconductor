import numpy as np

class hamiltonian():
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
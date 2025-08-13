import sys
import os
import h5py

class parameter():
    def __init__(self, filename):
        self.imp_numbers = 0
        self.filename = filename
        
    def read_parameter(self):
        if os.path.exists(self.filename + '.h5'):
            print('read file ' + self.filename + '.h5 parameter.')
            with h5py.File(self.filename + '.h5', 'r') as f:
                for key, value in f['parameter'].items():
                    if value.dtype.kind == 'O':
                        self.__dict__[key] = value[()].decode('utf-8')
                        print(str(key) + ': ' + str(self.__dict__[key]))
                    else:
                        self.__dict__[key] = value[()]
                        print(str(key) + ': ' + str(self.__dict__[key]))
        
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
        # orbit_rotate for t2 <-> t3 exchange
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
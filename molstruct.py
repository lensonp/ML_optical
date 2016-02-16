import numpy as np

class MolStruct:

	def __init__(self,name_in):
		self.name = name_in
		self.readfiles()
		self.z_dict={'H':1, 'C':6, 'O':8}
		self.get_atom_types()
		self.compute_dmat()
		self.compute_cmat()

	def readfiles(self):
		self.zmax = 80 
		self.pop_type = np.zeros(self.zmax+1) #population of each type
		f = open('mol_data/{0}/{0}_svwn.xyz'.format(self.name),'r')
		self.natoms = int(f.readline())
		self.at_z = np.zeros(self.natoms)
		self.atnames = [None]*self.natoms
		f.readline()
		self.coords = np.zeros((self.natoms,3)) 
		for i in range(0,self.natoms):
			line = f.readline().split()
			self.atnames[i] = line[0]
			self.coords[i,:] = [float(line[k]) for k in [1,2,3]]
		f.close()
		
	def get_atom_types(self):
		for i in range(0,self.natoms):
			if self.atnames[i] in self.z_dict.keys():
				self.at_z[i] = self.z_dict[self.atnames[i]] 
			else:
				raise ValueError('Add atom type {0} to molstruct.py before continuing'.format(self.atnames[i]))
	
	def compute_dmat(self):
		#compute distance matrix
		#could be vectorized. runs fast enough as is.
		self.dmat = np.zeros([self.natoms,self.natoms])
		for l in range(0,self.natoms):
			for m in range(0,self.natoms):
				if ( not l == m ):
					diff = self.coords[l,:] - self.coords[m,:]
					self.dmat[l,m] = np.sqrt(np.dot(diff,diff))

	def compute_cmat(self):
		#compute Coulomb matrix
		#could be vectorized. runs fast enough as is.
		self.cmat = np.zeros([self.natoms,self.natoms])
		for l in range(0,self.natoms):
			for m in range(l,self.natoms):
				zl = self.at_z[l]	
				zm = self.at_z[m]
				if ( not l == m ):
					self.cmat[l,m] = zl*zm/self.dmat[l,m] 
					self.cmat[m,l] = zl*zm/self.dmat[l,m] 
				else:
					self.cmat[m,l] = 0.5*zl**2.4 

   


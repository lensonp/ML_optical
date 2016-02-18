import numpy as np
import pickle
import os.path 
import struct_data as sd

class Cmat_Dtb:

	def __init__(self):
		# load pickle or start a new dict
		self.cmat_file = './cmat_save.p'
		if os.path.isfile(self.cmat_file):
			print ' loading pickled cmat data.'
			f = open(self.cmat_file,'r')
			self.cmat = pickle.load(f)
			f.close()
		else:
			self.cmat = {} 

	def save(self):
		# pickle dump
		f = open(self.cmat_file,'w')
		pickle.dump(self.cmat, f)
		f.close()	

	def add_cmat(self,name_in,cmat_in):
		self.cmat[name_in] = cmat_in

	def sort_matrix(self,cmat_in):
		# sort the coulomb matrix in order of increasing row norms
		row_norms = np.linalg.norm(cmat_in,axis=1)
		dim = len(row_norms)
		perm_left = np.zeros([dim,dim])
		for i in range(0,dim):
			i_max = np.argmax(row_norms)
			row_norms[i_max] = '-inf'
			perm_left[i,i_max] = 1
		perm_right = np.transpose(perm_left)
		cmat_out = np.dot(perm_left,np.dot(cmat_in,perm_right)) 
		return cmat_out

	def pad_matrix(self,mat_in,nd):
		# pads an input matrix mat_in with zeros up to nd dimensions,
		# populates the upper left corner with mat_in 
		mat_out = np.zeros([nd,nd])
		s_in = mat_in.shape
		mat_out[0:s_in[0],0:s_in[1]] = mat_in
		return mat_out

	def cmat_matrix(self):
		# returns a matrix of sorted, padded, flattened coulomb matrices,
		# for use as input data.
		# return shape is n_samples by n_features,
		# n_features = max(n_atoms)**2
		nat_max = np.max([sd.structs[name].natoms for name in sd.mol_list])
		dim = nat_max**2
		mat_out = np.zeros([sd.nmols,int(dim)])
		for k in range(sd.nmols):
			name = sd.mol_list[k]
			cmat = self.cmat[name]
			cmat = self.sort_matrix(cmat)
			cmat = self.pad_matrix(cmat,nat_max)
			cmat = np.reshape(cmat,[1,dim])
			mat_out[k,:] = cmat
		return mat_out



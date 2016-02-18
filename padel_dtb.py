import os.path 
import numpy as np
import pickle
import struct_data as sd

class PaDel_Dtb:

	def __init__(self):
		self.pdl_file = './pdl_save.p'
		if os.path.isfile(self.pdl_file):
			print ' loading pickled PaDel data'
			f = open(self.pdl_file,'r')
			self.pdl = pickle.load(f)
			f.close()
		else:
			self.pdl = {}

	def save(self):
		f = open(self.pdl_file,'w')
		pickle.dump(self.pdl, f)
		f.close()	

	def add_pdl(self,name_in):
		# builds a dictionary of PaDel features for the input molecule
		# from mol_data/<mol>/<mol>.pdl
		pdl_file = './mol_data/{0}/{0}.pdl'.format(name_in)
		f = open(pdl_file,'r')
		line1 = f.readline()
		line2 = f.readline()
		names = line1.split(',')
		vals = line2.split(',')	
		pdl_features = {} 
		#first padel feature is just the name- leave it out
		for i in range(1,len(names)):
			f_name = names[i]
			f_val = vals[i]
			if names[i][-1:] == '\n':
				f_name = names[i][:-1]
			if vals[i][-1:] == '\n':
				f_val = vals[i][:-1]
			if ( vals[i] == 'Infinity'
			or vals[i] == '-Infinity'
			or vals[i] == ''
			or vals[i] == '\n' ):
				f_val = float('nan')
				pdl_features[f_name] = f_val
			elif np.abs(float(vals[i])) > 1E12:
				f_val = float('nan')
				pdl_features[f_name] = f_val
			else:
				try:
					f_val = float(f_val)
					pdl_features[f_name] = f_val
				except:
					raise ValueError(
						'Value {} ({}) not understood for {}: {}'.format(
						i,f_name,name_in,vals[i]))
		self.pdl[name_in] = pdl_features	

	def make_pdl_feats(self):
		# this builds a PaDel feature dict 
		# each entry in this dict is also a dict, keyed by molecule names
		print ' building PaDel descriptor dict'
		self.pdl_feats_all = {} 
		for name in sd.mol_list:
			for k in self.pdl[name].keys():
				if (not self.pdl_feats_all.has_key(k) ):
					self.pdl_feats_all[k] = {}
				self.pdl_feats_all[k][name] = self.pdl[name][k]
		k_all = self.pdl_feats_all.keys()
		nfeats = len(self.pdl_feats_all)
		print ' total number of features: {}'.format(nfeats)	
		for j in range(nfeats):
			vals_dict = self.pdl_feats_all[k_all[j]]
			#key vals_dict with mol_names to extract set of values of feature j
			nums = np.array([vals_dict[k] for k in vals_dict.keys()])
			#many features may be useless over this data set - remove them
			if ( any(np.isnan(nums)) 
				or len(vals_dict) < sd.nmols
				or all(nums==0)
				or all(nums==1)):
				del self.pdl_feats_all[k_all[j]]
		k_all = self.pdl_feats_all.keys()
		self.nfeats = len(self.pdl_feats_all)
		print ' number of useful features: {}'.format(self.nfeats)	

	def pdl_matrix(self):
		# this returns a matrix of values taken from the PaDel feature dict
		# for use as input data.
		# return shape is n_samples by n_features,
		# n_features = number of useful PaDel features over the molecule set
	        k = self.pdl_feats_all.keys()
	        pdl_mat = np.zeros([sd.nmols,self.nfeats])
	        for j in range(self.nfeats):
	                vals_dict = self.pdl_feats_all[k[j]]
	                for i in range(sd.nmols):
	                        pdl_mat[i,j] = vals_dict[sd.mol_list[i]]
		return pdl_mat



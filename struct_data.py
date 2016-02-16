from molstruct import MolStruct
import numpy as np
#import matplotlib.pyplot as plt
#import regress
#import plot_routines
#import time

print 'loading molecule structure data'

mol_file='mols.txt'
f=open(mol_file,'r')
mol_list=[]
for k,line in enumerate(f):
	mol_list.append(line.split()[0])
nmols=len(mol_list)

mols={}
pks={}
pks_comp={}

for i in range(0,nmols):
        name_i = mol_list[i]
	struct_add = MolStruct(name_i)
	mols[name_i] = struct_add
	pks_file = 'mol_data/{}/w_pk.dat'.format(name_i)
	f=open(pks_file,'r')
	pks_comp[name_i] = np.array(f.readline().split(),dtype=float) 
	pks[name_i] = np.array(f.readline().split(),dtype=float) 

print 'struct_data finished loading mols.txt'

def pkN_all(i_pk):
	pkN = []
	pkN_comp = []
	pkN_err = []
	for molname in mol_list:
		wpk=float('nan')
		wpk_comp=float('nan')
		wpk_err=float('nan')
		if len(pks[molname]) <= i_pk+1:
			if not pks_comp[molname][i_pk] == 0:
				wpk=pks[molname][i_pk]
				wpk_comp=pks[molname][i_pk]
				wpk_err=wpk_comp-wpk
		pkN.append(wpk)
		pkN_comp.append(wpk_comp)
		pkN_err.append(wpk_err)
	return np.array(pkN), np.array(pkN_comp), np.array(pkN_err)



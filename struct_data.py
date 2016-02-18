from molstruct import MolStruct
import numpy as np

print 'loading molecule structure data'

mol_file='mols.txt'
f=open(mol_file,'r')
mol_list=[]
for k,line in enumerate(f):
	mol_list.append(line.split()[0])
nmols=len(mol_list)

structs={}
pks={}
pks_comp={}

for name_i in mol_list:
	struct_add = MolStruct(name_i)
	structs[name_i] = struct_add
	pks_file = 'mol_data/{}/w_pk.dat'.format(name_i)
	f=open(pks_file,'r')
	pks_comp[name_i] = np.array(f.readline().split(),dtype=float) 
	pks[name_i] = np.array(f.readline().split(),dtype=float) 
	#print pks_comp[name_i]
	#print pks[name_i]

print 'struct_data finished loading mols.txt'

def pkN_all(i_pk):
	pkN = []
	pkN_comp = []
	pkN_err = []
	for molname in mol_list:
		if len(pks_comp[molname]) > i_pk:
			wpk=pks[molname][i_pk]
			wpk_comp=pks_comp[molname][i_pk]
			if not pks_comp[molname][i_pk] == 0:
				wpk_err=wpk_comp-wpk
			else:
				wpk_err=float('nan')
		else:
			wpk_err=float('nan')
		pkN.append(wpk)
		pkN_comp.append(wpk_comp)
		pkN_err.append(wpk_err)
	return np.array(pkN), np.array(pkN_comp), np.array(pkN_err)



from molstruct import MolStruct
import numpy as np
#import matplotlib.pyplot as plt
#import regress
#import plot_routines
#import time

mol_list = [
	'water', 		
	'methane',		
	'CO',				
	'acetylene',		
	'O2',		
	'CO2',	
	'ethane',
	'propane',
	'n_butane',
	'n_pentane',
	'n_hexane',
	'ethylene',
	'cyclopropane',
	'cyclopentadiene',
	'cyclopentene',
	'cyclohexene',
	'benzene',
	'formaldehyde',
	'methanol',
	'formic_acid',
	'ethanol',
	'isopropanol',
	'furan',		
	'tetrahydrofuran',
	'toluene',
	'phenol',
	'benzaldehyde',
	'trimethylphenol',
	'naphthalene',
	'azulene',
	'anthracene',
	'pyrene',
	'biphenylene',
	'tetracene'
	]

nmols=len(mol_list)
mols={}
funcs=['svwn','pbe','blyp','b3lyp','hse06','camb3lyp']
nfuncs=len(funcs)

print 'initializing mols:'
for i in range(0,nmols):
        name_i = mol_list[i]
	struct_add = MolStruct(name_i,funcs)
	mols[name_i] = struct_add
	#pre-processing...
	mols[name_i].read_nbo()
	mols[name_i].read_nto()
	mols[name_i].struct.compute_groups()
	mols[name_i].compute_xs_groups()
	#plot_routines.plot_groups(mols[name_i],1)
	mols[name_i].compute_ct()
	mols[name_i].peak_xs_props()
	#plot_routines.plot_molspec(mols[i].spec,1)
	for j in range(nfuncs):
		nexc_tot[j] += mols[name_i].spec.nexc[j]
		n_nbo_tot[j] += np.sum(mols[name_i].nbo_flag[j])
		n_nbo_pk_tot[j] += np.sum(mols[name_i].nbo_pk_flag[j])
	npks_tot += mols[name_i].spec.npks
print 'done!'

def barchart_peaks(fignum,mol_list_in,do_hist=False):
	props = pk_data(mol_list_in)
	#return [0:pk_id,1:w,2:sigma,3:eta,4:ct,5:dt,6:nnto,7:gt,8:deloc,9:coult,10:err]
	pk_err = props[10]
	plot_routines.barchart_peaks(fignum,pk_err,do_hist)

def barchart_corr(fignum,mol_list_in,do_hist=False):
	props = mol_data(mol_list_in)
	rlog = props[0]
	plot_routines.barchart_corr(fignum,rlog,do_hist)
	#props = spd.pk_data()


def idx_of_mol(name_in):
        for i in range(nmols):
                if mol_list[i] == name_in:
                        return i
def mol_data(mol_list_in):
	rlog = [None]*nfuncs	
	for i in range (0,nfuncs):
		rlog[i] = []
		for molname in mol_list_in:
			mol = mols[molname]
			rlog[i].append(mol.spec.rlog[i])
	return [rlog]

def pk_data(mol_list_in):
	#return [pk_id,w,sigma,eta,ct,dt,nnto,gt,deloc,coult,err]
	pk_id = [None]*nfuncs		#molname and pk indx
	w_pk = [None]*nfuncs		#absorption peaks
	sig_pk = [None]*nfuncs		#absorption peak xsections
	eta_pk = [None]*nfuncs		#broadening factors
	pk_ct = [None]*nfuncs		#pk charge transfers
	pk_dt = [None]*nfuncs		#pk distance-weighted charge transfers
	pk_nnto = [None]*nfuncs		#pk number of natural transition orbitals
	pk_gt = [None]*nfuncs		#pk group charge transfers
	pk_deloc = [None]*nfuncs    	#pk delocalizations
	pk_coult = [None]*nfuncs	#pk coulomb-weighted charge transfer 
	pk_err = [None]*nfuncs		#pk energy errors with each func
	for i in range (0,nfuncs):
		pk_id[i] = []
		w_pk[i] = []
		sig_pk[i] = []
		eta_pk[i] = []
		pk_ct[i] = []
		pk_dt[i] = []
		pk_nnto[i] = []
		pk_gt[i] = []
		pk_deloc[i] = []
		pk_coult[i] = []
		pk_err[i] = []
		for molname in mol_list_in:
			mol = mols[molname]
			for j in range(0,mol.spec.npks):
				if (not mol.spec.w_pk[i,j] == 0):
					if mol.nbo_pk_flag[i][j] == 1:
						pk_id[i].append((molname,j))	
						w_pk[i].append(mol.spec.w_pk[i,j])	
						sig_pk[i].append(mol.spec.sigma_pk[i,j])	
						eta_pk[i].append(mol.spec.eta[i,j])	
						pk_ct[i].append(mol.pk_ct[i,j])
						pk_dt[i].append(mol.pk_dt[i,j])
						pk_nnto[i].append(mol.pk_nnto[i,j])
						pk_gt[i].append(mol.pk_gt[i,j])
						pk_deloc[i].append(mol.pk_deloc[i,j])
						pk_coult[i].append(mol.pk_coult[i,j])
						pk_err[i].append(mol.spec.pk_err[i,j])
	return [pk_id,w_pk,sig_pk,eta_pk,pk_ct,pk_dt,pk_nnto,pk_gt,pk_deloc,pk_coult,pk_err]

def rlog_all(func_indx):
	return np.array([mols[n].spec.rlog[func_indx] for n in mol_list])

def pkN_err_all(func_indx,N):
	pkN_err_out = []
	for molname in mol_list:
		mol = mols[molname]
		if mol.spec.npks > N:
			pkN_err_out.append(mol.spec.pk_err[func_indx,N])
	return np.array(pkN_err_out)

def MAE_all(func_indx):
	MAE_out = []
	for molname in mol_list:
		mol = mols[molname]
		MAE_out.append(mol.spec.pk_MAE[func_indx])
	return np.array(MAE_out)

def MSigE_all(func_indx):
	MSigE_out = []
	for molname in mol_list:
		mol = mols[molname]
		MSigE_out.append(mol.spec.pk_MSigE[func_indx])
	return np.array(MSigE_out)

def corr_all(func_indx):
	corr_out = []
	for molname in mol_list:
		mol = mols[molname]
		corr_out.append(mol.spec.rlog[func_indx])
	return np.array(corr_out)

def pkN_all(N):
	pkN_out = []
	for molname in mol_list:
		mol = mols[molname]
		if mol.spec.npks > N:
			pkN_out.append(mol.spec.w_pk[nfuncs,N])
	return np.array(pkN_out)

def pkerr_all(func_indx):
	pkerr_all_out = []
	npks_good = 0
	npks_bad = 0
	for molname in mol_list:
		mol = mols[molname]
		for j in range(0,mol.spec.npks):
			if (not mol.spec.w_pk[func_indx,j] == 0):
				pkerr_all_out.append(mol.spec.pk_err[func_indx,j])
				npks_good += 1
			else:
				pkerr_all_out.append(float('nan'))
				npks_bad += 1
	print funcs[func_indx], npks_good, 'peaks good', npks_bad, 'peaks bad'
	return np.array(pkerr_all_out)

def dt_all(func_indx):
	pkdt_all = []
	xsdt_all = []
	for molname in mol_list:
		mol = mols[molname]
		for j in range(0,mol.spec.npks):
			if (not mol.spec.w_pk[func_indx,j] == 0) and mol.nbo_pk_flag[func_indx][j] == 1 and np.abs(mol.pk_dt[func_indx,j])<5:
				pkdt_all.append(mol.pk_dt[func_indx,j])
			else:
				pkdt_all.append(float('nan'))
		for j in range(0,int(mol.spec.nexc[func_indx])):
			if mol.nbo_flag[func_indx][j] == 1 and np.abs(mol.dt[func_indx][j]) < 5:
				xsdt_all.append(mol.dt[func_indx][j])
			else:
				xsdt_all.append(float('nan'))
	return np.array(pkdt_all), np.array(xsdt_all)

def dlc_all(func_indx):
	pkdlc_all = []
	xsdlc_all = []
	for molname in mol_list:
		mol = mols[molname]
		for j in range(0,mol.spec.npks):
			if (not mol.spec.w_pk[func_indx,j] == 0) and mol.nbo_pk_flag[func_indx][j] == 1:
				pkdlc_all.append(mol.pk_deloc[func_indx,j])
			else:
				pkdlc_all.append(float('nan'))
		for j in range(0,int(mol.spec.nexc[func_indx])):
			if mol.nbo_flag[func_indx][j] == 1:
				xsdlc_all.append(mol.deloc[func_indx][j])
			else:
				xsdlc_all.append(float('nan'))
	return np.array(pkdlc_all), np.array(xsdlc_all)


def ml_xs_shift(X_pk,X_xs,i,method,params):
	pkerr = pkerr_all(i)
	X_real,pkerr = regress.elim_nan(X_pk,pkerr)
	rg = ' ' 

def linreg_xs_shift(x_pk,x_xs,i):
	#i is func index
	pkerr = pkerr_all(i)
	x_real,pkerr = regress.elim_nan_1d(x_pk,pkerr)
	lr1 = np.polyfit(x_real,pkerr,1)
	pkerr_pred1 = lr1[0]*x_real + lr1[1]
	plt.figure(121)
	plt.plot(x_real,pkerr,'ro')
	plt.plot(x_real,pkerr_pred1)
	#plt.show()
	#plt.close()
	for n in mol_list:
		mol = mols[n]
		xs_shifts = np.zeros(mol.spec.nexc[i])
		pk_shifts = np.zeros(mol.spec.npks)
		for k in range(int(mol.spec.nexc[i])):
			if not np.isnan(x_xs[k]):
				xs_shifts[k] = -1*lr1[0]*x_xs[k] #- lr1[1]
		for k in range(mol.spec.npks):
			if ( not mol.spec.w_pk[i,k] == 0
				and not np.isnan(x_pk[k])):
				pk_shifts[k] = -1*lr1[0]*x_pk[k] #- lr1[1]
		mol.spec.apply_shift(i,xs_shifts,pk_shifts)
		print 'new peaks for {}, {}'.format(n,funcs[i])
		mol.spec.peaks(i)
		mol.spec.sos(mol.spec.dw)	
		mol.spec.compute_corr()
		mol.spec.compute_pkerr()

def gp_matrix():
	gp_mat_out = []
	for molname in mol_list:
		mol = mols[molname]
		mat_row = np.zeros(4)
		mat_row[0] = ( mol.struct.group_pop['atomic'] #/ mol.group_n_at('atomic') 
			+ mol.struct.group_pop['alkene'] #/ mol.group_n_at('alkene') 
			+ mol.struct.group_pop['alkyne'] #/ mol.group_n_at('alkyne') 
			+ mol.struct.group_pop['ketone'] )#/ mol.group_n_at('ketone') )
		mat_row[1] = mol.struct.group_pop['arom'] #/ mol.group_n_at('arom') 
		mat_row[2] = mol.struct.group_pop['hydroxyl'] #/ mol.group_n_at('hydroxyl') 
		mat_row[3] = mol.struct.group_pop['alkane'] #/ mol.group_n_at('alkane') 
		#mat_row[4] = mol.struct.natoms
		#group_labels = ['atomic','alkane','alkene','alkyne','hydroxyl','ketone','arom'] 
		#mat_row = np.array( [mol.struct.group_pop[k] for k in group_labels] )
		gp_mat_out.append( mat_row )
	return np.array(gp_mat_out)

def pk_nto_matrix(i):
	pknto_mat_out = []
	for molname in mol_list:
		mol = mols[molname]
		for j in range(0,mol.spec.npks):
			#pk_row = np.zeros([ngroups*2])
			pk_row = np.zeros(10)
			if (not mol.spec.w_pk[i,j] == 0
			and mol.nto_pk_flag[i][j] == 1):
				#add the pk nto of pk j 
				pk_row = pk_row + np.array(mol.nto_pk[i][j])
			else:
				pk_row[:] = float('nan')
			pknto_mat_out.append(pk_row)
	return np.array(pknto_mat_out)

def pk_orbt_matrix(i):
	pkorbt_mat_out = []
	for molname in mol_list:
		mol = mols[molname]
		#print molname
		#print np.shape(mol.pk_orbt) 
		#print np.shape(mol.pk_orbt[0]) 
		#print np.shape(mol.pk_orbt[0][0]) 
		for j in range(0,mol.spec.npks):
			if (not mol.spec.w_pk[i,j] == 0
			and mol.nbo_pk_flag[i][j] == 1):
				orbt_row = np.reshape( mol.pk_orbt[i][j],(mol.norb**2,) ) 
				if all(np.isnan(orbt_row)):
					print molname,' pk ',j,' ',funcs[i],' has nan pk orbts'
			else:
				orbt_row = float('nan')*np.ones(mol.norb**2) 

			pkorbt_mat_out.append(orbt_row)
	#print np.shape(pkorbt_mat_out)
	#print pkorbt_mat_out
	#time.sleep(5)
	return np.array(pkorbt_mat_out)

def pk_group_matrix(i):
	pkgp_mat_out = []
	for molname in mol_list:
		mol = mols[molname]
		for j in range(0,mol.spec.npks):
			#pk_row = np.zeros([ngroups*2])
			pk_row = np.zeros([ngroups])
			if (not mol.spec.w_pk[i,j] == 0
			and mol.nbo_pk_flag[i][j] == 1):
				#add the group population of the ground state? 
				#pk_row = pk_row + np.array([mol.pk_groups[i][-1][k] for k in group_labels])
				#add the group population of the excited state
				pk_row = pk_row + np.array([mol.pk_groups[i][j][k] for k in group_labels])
			else:
				pk_row[:] = float('nan')
			pkgp_mat_out.append(pk_row)
	return np.array(pkgp_mat_out)

#def pknnto_all(func_indx):
#	pknnto_all_out = []
#	for molname in mol_list:
#		mol = mols[molname]
#		for j in range(0,mol.spec.npks):
#			if (not mol.spec.w_pk[func_indx,j] == 0):
#				if mol.nto_pk_flag[func_indx][j] == 1:
#					pknnto_all_out.append(mol.pk_nnto[func_indx,j])
#				else:
#					pknnto_all_out.append(float('nan'))
#	return np.array(pknnto_all_out)

#def pkgt_all(func_indx):
#	pkgt_all_out = []
#	for molname in mol_list:
#		mol = mols[molname]
#		for j in range(0,mol.spec.npks):
#			if (not mol.spec.w_pk[func_indx,j] == 0):
#				if mol.nbo_pk_flag[func_indx][j] == 1:
#					pkgt_all_out.append(mol.pk_gt[func_indx,j])
#				else:
#					pkgt_all_out.append(float('nan'))
#	return np.array(pkgt_all_out)



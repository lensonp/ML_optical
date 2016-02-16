import numpy as np
import ML_plot_routines 
import regress
import struct_data as sd
from cmat_dtb import Cmat_Dtb
from padel_dtb import PaDel_Dtb
#import matplotlib.pyplot as plt
#import time
#import copy
#import scipy.optimize as scopt
#import pickle
#import sklearn
#from sklearn import svm
#from sklearn import cross_validation as cv
#from sklearn import linear_model
#from sklearn.cross_decomposition import PLSRegression

build_cmat=False
if build_cmat:
	for j in range(spd.nfuncs):
		for name_i in spd.mol_list:
			cdata.add_cmat(j,name_i,spd.mols[name_i].struct.cmat[j])
		#for name_i in spd.mol_list:
			#for name_j in spd.mol_list:
				#cdata.compute_cmat_dist(j,name_i,name_j)
				#cdata.compute_cmat_projdist(j,name_i,name_j)
	cdata.save()

build_pdl=False
if build_pdl:
	for name_i in spd.mol_list:
		pdldata.add_pdl(name_i)
	pdldata.make_pdl_feats()
	pdldata.make_pdl_matrix()
	for name_i in spd.mol_list:
		for name_j in spd.mol_list:
			pdldata.compute_pdl_dist(name_i,name_j)
	pdldata.save()

struct_regress=False
if struct_regress:
	pdldata.make_pdl_feats()
	pdldata.make_pdl_matrix()
	X_pdl = pdldata.pdl_mat
	y_pk0 = spd.pkN_all(0)
	y_pk1 = spd.pkN_all(1)
	X_cmat = cdata.cmat_matrix()
	y_pk0_err = spd.pkN_err_all(0)

	#############################REGRESSIONS
	#method = 'KRR-RBF'
	#method = 'KRR-SIG'
	#method = 'KRR-POLY'
	#method = 'SVR-RBF'
	#method = 'LASSO'
	method = 'RR'
	#method = 'PLS'
	params = [[2]] 	 
	#params = [[-2], [-4]] 	 
	#params = [range(1,10)]   

	#Regress padel -> pk0
	#regress.test_regressor([X_pdl],[y_pk0],method,params,stdize=True)
	#regress.run_CV([X_pdl],[y_pk0],method,params,stdize=True,val_plots=True)
	#regress.test_regressor([X_pdl],[sklearn.utils.shuffle(y_pk0)],method,params,stdize=True)
	#regress.run_CV([X_pdl],[sklearn.utils.shuffle(y_pk0)],method,params,stdize=True,val_plots=True)
	
	#Regress cmat -> pk0
	#regress.test_regressor([X_cmat[0]],[y_pk0],method,params,stdize=True)
	#regress.run_CV([X_cmat[0]],[y_pk0],method,params,stdize=True,val_plots=False)
	
	#Regress padel -> pk1
	#regress.test_regressor([X_pdl],[y_pk1],method,params,stdize=True)
	#regress.run_CV([X_pdl],[y_pk1],method,params,stdize=True,val_plots=True)
	
	#Regress padel -> pk0 error
	#regress.test_regressor([X_pdl]*len(y_pk0_err),y_pk0_err,method,params,stdize=True)
	#regress.run_CV([X_pdl]*len(y_pk0_err),y_pk0_err,method,params,stdize=True,val_plots=True)

	#Regress padel -> pk_MAE 
	#regress.test_regressor([X_pdl]*len(y_MAE),y_MAE,method,params,stdize=True)
	#regress.run_CV([X_pdl]*len(y_MAE),y_MAE,method,params,stdize=True,val_plots=True)
	
	#Regress padel -> corr 
	#regress.test_regressor([X_pdl]*len(y_MAE),y_corr,method,params,stdize=True)
	#regress.run_CV([X_pdl]*len(y_MAE),y_corr,method,params,stdize=True,val_plots=True)
	
	#Regress padel -> pk_MSigE 
	#regress.test_regressor([X_pdl]*len(y_MSigE),y_MSigE,method,params,stdize=True)
	#regress.run_CV([X_pdl]*len(y_MSigE),y_MSigE,method,params,stdize=True,val_plots=True)


	####################CLASSIFICATIONS	
	method = 'SVC-RBF'
	params = [[-1,0,1],[-4,-3,-2,-1]]  

	#Classify pk MAE bounds by padel vectors
	#regress.test_classifier([X_pdl]*len(y_MAE), regress.classify_by_limit(y_MAE,[0.3]*len(y_MAE)) ,method,params,stdize=True)
	#regress.run_CV([X_pdl]*len(y_MAE), regress.classify_by_limit(y_MAE,[0.3]*len(y_MAE)) ,method,params,stdize=True,val_plots=True)

	#Classify best xc func by padel vectors
	#with corr:
	#regress.test_classifier([X_pdl], [regress.classify_by_best(y_corr)],method,params,stdize=True)
	#regress.run_CV([X_pdl], [regress.classify_by_best(y_corr)],method,params,stdize=True,val_plots=True)
	#with MAE:
	regress.test_classifier([X_pdl], [regress.classify_by_best(y_corr)],method,params,stdize=True)
	regress.run_CV([X_pdl], [regress.classify_by_best(y_corr)],method,params,stdize=True,val_plots=True)

	#spd.barchart_peaks(1,spd.mol_list,do_hist=True)
	#spd.barchart_corr(2,spd.mol_list,do_hist=True)
	#plt.show()
	#plt.close()

	#k-means 
	ncl = [4]
	#print X_corr
	#cl,cl2,ss = regress.run_kmeans(np.reshape(X_corr[:,5],(len(X_corr[:,5]),1)), ncl, stdize=True)
	cl,cl2,ss = regress.run_kmeans(X_groups, ncl, stdize=True)
	for i in range(len(ncl)):
		print '\n\n-- {} clusters, ss = {} --'.format(ncl[i],np.mean(ss[i]))

		cluster_list = [None]*ncl[i]
		for j in range(ncl[i]):
			cluster_list[j] = []
			print '\ncluster {}: '.format(j)
			for k in range(spd.nmols):
				if cl[i][k] == j:
					print spd.mol_list[k], 'ss = ', ss[i][k]
					cluster_list[j].append( spd.mol_list[k] )
			spd.barchart_peaks(1,cluster_list[j])
			spd.barchart_corr(2,cluster_list[j])
			plt.show()
			plt.close()
			#for k in range(spd.nmols):
			#	if cl2[i][k] == j:
			#		print '(',spd.mol_list[k],')'



import numpy as np
import regress
import struct_data as sd
from cmat_dtb import Cmat_Dtb
from padel_dtb import PaDel_Dtb
import ML_plot_routines 
import time
import matplotlib.pyplot as plt
#import copy
#import scipy.optimize as scopt
#import pickle
#import sklearn
#from sklearn import svm
#from sklearn import cross_validation as cv
#from sklearn import linear_model
#from sklearn.cross_decomposition import PLSRegression

print '------------- ML optical code ----------------\n', time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()) 

cdata = Cmat_Dtb()
pdldata = PaDel_Dtb()
build_cmat=True
if build_cmat:
	print 'loading coulomb matrix data...'
	for name_i in sd.mol_list:
		cdata.add_cmat(name_i,sd.structs[name_i].cmat)
	#cdata.save() # generates a pickle file
	print 'done.'

build_pdl=True
if build_pdl:
	print 'loading PaDel descriptor data...'
	for name_i in sd.mol_list:
		pdldata.add_pdl(name_i)
	pdldata.make_pdl_feats()
	print 'done.'
	#pdldata.save() # generates a pickle file

X_prompt='Select feature vectors: \n 0: use Coulomb matrix data \n 1: use PaDel feature data \n 2: read file ML_Xvals.dat \n >> '
y_prompt='Select targets:\n 0: use absorption pk data \n 1: read file ML_yvals.dat \n >> '

run_ML=True
if run_ML:
	X_flag = input(X_prompt)
	if X_flag == 0:
		Xvals = cdata.cmat_matrix()
	elif X_flag == 1:
		Xvals = pdldata.pdl_matrix()
	elif X_flag == 2:
		print 'loading ML_Xvals.dat'
		Xvals = np.array( [ np.array(line.split(),dtype=float) for k,line in enumerate(open('ML_Xvals.dat','r')) ] )
	else:
		raise ValueError('selection not understood: {} (should be an integer between 0 and 2)'.format(X_flag))
	y_flag = int(input(y_prompt))
	if y_flag == 0:
		i_pk = int(input('which pk index? (enter non-negative integer, 0 = lowest peak) \n >> '))
		if i_pk:
			yvals,y_comp,y_err = sd.pkN_all(i_pk) 
		else:
			yvals,y_comp,y_err = sd.pkN_all(0) 
	elif y_flag == 1:
		print 'loading ML_yvals.dat'
		yvals = np.array( [ float(line) for k,line in enumerate(open('ML_yvals.dat','r')) ] ) 
		y_comp=[]
		y_err=[]
	else:
		raise ValueError('selection not understood: {} (should be an integer between 0 and 1)'.format(y_flag))

	#############################REGRESSIONS
	print 'regressing. \n X shape: {} \n y shape: {}'.format(Xvals.shape,yvals.shape)
	#print 'y: {}'.format(yvals)
	#method = 'KRR-RBF'
	#method = 'KRR-SIG'
	#method = 'KRR-POLY'
	#method = 'SVR-RBF'
	#method = 'LASSO'
	method = 'RR'
	#method = 'PLS'
	params = ([2],) 	 
	#params = [[-2], [-4]] 	 
	#params = [range(1,10)]   

	#Regress padel -> pk0
	dmean,X,y_target,y_pred = regress.test_regressor(Xvals,yvals,method,params,stdize=True)
	ML_plot_routines.plot_training(dmean,y_target,y_pred,30)
	plt.show()
	plt.close()
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
	#method = 'SVC-RBF'
	#params = [[-1,0,1],[-4,-3,-2,-1]]  

	#Classify pk MAE bounds by padel vectors
	#regress.test_classifier([X_pdl]*len(y_MAE), regress.classify_by_limit(y_MAE,[0.3]*len(y_MAE)) ,method,params,stdize=True)
	#regress.run_CV([X_pdl]*len(y_MAE), regress.classify_by_limit(y_MAE,[0.3]*len(y_MAE)) ,method,params,stdize=True,val_plots=True)

	#Classify best xc func by padel vectors
	#with corr:
	#regress.test_classifier([X_pdl], [regress.classify_by_best(y_corr)],method,params,stdize=True)
	#regress.run_CV([X_pdl], [regress.classify_by_best(y_corr)],method,params,stdize=True,val_plots=True)
	#with MAE:
	#regress.test_classifier([X_pdl], [regress.classify_by_best(y_corr)],method,params,stdize=True)
	#regress.run_CV([X_pdl], [regress.classify_by_best(y_corr)],method,params,stdize=True,val_plots=True)

	#spd.barchart_peaks(1,spd.mol_list,do_hist=True)
	#spd.barchart_corr(2,spd.mol_list,do_hist=True)
	#plt.show()
	#plt.close()

	#k-means 
	#ncl = [4]
	#print X_corr
	#cl,cl2,ss = regress.run_kmeans(np.reshape(X_corr[:,5],(len(X_corr[:,5]),1)), ncl, stdize=True)
	#cl,cl2,ss = regress.run_kmeans(X_groups, ncl, stdize=True)
	#for i in range(len(ncl)):
	#	print '\n\n-- {} clusters, ss = {} --'.format(ncl[i],np.mean(ss[i]))
	#
	#	cluster_list = [None]*ncl[i]
	#	for j in range(ncl[i]):
	#		cluster_list[j] = []
	#		print '\ncluster {}: '.format(j)
	#		for k in range(spd.nmols):
	#			if cl[i][k] == j:
	#				print spd.mol_list[k], 'ss = ', ss[i][k]
	#				cluster_list[j].append( spd.mol_list[k] )
	#		spd.barchart_peaks(1,cluster_list[j])
	#		spd.barchart_corr(2,cluster_list[j])
	#		plt.show()
	#		plt.close()
	#		#for k in range(spd.nmols):
	#		#	if cl2[i][k] == j:
	#		#		print '(',spd.mol_list[k],')'

print '------------- ML optical: exit ----------------\n', time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()) 


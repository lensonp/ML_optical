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

# USER IO 
X_prompt='Select feature vectors: \n 0: use Coulomb matrix data \n 1: use PaDel feature data \n 2: read file ML_Xvals.dat \n >> '
y_prompt='Select targets:\n 0: use absorption pk data \n 1: use computed pk data \n 2: use computed-measured pk error data \n 3: read file ML_yvals.dat \n >> '
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
		raise ValueError('selection not understood: {} '.format(X_flag))
	y_flag = int(input(y_prompt))
	if y_flag >= 0 and y_flag <= 2:
		i_pk = int(input('which pk index? (enter non-negative integer, 0 = lowest peak) \n >> '))
		yvals,y_comp,y_pkerr = sd.pkN_all(i_pk) 
		if y_flag == 1:
			yvals = y_comp
		if y_flag == 2:
			yvals = y_pkerr
	elif y_flag == 3:
		print 'loading ML_yvals.dat'
		yvals = np.array( [ float(line) for k,line in enumerate(open('ML_yvals.dat','r')) ] ) 
	else:
		raise ValueError('selection not understood: {} '.format(y_flag))

	#######CHOICE##OF##REGRESSION
	methods = ['RR','KRR','PLS','LASSO','MLP']
	#run_flags = [True,False,False,False,False]
	run_flags = [True,True,True,True,True]
	two_p_flags = [False,True,False,False,False]
	#param ranges for each method
	p0 = [ range(-3,5),	#RR l2 regularization
		range(-3,3),	#KRR l2 regularization
		range(1,10),	#PLS dimension
		range(-4,3),	#LASSO l1 regularization
		range(-3,6) ]	#MLP l2 regularization
	p1 = range(-5,2) #kernel parameter for methods that take one
	for j in range(len(methods)):
		if run_flags[j]:
			m = methods[j]
			if two_p_flags[j]:
				p = [(p0i,p1i) for p0i in p0[j] for p1i in p1]
			else:
				p = p0[j]
			n_params = len(p)

			#############################REGRESSIONS: TEST, TRAIN, VALIDATE 
			print 'testing regression... \n method: {} '.format(m)
			X,y,d,y_pred = regress.test_regressor(Xvals,yvals,m,p,stdize=True)
			print 'running leave-one-out cross-validation ... \n method: {} '.format(m)
			X,y,d,y_val = regress.cv_regressor(Xvals,yvals,m,p,stdize=True)

			#plots of training and CV error over param space:
			if two_p_flags[j]:
				np0 = len(p0[j])
				np1 = len(p1)
				y_terr = np.array( [ 
					np.array([np.mean(np.abs(y_pred[k+n*np1]-y)) for k in range(np1)])
					for n in range(np0) ] ).T
				y_cverr = np.array( [ 
					np.array([np.mean(np.abs(y_val[k+n*np1]-y)) for k in range(np1)])
					for n in range(np0) ] ).T
				ML_plot_routines.surf_error([p0[j],p1],y_terr,10)
				plt.title('mean {} training error wrt regression parameter'.format(m))
				ML_plot_routines.surf_error([p0[j],p1],y_cverr,110)
				plt.title('mean {} cross-validation error wrt regression parameter'.format(m))
			else:
				y_terr = np.array( [ np.mean(np.abs(y_pred[k]-y)) for k in range(n_params)] )
				y_cverr = np.array( [ np.mean(np.abs(y_val[k]-y)) for k in range(n_params)] )
				ML_plot_routines.plot_error(p,y_terr,10)
				plt.title('mean training error wrt regression parameter')
				ML_plot_routines.plot_error(p,y_cverr,110)
				plt.title('mean cross-validation error wrt regression parameter')
			plt.show()

			# plots of training and CV error wrt targets:
			plot_all = False
			if plot_all:
				# dangerous - going to make a lot of plots
				plot_range = range(n_params)
			else:
				# plot only the best-cross-validated model
				plot_range = [ np.argmin([ np.mean(np.abs(y_val[k]-y)) for k in range(n_params)]) ]
			for k in plot_range: 
				ML_plot_routines.plot_training(y,y_pred[k],30+k)
				plt.title('params: {}'.format(p[k]))
				ML_plot_routines.plot_validation(y,y_val[k],130+k)
				plt.title('params: {}'.format(p[k]))
			plt.show()


print '------------- ML optical: exit ----------------\n', time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()) 


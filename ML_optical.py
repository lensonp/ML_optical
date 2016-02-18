import numpy as np
import regress
import struct_data as sd
from cmat_dtb import Cmat_Dtb
from padel_dtb import PaDel_Dtb
import ML_plot_routines 
import time
import matplotlib.pyplot as plt



################################################################
##### Flags for whether or not to run parts of this script #####
# Flag for building Coulomb matrix database
build_cmat=True
# Flag for building PaDel feature database
build_PaDel=True
# Flag for running ML models
run_ML=True

##### Choice of models and parameters #####
# Models
methods = ['RR','KRR','PLS','LASSO']#,'MLP']
# A run flag for each model
run_flags = [True,True,True,True]
# A flag indicating whether each model uses two parameters 
# (e.g. regularization and kernel width)
two_p_flags = [False,True,False,False]
# Parameter values to try for each method
p0 = [ range(-3,5),	#RR l2 regularization
	range(-3,3),	#KRR l2 regularization
	range(1,10),	#PLS dimension
	range(-4,3) ]	#LASSO l1 regularization
#	range(-3,6) ]	#MLP l2 regularization
p1 = range(-5,2) #kernel parameter for methods that take one
# Warning: setting this to true and using many parameter will generate many plots
plot_all = False
################################################################



print '------------- ML optical code ----------------\n', time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()) 

# Build Coulomb Matrix and PaDel databases
cdata = Cmat_Dtb()
pdldata = PaDel_Dtb()
if build_cmat:
	print 'loading coulomb matrix data...'
	for name_i in sd.mol_list:
		cdata.add_cmat(name_i,sd.structs[name_i].cmat)
	#cdata.save() # generates a pickle file
	print 'done.'
if build_PaDel:
	print 'loading PaDel descriptor data...'
	for name_i in sd.mol_list:
		pdldata.add_pdl(name_i)
	pdldata.make_pdl_feats()
	print 'done.'
	#pdldata.save() # generates a pickle file

# User selects input data by prompts
X_prompt='Select feature vectors: \n 0: use Coulomb matrix data \n 1: use PaDel feature data \n 2: read files ML_Xvals.dat and ML_yvals.dat \n >> '
y_prompt='Select targets:\n 0: use absorption pk data \n 1: use computed pk data \n 2: use computed-measured pk error data \n >> '
if run_ML:
	X_flag = input(X_prompt)
	if X_flag == 0:
		Xvals = cdata.cmat_matrix()
	elif X_flag == 1:
		Xvals = pdldata.pdl_matrix()
	elif X_flag == 2:
		print 'loading ML_Xvals.dat'
		Xvals = np.array( [ np.array(line.split(),dtype=float) for k,line in enumerate(open('ML_Xvals.dat','r')) ] )
		yvals = np.array( [ float(line) for k,line in enumerate(open('ML_yvals.dat','r')) ] ) 
	else:
		raise ValueError('selection not understood: {} '.format(X_flag))
	if not X_flag == 2:
		y_flag = int(input(y_prompt))
		i_pk = int(input('which pk index? (enter non-negative integer, 0 = lowest peak, higher peaks = smaller data sets) \n >> '))
		y_meas,y_comp,y_pkerr = sd.pkN_all(i_pk) 
		if y_flag == 0:
			yvals = y_meas
		elif y_flag == 1:
			yvals = y_comp
		elif y_flag == 2:
			yvals = y_pkerr
		else:
			raise ValueError('selection not understood: {} '.format(y_flag))

	# For each method that is flagged true, set up parameters, 
	# test the regression, and run leave-one-out validation 
	for j in range(len(methods)):
		if run_flags[j]:
			m = methods[j]
			if two_p_flags[j]:
				p = [(p0i,p1i) for p0i in p0[j] for p1i in p1]
			else:
				p = p0[j]
			n_params = len(p)

			#############################REGRESSIONS: TEST, VALIDATE 
			print 'testing regression... \n method: {} '.format(m)
			X,y,d,y_pred = regress.test_regressor(Xvals,yvals,m,p,stdize=True)
			print 'running leave-one-out cross-validation ... \n method: {} '.format(m)
			X,y,d,y_val = regress.cv_regressor(Xvals,yvals,m,p,stdize=True)

			#plots or surfaces of training and CV error over param space:
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

			# plots of target and training/validation values 
			if plot_all:
				# dangerous - going to make plots for each parameter set
				plot_range = range(n_params)
			else:
				# plot only the best-cross-validated parameter set
				plot_range = [ np.argmin([ np.mean(np.abs(y_val[k]-y)) for k in range(n_params)]) ]
			for k in plot_range: 
				ML_plot_routines.plot_training(y,y_pred[k],30+k)
				plt.title('params: {}'.format(p[k]))
				ML_plot_routines.plot_validation(y,y_val[k],130+k)
				plt.title('params: {}'.format(p[k]))
			plt.show()

print '------------- ML optical: exit ----------------\n', time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()) 


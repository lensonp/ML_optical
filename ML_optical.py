import numpy as np
import sklearn.utils as sklu
import regress
import struct_data as sd
from cmat_dtb import Cmat_Dtb
from padel_dtb import PaDel_Dtb
import ML_plot_routines 
import time
import matplotlib.pyplot as plt



###################################################################################
##### Flags for controlling this script #####

# Flag for building Coulomb matrix database
build_cmat=True

# Flag for building PaDel feature database
build_PaDel=True

# Flag for running ML models
run_ML=True

##### Choice of models and parameters #####

# Models
methods = ['RR','KRR-RBF','PLS','LASSO']#,'MLP']	- MLP is only in sklearn-dev 0.18

# A flag for running each model
run_flags = [True,True,True,True]

# Parameter values to try for each method
params = [
[ range(-3,5) ],				#RR l2 regularization
[ range(-3,3) , range(-5,2) ],			#KRR-RBF l2 regularization and kernel resolution
[ range(1,10) ],				#PLS dimension
[ range(-4,3) ]					#LASSO l1 regularization
#[range(-3,6)]					#MLP l2 regularization	- MLP is only in sklearn-dev 0.18
]
pdim = [ len(pj) for pj in params ]


# Warning: setting this to true generates a plot for each parameter set 
plot_all = False

###################################################################################



print '------------- ML optical: start ----------------\n', time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()) 

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

if run_ML:
	# User selects input data by prompts
	X_prompt='Select feature vectors: \n 0: use Coulomb matrix data \n 1: use PaDel feature data \n 2: read files ML_Xvals.dat and ML_yvals.dat \n >> '
	y_prompt='Select targets:\n 0: use absorption pk data \n 1: use computed pk data \n 2: use computed-measured pk error data \n >> '
	X_flag = input(X_prompt)
	if X_flag == 0:
		print 'loading Coulomb matrices (high-dimensional, units e**2/angstrom)'
		Xvals = cdata.cmat_matrix()
	elif X_flag == 1:
		print 'loading PaDel descriptor matrix (high-dimensional, various units)'
		Xvals = pdldata.pdl_matrix()
	elif X_flag == 2:
		print 'loading ML_Xvals.dat and ML_yvals.dat (anonymous units)'
		Xvals = np.array( [ np.array(line.split(),dtype=float) for k,line in enumerate(open('ML_Xvals.dat','r')) ] )
		yvals = np.array( [ float(line) for k,line in enumerate(open('ML_yvals.dat','r')) ] ) 
	else:
		raise ValueError('selection not understood: {} '.format(X_flag))
	if not X_flag == 2:
		y_flag = int(input(y_prompt))
		i_pk = int(input('which pk index? (enter non-negative integer, 0 = lowest peak, higher peaks have smaller data sets) \n >> '))
		print 'loading absorption peak data (units: eV)'
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
	# test the regression, and run validations 
	for j in range(len(methods)):
		if run_flags[j]:
			m = methods[j]
			p = sklu.extmath.cartesian(params[j]) 
			np_j = len(p)
			if pdim[j] == 1: #unpack p
				p = [p[k][0] for k in range(np_j)]
			#############################REGRESSIONS: TEST, CROSS-VALIDATE, Y-RANDOMIZED VALIDATE
			print 'testing regression... \n method: {} '.format(m)
			X_s,y_s,y_pred = regress.test_regressor(Xvals,yvals,m,p)
			print 'running leave-one-out cross-validation ... \n method: {} '.format(m)
			X_s_cv,y_s_cv,y_val = regress.cv_regressor(Xvals,yvals,m,p)
			print 'running y-randomized cross-validation ... \n method: {} '.format(m)
			X_s_rand,y_s_rand,y_rand = regress.cv_regressor(Xvals,sklu.shuffle(yvals),m,p)

			#plots or surfaces of training and CV error over param space:
			if pdim[j] == 2:
				np0 = len(params[j][0])
				np1 = len(params[j][1])
				# package y values in matrices for contour plotting
				y_terr = np.array( [ np.array([np.mean(np.abs(y_pred[k+l*np1]-y_s)) for k in range(np1)]) for l in range(np0) ] ).T
				y_cverr = np.array( [ np.array([np.mean(np.abs(y_val[k+l*np1]-y_s_cv)) for k in range(np1)]) for l in range(np0) ] ).T
				y_randerr = np.array( [ np.array([np.mean(np.abs(y_rand[k+l*np1]-y_s_rand)) for k in range(np1)]) for l in range(np0) ] ).T
				ML_plot_routines.surf_error([params[j][0],params[j][1]],y_terr,10)
				plt.title('mean {} training error, standardized'.format(m))
				ML_plot_routines.surf_error([params[j][0],params[j][1]],y_cverr,110)
				plt.title('mean {} cross-validation error, standardized'.format(m))
				ML_plot_routines.surf_error([params[j][0],params[j][1]],y_randerr,210)
				plt.title('mean {} Y-RANDOMIZED cross-validation error, standardized'.format(m))
				print 'close plots to continue'
			elif pdim[j] == 1:
				y_terr = np.array( [ np.mean(np.abs(y_pred[k]-y_s)) for k in range(np_j)] )
				y_cverr = np.array( [ np.mean(np.abs(y_val[k]-y_s_cv)) for k in range(np_j)] )
				y_randerr = np.array( [ np.mean(np.abs(y_rand[k]-y_s_rand)) for k in range(np_j)] )
				ML_plot_routines.plot_error(p,y_terr,10)
				plt.title('mean {} training error, standardized'.format(m))
				ML_plot_routines.plot_error(p,y_cverr,110)
				plt.title('mean {} cross-validation error, standardized'.format(m))
				ML_plot_routines.plot_error(p,y_randerr,210)
				plt.title('mean {} Y-RANDOMIZED cross-validation error, standardized'.format(m))
				print 'close plots to continue'
			else:
				print 'no plotting routine for {} dimensions'.pdim[j]
			plt.show()

			# plots of target and training/validation values 
			if plot_all:
				# dangerous - going to make plots for each parameter set
				plot_range = range(np_j)
			else:
				# plot only the best-cross-validated parameter set
				plot_range = [ np.argmin([ np.mean(np.abs(y_val[k]-y_s_cv)) for k in range(np_j)]) ]
			for k in plot_range: 
				ML_plot_routines.plot_training(y_s,y_pred[k],30+k,output=True)
				print 'method: {} params: {}'.format(m,p[k])
				ML_plot_routines.plot_validation(y_s_cv,y_val[k],130+k,output=True)
				print 'method: {} params: {}'.format(m,p[k])
				ML_plot_routines.plot_validation(y_s_rand,y_rand[k],230+k,output=False)
				print 'method: {} params: {}, Y-RANDOMIZED'.format(m,p[k])
			print 'close plots to continue'
			plt.show()

print '------------- ML optical: exit ----------------\n', time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()) 


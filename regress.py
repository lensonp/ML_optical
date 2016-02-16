import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
#from sklearn import cross_validation as cv
import scipy.optimize as scopt
import ML_plot_routines 
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
#import spectra_data as spd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neural_network import multilayer_perceptron as mlp
#import scikit-learn.sklearn.neural_network.multilayer_perceptron.MLPRegressor
import kernels as ckr 
import time

def standardize_cols(cols):
	dims = np.shape(cols)
	if len(dims) > 1:
		ncols = dims[1]
		for i in range(ncols):
			cols[:,i] = (cols[:,i] - np.mean(cols[:,i]))/np.std(cols[:,i])
	else:
		cols = (cols - np.mean(cols))/np.std(cols)	
	return cols

def elim_nan_1d(x,y):
	xret = []
	yret = []
	print len(x)
	print len(y)
	for i in range(len(y)):
		if not np.isnan(x[i]) and not np.isnan(y[i]):
			xret.append(x[i])
			yret.append(y[i])
	return np.array(xret), np.array(yret)

def elim_const(X):
	Xret = []
	for i in range(X.shape[1]):
		if not np.std(X[:,i])==0:
			Xret.append(X[:,i])
	return np.array(Xret).T

def elim_nan(X,y):
	#assume x and y have same number of rows 
	#print 'size before elim_nan: ',len(y)
	Xret = []
	yret = []
	for i in range(len(y)):
		if not any(np.isnan(X[i,:])) and not np.isnan(y[i]):
			Xret.append(X[i,:])
			yret.append(y[i])
		#else:
		#	print 'nan data found in row ',i
		#	print X[i,np.isnan(X[i,:])]
	#print 'after elim_nan: ',len(yret)
	return np.array(Xret), np.array(yret)

def build_classifier(method,p):
	if method == 'SVC-RBF':
		cl = SVC(C=10**p[0],gamma=10**p[1]) 
	else:
		raise ValueError('method not supported: {}'.format(method))
	return cl 

def build_regressor(method,p):
	if method == 'PLS':
		rg = PLSRegression(n_components=p)
	elif method == 'LASSO':
		rg = Lasso(alpha=10**p)
	elif method == 'RR':
		rg = Ridge(alpha=10**p)
	elif method == 'KRR-RBF':
		rg = KernelRidge(alpha=10**p[0],kernel='rbf',gamma=10**p[1]) 
	elif method == 'KRR-SIG':
		rg = KernelRidge(alpha=10**p[0],kernel='sigmoid',gamma=10**p[1]) 
	elif method == 'KRR-POLY':
		rg = KernelRidge(alpha=10**p[0],kernel='poly',gamma=10**p[1],degree=3) 
	elif method == 'SVR-RBF':
		rg = SVR(C=10**p[0],gamma=10**p[1],epsilon=0.01) 
	elif method == 'KRR-CMAT':
		rg = KernelRidge(alpha=10**p[0],kernel=ckr.cmat_kernel,kernel_params={"sigma":10**p[1]}) 
	elif method == 'MLP':
		rg = mlp.MLPRegressor(alpha=10**p,activation='relu',algorithm='l-bfgs',hidden_layer_sizes=(10,)) 
	else:
		raise ValueError('method not supported: {}'.format(method))
	return rg

def classify_by_best(y_in):
	nclasses = len(y_in)
	cls_out = np.zeros(len(y_in[0]))
	for i in range(len(y_in[0])):
		yvals = [y_in[j][i] for j in range(nclasses)]
		cls_out[i] = np.argmax(yvals)
		if cls_out[i] == 0 or cls_out[i] == 1 or cls_out[i] == 2:
			cls_out[i] = 0
		else:
			cls_out[i] = 1 
	return np.reshape(cls_out,(len(y_in[0]),))

def classify_by_limit(y_in,c_limit):
	cl_out = [None]*len(y_in)
	for i in range(len(y_in)):
		cl_out[i] = np.zeros(len(y_in[i]))
		for j in range(len(y_in[i])):
			if y_in[i][j] >= c_limit[i]:
				cl_out[i][j] = 1
			else:
				cl_out[i][j] = 0
	return cl_out

def run_kmeans(X,nc_in,stdize=False):
	if stdize:
		X_s = standardize_cols(X)
	else:
		X_s = X
	cl = [None]*len(nc_in)
	cl2 = [None]*len(nc_in)
	ss = [None]*len(nc_in)
	for i in range(len(nc_in)):
		nc = nc_in[i]
		kmc = KMeans(nc)
		kmc.fit(X_s)
		cl[i] = kmc.predict(X_s)
		#print silhouette_score(X_s,cl[i]) 
		ss[i] = silhouette_samples(X_s,cl[i])
		#print np.mean(ss[i])
		cl2[i] = np.zeros(len(cl[i]))
		X_t = kmc.transform(X_s)
		for k in range(len(cl[i])):
		#	print X_t[k,:] 
			X_t[k,cl[i][k]] = float('inf')
		#	print X_t[k,:] 
			cl2[i][k] = np.argmin(X_t[k,:])
		#	print cl2[i][k]
		#for k in range(len(cl[i])):
		#	cl_expected = np.argmin(X_t[k,:])
		#	print cl[i][k], '=', cl_expected
	return cl,cl2,ss
#	for k in range(nc_in):
#		clusters[k] = []
#	for i in range(spd.nmols):
#		name = spd.mol_list[i]
#		l = kmc.predict(X_s[i,:])
#		clusters[l].append(name)
#	for k in range(nc_in):
#		print '------\ncluster {}:'.format(k)
#		print clusters[k]


def test_classifier(X,y,method,params,stdize=False):
	if len(params) > 1:
		param_set = [(p0,p1) for p0 in params[0] for p1 in params[1]]
	else:
		param_set = params[0]
	X_s = [None]*len(X)
	for i in range(len(X)):
		#X[i],y[i] = elim_nan(X[i],y[i])
		if stdize:
			X_s[i] = standardize_cols(X[i])
		else:	
			X_s[i] = X[i]
	for k in range(len(param_set)):
		p = param_set[k]
		print '--- params ---', p
		cl = build_classifier(method,p)
		for i in range(len(y)):
			print 'classifying data set', i
			cl.fit(X_s[i],y[i])
			dmean = np.zeros(len(y[i]))
			d = np.zeros(len(y[i]))
			for j in range(len(y[i])):
				d = [np.linalg.norm(X_s[i][j,:]-X_s[i][m,:]) for m in range(len(y[i]))]
				dmean[j] = np.mean( d )
			y_pred = cl.predict(X_s[i])
			y_pred = y_pred.reshape(len(y[i],))
			y_target = np.array(y[i])
			print 'number of mis-classifications in training: {}'.format(np.sum(np.abs(y_pred-y_target)))
			ML_plot_routines.plot_classification(dmean,y_target,y_pred,30)
			plt.show()
			plt.close()


def test_regressor(X,y,method,params,stdize=False):
	if len(params) > 1:
		param_set = [(p0,p1) for p0 in params[0] for p1 in params[1]]
	else:
		param_set = params[0]
	X_s = [None]*len(X)
	y_s = [None]*len(y)
	std_y = np.zeros(len(y))
	mean_y = np.zeros(len(y))
	for i in range(len(y)):
		X[i],y[i] = elim_nan(X[i],y[i])
		X[i] = elim_const(X[i])
		mean_y[i] = np.mean(y[i])
		std_y[i] = np.std(y[i])
		if stdize:
			X_s[i] = standardize_cols(X[i])
			y_s[i] = (y[i] - mean_y[i])/std_y[i]
		else:	
			X_s[i] = X[i]
			y_s[i] = y[i]
	for k in range(len(param_set)):
		p = param_set[k]
		print '--- params ---', p
		rg = build_regressor(method,p)
		#if method == 'MLP':
		#	print 'mlp n layers:'
		#	print rg.n_layers_
		for i in range(len(y)):
			print 'regressing for data set', i
			rg.fit(X_s[i],y_s[i])
			dmean = np.zeros(len(y_s[i]))
			d = np.zeros(len(y_s[i]))
			for j in range(len(y_s[i])):
				d = [np.linalg.norm(X_s[i][j,:]-X_s[i][m,:]) for m in range(len(y_s[i]))]
				dmean[j] = np.mean( d )
			y_pred = rg.predict(X_s[i])
			y_pred = y_pred.reshape(len(y_s[i],))
			if stdize:
				y_pred = y_pred*std_y[i] + mean_y[i] 
			y_target = np.array(y[i])
			print 'mean absolute training error: {}'.format(np.mean(np.abs(y_pred-y_target)))
			ML_plot_routines.plot_training(dmean,y_target,y_pred,30)
			plt.show()
			plt.close()

def cv_classifier(X,y,method,params,stdize,val_plots):
	if stdize:
		X_s = standardize_cols(X)
	else:	
		X_s = X
	if len(params) > 1:
		param_set = [(p0,p1) for p0 in params[0] for p1 in params[1]]
	else:
		param_set = params[0]
	xdist = np.zeros( [len(y),len(param_set)] )
	loocv = np.zeros( [len(y),len(param_set)] )
	for k in range(len(param_set)):
		y_val = np.zeros(len(y))
		p = param_set[k]
		print '--- params:', p
		cl = build_classifier(method,p)
		for i in range(len(y)):
			if i == 0:
				X_train = X_s[1:,:] 
				y_train = y[1:] 
			elif i == len(y)-1:
				X_train = X_s[:-1,:] 
				y_train = y[:-1] 
			else:
				X_train = np.concatenate((X_s[:i,:],X_s[i+1:,:]),axis=0)
				y_train = np.concatenate((y[:i],y[i+1:]),axis=0)
			cl.fit(X_train,y_train)
			X_test = X_s[i,:]
			y_target = y[i]
			d = [np.linalg.norm(X_test-X_train[m,:]) for m in range(len(y_train))]
			xdist[i,k] = np.mean( d )
			y_pred = cl.predict(X_train)
			y_val[i] = cl.predict(X_test)
			if y_val[i] == y_target:
				loocv[i,k] = 0
			else:
				loocv[i,k] = 1 
		print 'mis-classification rate in CV: {}'.format(np.mean(loocv[:,k]))
		if val_plots:
			ML_plot_routines.plot_classification(xdist[:,k],y,y_val,40)
			plt.show()
			plt.close()
	return loocv,xdist 

def cv_regressor(X,y,method,params,stdize,val_plots):
	y_sd = np.std(y)
	y_mean = np.mean(y)
	if stdize:
		X_s = standardize_cols(X)
		y_s = (y - y_mean)/y_sd
	else:	
		X_s = X
		y_s = y
	if len(params) > 1:
		param_set = [(p0,p1) for p0 in params[0] for p1 in params[1]]
	else:
		param_set = params[0]
	xdist = np.zeros( [len(y),len(param_set)] )
	loocv = np.zeros( [len(y),len(param_set)] )
	for k in range(len(param_set)):
		y_val = np.zeros(len(y))
		p = param_set[k]
		print '--- params:', p
		rg = build_regressor(method,p)
		for i in range(len(y)):
			if i == 0:
				X_train = X_s[1:,:] 
				y_train = y_s[1:] 
			elif i == len(y)-1:
				X_train = X_s[:-1,:] 
				y_train = y_s[:-1] 
			else:
				X_train = np.concatenate((X_s[:i,:],X_s[i+1:,:]),axis=0)
				y_train = np.concatenate((y_s[:i],y_s[i+1:]),axis=0)
			rg.fit(X_train,y_train)
			#if method == 'PLS':
			#	#compute X_test and X_train in the PLS basis
			#	Xtest_rg = rg.transform(X_test)
			#	Xtrain_rg = rg.transform(X_train) 
			#	xdist[i,k] = np.linalg.norm(Xtest_rg - np.mean(Xtrain_rg,axis=0)) 
			#else:
			#	xdist[i,k] = np.linalg.norm(X_test - np.mean(X_train,axis=0)) 
			X_test = X_s[i,:]
			d = [np.linalg.norm(X_test-X_train[m,:]) for m in range(len(y_train))]
			xdist[i,k] = np.mean( d )
			y_pred = rg.predict(X_train)
			y_val[i] = rg.predict(X_test.reshape(1,-1))
			#if we standardized the data, 
			#I still want y_target, y_val, and the cverr in the original units
			if stdize:
				y_target = y_s[i]*y_sd + y_mean
				y_val[i] = y_val[i]*y_sd + y_mean
				cverr = np.abs(y_target - y_val[i])
			else:
				y_target = y_s[i]
				cverr = np.abs(y_target - y_val[i])
			loocv[i,k] = cverr 
		print 'mean absolute validation error: {}'.format(np.mean(np.abs(y-y_val)))
		if val_plots:
			ML_plot_routines.plot_validation(xdist[:,k],y,y_val,40)
			plt.show()
			plt.close()
	return loocv,xdist 

def run_CV(X,y,method,params,stdize=True,val_plots=False):
	if not (len(X) == len(y)):
		raise ValueError('X (size {}) and y (size {}) must contain the same number of data sets'.format(len(X),len(y)))
	nsets = len(X)
	mean_y = np.zeros(nsets)
	std_y = np.zeros(nsets)
	loocv_rg = [None]*nsets
	xdist_rg = [None]*nsets
	for i in range(nsets):
		print 'CV for data set ', i
		# remove samples involving nan entries
		X[i],y[i] = elim_nan(X[i],y[i])
		# remove features that have no span 
		#y[i] = np.reshape(y[i],(-1,1))
		X[i] = elim_const(X[i])
		mean_y[i] = np.mean(y[i])
		std_y[i] = np.std(y[i])
		if method == 'SVC-RBF':
			rg_stats = cv_classifier(X[i],y[i],method,params,stdize,val_plots) 
		else:
			rg_stats = cv_regressor(X[i],y[i],method,params,stdize,val_plots) 
		loocv_rg[i] = rg_stats[0]
		xdist_rg[i] = rg_stats[1]
	print 'finished regression method {}.\nstd of target: {}'.format(method,std_y)
	if len(params) == 2:
		ML_plot_routines.surf_loocv(params,xdist_rg,loocv_rg,50)
	else:
		ML_plot_routines.plot_loocv(params,xdist_rg,loocv_rg,50)
	plt.show()
	plt.close()


def PCA_X(X_mat,npc_in,stdize=False):
	#X_mat should be n_samples by n_features
	#X_mat,y_dummy = elim_nan(X_mat,np.ones(np.shape(X_mat)[0]))
	if stdize:
		X_s = standardize_cols(X_mat)
	else:
		X_s = X_mat
	if npc_in == 0:
		pca = PCA(n_components='mle')
	else:
		pca = PCA(n_components=npc_in)
	pca.fit(X_s)
	score = pca.explained_variance_ratio_
	pc = pca.components_
	npc = pca.n_components_
	feats_new = pca.transform(X_s)
        return feats_new,score,pc

#def run_CV_DKRR(D,y,params):
#	#D and y are lists of np.arrays
#	#each list entry corresponds to a func in spd.funcs
#	#D_s = [None]*spd.nfuncs
#	#for i in range(spd.nfuncs):
#	#	D_s[i] = D[i]/np.std(D[i]) 
#	#	D_s[i] = standardize_cols(D[i])
#	std_y = np.zeros(spd.nfuncs)
#	mean_y = np.zeros(spd.nfuncs)
#	loocv_rg = [None]*spd.nfuncs
#	xdist_rg = [None]*spd.nfuncs
#	#y_s = [None]*spd.nfuncs
#	#for i in range(spd.nfuncs):
#	for i in range(1):
#		mean_y[i] = np.mean(y[i])
#		std_y[i] = np.std(y[i])
#		#y_s[i] = (y[i] - mean_y[i])/std_y[i]
#		#rg_stats = cv_distance_regressor(D_s[i],y_s[i],'D-KRR',params,False) 
#		rg_stats = cv_distance_regressor(D[i],y[i],'D-KRR',params,False) 
#		loocv_rg[i] = rg_stats[0]
#		xdist_rg[i] = rg_stats[1]
#	#plot_routines.plot_loocv(params,xdist_rg,loocv_rg,25)
#	#plt.show()
#	#plt.close()
#
#def cv_distance_regressor(D,y,method,params,loo_plots=False):
#	#D is square distance matrix, y is vector of targets with same dimension
#	xdist = np.zeros( [len(params),len(y)] )
#	loocv = np.zeros( [len(params),len(y)] )
#	for k in range(len(params)):
#		p = params[k]
#		if method == 'D-KRR':
#			#kernel can be ... 
#			rg = D_KRR(alpha=10**p,sigma=100,kernel='exp') 
#		else:
#			raise ValueError('method not supported: {}'.format(method))
#		for i in range(len(y)):
#			if i == 0:
#				D_train = D[1:,1:] 
#				y_train = y[1:] 
#				D_test = D[1:,0]
#			elif i == len(y)-1:
#				D_train = D[:-1,:-1] 
#				y_train = y[:-1] 
#				D_test = D[:-1,-1]
#			else:
#				#remove ith row:
#				D_train = np.concatenate((D[:i,:],D[i+1:,:]),axis=0)
#				#remove ith column:
#				D_train = np.concatenate((D_train[:,:i],D_train[:,i+1:]),axis=1)
#				y_train = np.concatenate((y[:i],y[i+1:]),axis=0)
#				D_test = np.concatenate((D[i,:i],D[i,i+1:]),axis=0)
#			y_test = y[i]
#			yt_mean = np.mean(y_train)
#			rg.fit(D_train,y_train)
#			#mean D_test:
#			#xdist[k,i] = np.linalg.norm(D_test) 
#			y_pred = rg.predict(D_train)
#			y_val = rg.predict(D_test)
#			yp_mean = np.mean(y_pred)
#			cverr = np.abs(y_test - y_val)
#			loocv[k,i] = cverr 
#			if loo_plots:
#				plt.figure(k)
#				#plt.plot(y_train-yt_mean,y_pred-yp_mean,'ob')
#				#plt.plot((y_test-yt_mean,y_test-yt_mean),(y_test-yt_mean,y_val-yp_mean),'or')
#				plt.plot(y_train,y_pred,'ob')
#				plt.plot(y_test,y_val,'or')
#				plt.plot((-3,3),(-3,3),'k')
#				plt.text(-2,2,'{} left out, err = {}'.format(spd.mol_list[i],cverr))
#				plt.show()
#				plt.close()
#	pls_stats = loocv,xdist
#	return pls_stats
#
#

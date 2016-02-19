import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
#from sklearn.neural_network import multilayer_perceptron as mlp

# BUILD REGRESSION OBJECTS with some default parameters and one or two input parameters
def build_regressor(method,p):
	#build a regressor object and plug in parameters p
	if method == 'RR':
		rg = Ridge(alpha=10**p)
	elif method == 'KRR-RBF':
		rg = KernelRidge(alpha=10**p[0],kernel='rbf',gamma=10**p[1]) 
	elif method == 'PLS':
		rg = PLSRegression(n_components=p)
	elif method == 'LASSO':
		rg = Lasso(alpha=10**p)
#	MLP is only available in sklearn-dev 0.18
#	elif method == 'MLP':
#		rg = mlp.MLPRegressor(alpha=10**p,activation='relu',algorithm='l-bfgs',hidden_layer_sizes=(10,)) 
	else:
		raise ValueError('method not supported: {}'.format(method))
	return rg

# test a regression technique: return the standardized training data and the predicted y
def test_regressor(X,y,method,param_set):
	# clean up any nans and standardize the data
	X_s,y_s,mean_y,std_y,n_samples = prep_data(X,y)
	# mean distance from each sample to the rest of the set: not used in this version
	# d = np.array( [ np.mean( [np.linalg.norm(X_s[j,:]-X_s[m,:]) for m in range(n_samples)] ) for j in range(n_samples) ] )
	nparams = len(param_set)
	y_pred = [None]*nparams
	for k in range(nparams):
		p = param_set[k]
		# build and fit the regressor
		rg = build_regressor(method,p)
		rg.fit(X_s,y_s)
		y_pred[k] = rg.predict(X_s)
		print 'params: {} \n mean abs training error (standard devs): {} \n (mean, std of y: {}, {})'.format(
		p,np.mean(np.abs(y_pred[k]-y_s)),mean_y,std_y)
	return X_s,y_s,y_pred

# cross-validate a regression technique: 
# return the standardized training data and the cross-predicted y
def cv_regressor(X,y,method,param_set):
	X_s,y_s,mean_y,std_y,n_samples = prep_data(X,y)
	# mean distance from each sample to the rest of the set: not used in this version
	# d = np.array( [ np.mean( [np.linalg.norm(X_s[j,:]-X_s[m,:]) for m in range(n_samples)] ) for j in range(n_samples) ] )
	nparams = len(param_set)
	y_val = [None]*nparams
	for k in range(nparams):
		y_val[k] = np.zeros([n_samples])
		p = param_set[k]
		rg = build_regressor(method,p)
		# fit rg to each of the possible LOO sets
		for m in range(n_samples):
			X_loo = np.vstack((X_s[:m],X_s[m+1:])) 
			y_loo = np.hstack((y_s[:m],y_s[m+1:])) 
			X_test = X_s[m,:]
			y_test = y_s[m]
			rg.fit(X_loo,y_loo)
			y_val[k][m] = rg.predict(X_test.reshape(1,-1)) 
		print 'params: {} \n mean abs CV error (standard devs): {} \n (mean, std of y: {}, {})'.format(
		p,np.mean(np.abs(y_val[k]-y_s)),mean_y,std_y)
	return X_s,y_s,y_val

# data preparation: remove nans, standardize 
def prep_data(X,y):
	X_r,y_r = elim_nan(X,y)
	X_r = elim_const(X_r)
	n_samples = len(y_r)
	if not n_samples == X_r.shape[0]:
		raise ValueError('X and y data do not have matching dimensions')
	mean_y = np.mean(y_r)
	std_y = np.std(y_r)
	# standardize data
	y_s = (y_r - mean_y)/std_y
	X_s = standardize_cols(X_r)
	return X_s,y_s,mean_y,std_y,n_samples

# beyond this point: minor data management routines

def standardize_cols(cols):
	# return a standardization of the input data (mean-centered and normalized by the SD)
	dims = np.shape(cols)
	if len(dims) > 1:
		ncols = dims[1]
		for i in range(ncols):
			cols[:,i] = (cols[:,i] - np.mean(cols[:,i]))/np.std(cols[:,i])
	else:
		cols = (cols - np.mean(cols))/np.std(cols)	
	return cols

def elim_const(X):
	#return the features in X that are not constant across all samples
	Xret = []
	for i in range(X.shape[1]):
		if not np.std(X[:,i])==0:
			Xret.append(X[:,i])
	return np.array(Xret).T

def elim_nan(X,y):
	#return the samples in X and y that do not contain nan
	Xret = []
	yret = []
	for i in range(len(y)):
		if not any(np.isnan(X[i,:])) and not np.isnan(y[i]):
			Xret.append(X[i,:])
			yret.append(y[i])
	return np.array(Xret), np.array(yret)

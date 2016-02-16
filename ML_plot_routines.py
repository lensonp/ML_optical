import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import linear_model as LM
import matplotlib.markers as mk
from matplotlib import gridspec
#import copy
#import regress

def surf_loocv(params,xdist,loocv,fignum):
	param0 = params[0]
	param1 = params[1] 
	for k in range(len(loocv)):
		plt.figure(fignum+k)
		plt.title('sample-mean loocv vs parameter')
		cv_z = np.zeros([len(param1),len(param0)])
		for j in range(len(param0)):
			#fill cv_z one column at a time, for each value of param0 
			for i in range(len(param1)):
				#param_set = [(p0,p1) for p0 in params[0] for p1 in params[1]]
				cv_z[i,j] = np.mean(loocv[k][:,j*len(param1)+i])
		minval = np.min(cv_z)
		maxval = np.max(cv_z)
		print('max: {} \t min: {}'.format(maxval,minval))
		cplot = plt.contour(param0,param1,cv_z,np.arange(minval,maxval,(maxval-minval)/10))
		plt.clabel(cplot, inline=1, fontsize=10)
		plt.xlabel('param 0')
		plt.ylabel('param 1')
#	plt.figure(fignum+1)
#	for k in range(len(loocv)):
#		plt.title('log parameter-mean loocv vs log parameter-mean x distance')
#		plt.plot(np.log10(np.mean(xdist[k],axis=0)),np.log10(np.mean(loocv[k],axis=0)),
#		marker='o',markerfacecolor=spd.colors[k],linestyle='none')
#	plt.figure(fignum+2)
#	for k in range(len(loocv)):
#		plt.title('log parameter-mean loocv for each sample')
#		plt.bar(np.arange(spd.nmols)+k*0.8/len(loocv),np.log10(np.mean(loocv[k],axis=0)),
#			align='center',width=0.8/len(loocv),facecolor=spd.colors[k])
#	plt.xticks(range(spd.nmols),spd.mol_list,rotation=90)

def plot_classification(dmean,y_target,y_pred,fignum):
	plt.figure(fignum)
	plt.axis([np.min(dmean)*.09,np.max(dmean)*1.1,np.min(y_target)-1,np.max(y_target)+1])
	plt.plot(dmean,y_target,'ob')
	plt.plot(dmean,y_pred,'or')
	plt.xlabel('mean dist from other samples')
	plt.ylabel('class')
	plt.legend(['target','prediction'])
	plt.figure(fignum+1)
	plt.axis([-1,len(y_target)+1,np.min(y_target)-1,np.max(y_target)+1])
	plt.plot(y_target,'ob')
	plt.plot(y_pred,'or')
	plt.ylabel('class')
	plt.legend(['target','prediction'])
	plt.savefig('./test_classifier.pdf',transparent=True)

def plot_training(dmean,y_target,y_pred,fignum):
	plt.figure(fignum)
	plt.plot(dmean,y_pred-y_target,'or')
	plt.xlabel('mean dist from other samples')
	plt.ylabel('training error')
	plt.figure(fignum+1)
	plt.plot(y_target,y_pred,'ob')
	plt.plot((np.min(y_target),np.max(y_target)),(np.min(y_target),np.max(y_target)),'k')
	plt.xlabel('target value')
	plt.ylabel('trained value')
	space = (np.max(y_target) - np.min(y_target))*0.1
	plt.text(np.min(y_target)+space,np.max(y_target)-space,'mean absolute error: {}'.format(np.mean(np.abs(y_pred-y_target))))
	plt.savefig('./test_regressor.pdf',transparent=True)

def pearson(x_in,y_in):
	npoints = len(x_in)
	mn_x = np.mean(x_in)
	mn_y = np.mean(y_in)
	sd_x = np.std(x_in,ddof=1)
	sd_y = np.std(y_in,ddof=1)
	return (1.0/(float(npoints-1)*sd_x*sd_y))*np.dot((x_in-mn_x),(y_in-mn_y))

def plot_validation(xdist,y_target,y_val,fignum):
	xdist,y_target,y_val,n_out = remove_validation_outliers(xdist,y_target,y_val)
	plt.figure(fignum)
	plt.plot(xdist,y_val-y_target,'or')
	plt.xlabel('dist from other samples')
	plt.ylabel('validation error')
	plt.figure(fignum+1)
	plt.plot((np.min(y_target),np.max(y_target)),(np.min(y_target),np.max(y_target)),'k')
	plt.plot(y_target,y_val,'or')
	space = (np.max(y_target) - np.min(y_target))*0.1
	plt.text(np.min(y_target)+space,np.max(y_target)-space,'mean absolute error: {}'.format(np.mean(np.abs(y_val-y_target))))
	plt.text(np.min(y_target)+space,np.max(y_target)-2*space,'{} samples included'.format(len(y_target)))
	plt.text(np.min(y_target)+space,np.max(y_target)-3*space,'{} samples left out'.format(n_out))
	plt.xlabel('target value')
	plt.ylabel('validation')
	plt.savefig('./validate_regressor.pdf',transparent=True)
	print 'pearson:', pearson(y_target,y_val)
	

def remove_validation_outliers(xdist,y_target,y_val):
	n_out = 0
	xd_out = []
	yt_out = []
	yv_out = []
	valerr = np.array(y_target-y_val)
	valerr_mean = np.mean(valerr)
	valerr_std = np.std(valerr)
	for i in range(len(xdist)):
		if abs(valerr[i]-valerr_mean) < 3*valerr_std:
			#print 'mean of distance out:', np.mean(dist_out)
			xd_out.append(xdist[i]) 
			yt_out.append(y_target[i])
			yv_out.append(y_val[i])
		else:
			print 'OUTLIER REMOVED - validation error is out by {} stds.'.format(
						np.abs((valerr[i]-valerr_mean)/valerr_std))
			n_out += 1
	return np.array(xd_out),np.array(yt_out),np.array(yv_out),n_out

def remove_loo_outliers(xdist,loocv):
	#loocv is numpy[nsamples by nparams] array
	n_out = 0
	xdist_out = []
	loocv_out = []
	loo_means = [np.mean(loocv[:,j]) for j in range(loocv.shape[1])]
	loo_sds = [np.std(loocv[:,j]) for j in range(loocv.shape[1])]
	#for each sample, check if its loocv is too crazy.
	#print 'mean of the sd:', np.mean(loo_sds)
	for i in range(loocv.shape[0]):
		dist_out = [np.abs(loocv[i,j] - loo_means[j]) for j in range(loocv.shape[1])]
		#if mean of dist_out > many sds away, don't keep it.
		if np.mean(dist_out) < 3*np.mean(loo_sds):
			#print 'mean of distance out:', np.mean(dist_out)
			xdist_out.append(xdist[i,:]) 
			loocv_out.append(loocv[i,:]) 
		else:
			print 'OUTLIER REMOVED - parameter-mean of LOO is {} stds.'.format(np.mean(dist_out)/np.mean(loo_sds))
			n_out += 1
	return np.array(xdist_out),np.array(loocv_out),n_out

def plot_loocv(params,xdist,loocv,fignum):
	#loocv is a list of numpy[nsamples by nparams] arrays
	#xdist is same shape as loocv 
	#row norm is mean over samples of loocv, for each parameter value (e.g. wrt nfeatures)
	#col norm is mean over parameters of loocv, for each sample
	params = np.reshape(params,(loocv[0].shape[1]))
	nsets = len(loocv)
	for k in range(nsets):
		#remove any crazy outliers...
		xdist_1,loocv_1,n_out = remove_loo_outliers(xdist[k],loocv[k])
		plt.figure(fignum+k)
		plt.title('sample-mean loocv vs parameter for data set {}'.format(k))
		loocv_mean = np.mean(loocv_1,axis=0)
		plt.plot(params,loocv_mean)
		plt.text(np.min(params),np.max(loocv_mean),'{} samples left out'.format(n_out))
		#print np.shape(np.array(params).T)
		#print np.shape(np.mean(loocv[k],axis=0))
		plt.figure(fignum+nsets+k)
		plt.title('param-mean loocv vs param-mean xdist, set {}'.format(k))
		plt.plot(np.mean(xdist_1,axis=1),np.mean(loocv_1,axis=1),'or')
#	plt.figure(fignum+2)
#	for k in range(len(loocv)):
#		plt.title('log parameter-mean loocv for each sample')
#		plt.bar(np.arange(spd.nmols)+k*0.8/len(loocv),np.log10(np.mean(loocv[k],axis=0)),
#			align='center',width=0.8/len(loocv),facecolor=spd.colors[k])
#	plt.xticks(range(spd.nmols),spd.mol_list,rotation=90)

def plot_xdist(xdist,fignum,verbose=False):
	plt.figure(fignum)
	if verbose:
		for k in range(len(xdist)):
			for j in range(np.shape(xdist[k])[0]):
		#for k in range(1):
				plt.title('LOO x log distance distribution for param {}'.format(j))
				#print xdist[k][j,:]
				h=np.histogram(np.log10(xdist[k][j,:]),bins=30)
				#print h
				plt.plot(h[1],np.concatenate(([0],h[0])),color=spd.colors[k])
		plt.show()
		plt.close()
	for k in range(len(xdist)):
		plt.title('parameter mean x distance')
		plt.bar(np.arange(spd.nmols)+k*0.8/len(xdist),np.mean(xdist[k],axis=0),
			align='center',width=0.8/len(xdist),facecolor=spd.colors[k])
	plt.xticks(range(spd.nmols),spd.mol_list,rotation=90)


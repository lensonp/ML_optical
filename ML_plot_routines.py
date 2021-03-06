import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

# compare predictions to targets for training
def plot_training(y_target,y_pred,fignum,output=False):
	plt.figure(fignum)
	plt.plot(y_target,y_pred,'ob')
	plt.plot((np.min(y_target),np.max(y_target)),(np.min(y_target),np.max(y_target)),'k')
	plt.xlabel('target value (std)',fontsize=24)
	plt.ylabel('trained value (std)',fontsize=24)
	ax = plt.gca()
	ticks_font = fm.FontProperties(family='times new roman', style='normal', size=32, weight='normal', stretch='normal')
	for label in ax.get_yticklabels():
		label.set_fontproperties(ticks_font)
	plt.tick_params(axis='y', which='major', labelsize=32)
	plt.tick_params(axis='x', which='major', labelsize=32)
	for label in ax.get_xticklabels():
		label.set_fontproperties(ticks_font)
	for label in ax.get_yticklabels():
		label.set_fontproperties(ticks_font)
	tickspec = range(-2,3)
	ax.set_yticks( tickspec )
	ax.set_yticklabels( tickspec )
	ax.set_xticks( tickspec )
	ax.set_xticklabels( tickspec )
	plt.axis([-2.9,2.9,-2.9,2.9])
	plt.tight_layout()
	if output:
		plt.savefig('./training.pdf',transparent=True)
	

# compare predictions to targets for cross validation
def plot_validation(y_target,y_val,fignum,output=False):
	plt.figure(fignum)
	# remove outliers that validate extremely poorly (> 3 standard deviations) and note them on the plot 
	y_target,y_val,n_out = remove_validation_outliers(y_target,y_val)
	plt.plot(y_target,y_val,'or')
	plt.plot((np.min(y_target),np.max(y_target)),(np.min(y_target),np.max(y_target)),'k')
	space = (np.max(y_target)-np.min(y_target))*0.1
	#plt.text(np.min(y_target)+space, np.max(y_target)-space, '{} outliers removed'.format(n_out)) 
	print '{} outliers removed'.format(n_out) 
	#plt.text( np.min(y_target)+space, np.max(y_target)-2*space, 'mean err of remaining: {}'.format(np.mean(np.abs(y_val-y_target))) ) 
	print 'mean err of remaining: {}'.format(np.mean(np.abs(y_val-y_target))) 
	plt.xlabel('target value (std)',fontsize=24)
	plt.ylabel('cross-validated value (std)',fontsize=24)
	ax = plt.gca()
	ticks_font = fm.FontProperties(family='times new roman', style='normal', size=32, weight='normal', stretch='normal')
	for label in ax.get_yticklabels():
		label.set_fontproperties(ticks_font)
	plt.tick_params(axis='y', which='major', labelsize=32)
	plt.tick_params(axis='x', which='major', labelsize=32)
	for label in ax.get_xticklabels():
		label.set_fontproperties(ticks_font)
	for label in ax.get_yticklabels():
		label.set_fontproperties(ticks_font)
	tickspec = range(-2,3)
	ax.set_yticks( tickspec )
	ax.set_yticklabels( tickspec )
	ax.set_xticks( tickspec )
	ax.set_xticklabels( tickspec )
	plt.axis([-2.9,2.9,-2.9,2.9])
	plt.tight_layout()
	if output:
		plt.savefig('./validation.pdf',transparent=True)

# plot or surface of (training or validation) error over selected parameters
def plot_error(params,y_err,fignum):
	plt.figure(fignum)
	plt.plot(params,y_err)
	plt.xlabel('param 0')
	plt.ylabel('y error (standard devs)')
def surf_error(params,y_err,fignum):
	plt.figure(fignum)
	minval = np.min(y_err)
	maxval = np.max(y_err)
	cplot = plt.contour(params[0],params[1],y_err,np.arange(minval,maxval,(maxval-minval)/10))
	plt.clabel(cplot, inline=1, fontsize=10)
	plt.xlabel('param 0')
	plt.ylabel('param 1')

# remove validation tests that lie more than 3 standard deviations outside the mean validation error
def remove_validation_outliers(y_target,y_val):
	n_out = 0
	yt_out = []
	yv_out = []
	valerr = np.array(y_target-y_val)
	valerr_mean = np.mean(valerr)
	valerr_std = np.std(valerr)
	for i in range(len(y_target)):
		if abs(valerr[i]-valerr_mean) < 3*valerr_std:
			#print 'mean of distance out:', np.mean(dist_out)
			yt_out.append(y_target[i])
			yv_out.append(y_val[i])
		else:
			print 'OUTLIER REMOVED - validation error is out by {} stds.'.format(
						np.abs((valerr[i]-valerr_mean)/valerr_std))
			n_out += 1
	return np.array(yt_out),np.array(yv_out),n_out


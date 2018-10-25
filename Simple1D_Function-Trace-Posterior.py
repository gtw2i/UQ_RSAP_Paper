# Graham West
from copy import deepcopy
import sys 
import random
import numpy as np
import math
import pandas as pd
from subprocess import call
from scipy import optimize
from scipy import misc
from matplotlib import pyplot as plt
from matplotlib import image as img

##############
#    MAIN    #
##############

def main():
	
	dataFile = "MultiStepTest_NoAdapt_13.txt"
	
	nGenP    = 501
	nBin     = 45
	
	nStep    = 5000
	nDim     = 1
	choice   = 0
	toMod    = 1
	
	if( toMod ):
		accMod  = 1.0
		modAmp  = [ 0.1, 10.0 ]
		modRate = [ 0.3, 0.3, 2000, 1000 ]
	else:
		accMod  = 1.0
		modAmp  = [ 1.0, 1.0 ]
		modRate = [ 0.0, 0.0, 0.0 ]
	# end
	
	x_start = np.zeros(nDim+1)
	# Gaussian 
	if( choice == 0 ):
		thresh  = 0.003
		initSig = 0.08
		p       = [0.00,  1.0,0.3,  0.4,2.5, 0.4,2.5 ]
		func = "Gaussian"
		
		xLim = []
		j_sig = []
		dj_sig = []
		for i in range(nDim):
			xLim.append( [-1.0, 1.0] )
			j_sig.append(  0.10 )
			dj_sig.append( 0.00 )
		# end
		xLim.append( [0, 100] )
		j_sig.append( 0.0008 )
		dj_sig.append( 0.0 )
		
		for j in range(nDim):
#			x_start[j] = 0.5*np.cos(i*2*np.pi/nGen)
			x_start[j] = 0.8
#			x_start[j] = np.random.uniform(xLim[j,0],xLim[j,1])
		# end
		x_start[-1] = initSig
	# rosenbrock
	elif( choice == 1 ):
		thresh  = 0.02
		initSig = 0.4
		p       = [0.00,  1.0,2.0,  0.4,2.5, 0.4,2.5 ]
		func = "Rosenbrock"
		
		xLim = []
		j_sig = []
		dj_sig = []
		for i in range(nDim):
			xLim.append( [-2.0, 2.0] )
			j_sig.append(  0.20 )
			dj_sig.append( 0.04 )
		# end
		xLim.append( [0, 100] )
		j_sig.append( 0.05 )
		dj_sig.append( 0.0 )
		
		for j in range(nDim):
#			x_start[j] = 0.5*np.cos(i*2*np.pi/nGen)
			x_start[j] = 1.0
#			x_start[j] = np.random.uniform(xLim[j,0],xLim[j,1])
		# end
		x_start[-1] = initSig
	# Ackley
	elif( choice == 2 ):
		thresh  = 0.8
		initSig = 3.5
		p       = [0.00,  1.0,2.0,  0.4,2.5, 0.4,2.5 ]
		func = "Ackley"
		
		xLim = []
		j_sig = []
		dj_sig = []
		for i in range(nDim):
			xLim.append( [-10.0, 10.0] )
			j_sig.append(  1.0 )
			dj_sig.append( 0.3 )
		# end
		xLim.append( [0, 100] )
		j_sig.append( 0.01 )
		dj_sig.append( 0.0 )
		
		for j in range(nDim):
#			x_start[j] = 0.5*np.cos(i*2*np.pi/nGen)
			x_start[j] = 8.0
#			x_start[j] = np.random.uniform(xLim[j,0],xLim[j,1])
		# end
		x_start[-1] = initSig
	# bi-modal
	elif( choice == 3 ):
		thresh  = 0.003
		initSig = 0.08
		p       = [0.00,  0.5,0.15,  8.0,0.333, 0.4,2.5 ]
		func = "Bimodal"
		
		xLim = []
		j_sig = []
		dj_sig = []
		for i in range(nDim):
			xLim.append( [-1.0, 1.0] )
			j_sig.append(  0.10 )
			dj_sig.append( 0.00 )
		# end
		xLim.append( [0, 100] )
		j_sig.append( 0.0008 )
		dj_sig.append( 0.0 )
		
		for j in range(nDim):
#			x_start[j] = 0.5*np.cos(i*2*np.pi/nGen)
			x_start[j] = 0.8
#			x_start[j] = np.random.uniform(xLim[j,0],xLim[j,1])
		# end
		x_start[-1] = initSig
	# many minima, gaussian
	elif( choice == 4 ):
		thresh  = 0.025
		initSig = 0.04
		p       = [0.00,  1.0,3.0,  0.3,2.5, 0.3,2.5 ]
		func = "ManyMins"
		
		xLim = []
		j_sig = []
		dj_sig = []
		for i in range(nDim):
			xLim.append( [-1.0, 1.0] )
			j_sig.append(  0.10 )
			dj_sig.append( 0.02 )
		# end
		xLim.append( [0, 100] )
		j_sig.append( 0.00005 )
		dj_sig.append( 0.0 )
		
		for j in range(nDim):
#			x_start[j] = 0.5*np.cos(i*2*np.pi/nGen)
			x_start[j] = 0.8
#			x_start[j] = np.random.uniform(xLim[j,0],xLim[j,1])
		# end
		x_start[-1] = initSig
	# end
	p       = np.array(p)
	xLim    = np.array(xLim)*1.0
	j_sig   = np.array(j_sig)*1.0
	
	param_toFit = range(nDim+1)
	fileBase = "OutFastMetro"
	progress = 0
	
	sigUsed  = []
	chainRaw = []
	chainAcc = []
	chainImp = []
	
	k = 0
	i = 0
	filename = ""
	if( k < 10 ):
		filename += fileBase + "_0" + str(k)
	else:
		filename += fileBase + "_" + str(k)
	# end
	if( i < 10 ):
		filename += "_000" + str(i) + ".txt"
	elif( i < 100 ):
		filename += "_00" + str(i) + ".txt"
	elif( i < 1000 ):
		filename += "_0" + str(i) + ".txt"
	else:
		filename += "_" + str(i) + ".txt"
	# end
	
	chain, accRate, max_ps, max_lp, jump = metropolis( x_start, p, xLim, j_sig, nStep, param_toFit, filename, 0, progress, modRate,modAmp,accMod,toMod,choice)
	
	w, x, y, z = getConv( filename, thresh, nStep, nDim )
	chainRaw.append(x)
	chainAcc.append(y)
	chainImp.append(z)
	
	# end
	
	chainRaw = np.array(chainRaw)[0,:,0]
	chainAcc = np.array(chainAcc)[0,:,0]
	chainImp = np.array(chainImp)[0,:,0]
	
	# PLOTTING
	# wide
#	fig, axes = plt.subplots( nrows=int(1), ncols=int(2), figsize=(18,12) )
#	fig, ax = plt.subplots( nrows=int(1), ncols=int(1), figsize=(18,12) )
	# normal
#	fig, axes = plt.subplots( nrows=int(2), ncols=int(3), figsize=(12,12) )
	# eMachine
#	fig, axes = plt.subplots( nrows=int(2), ncols=int(3), figsize=(9,9) )
	fig, ax = plt.subplots()
	
	fileInd = "00"
	
	plt.rc( 'text', usetex=True )
	plt.rc( 'font', family='serif' )
	
#	asd
	
	ind2 = 0
	for ind2 in range(2):
		plt.cla()
		
		if( ind2 == 0 ):
			# visualize f
			y = np.zeros((nDim,nGenP))
			xPlt = np.zeros(nGenP)
			for i in range(nDim):
				x = np.zeros(nDim+1)
				for j in range(nGenP):
					x[i] = xLim[i,0] + (xLim[i,1]-xLim[i,0])*j/(nGenP-1.0)
					y[i,j] = f(x, p, choice)
					if( i == 0 ):
						xPlt[j] = x[i]
					# end
				# end
			# end
			
			for i in range(nDim):
				ax.plot(xPlt, y[i,:], 'b')
			# end
			
			for tick in ax.xaxis.get_major_ticks():
				tick.label.set_fontsize(14)
			# end
			for tick in ax.yaxis.get_major_ticks():
				tick.label.set_fontsize(14)
			# end
			
			ax.set_xlabel(r'$\theta$', fontsize=16)
			ax.set_ylabel("error", fontsize=16)
			plt.savefig("UQPlot_" + func + "_" + fileInd + "_Function.jpg")
#			plt.savefig("PosterPlot_Ackley_Function_" + width + "_" + fileInd + ".jpg")
		# end
		if( ind2 == 1 ):
#			ax.plot(chainRaw)
			ax.plot(chainAcc, 'b', linewidth=0.4)
			ax.plot(chainImp, 'r')
			
			for tick in ax.xaxis.get_major_ticks():
				tick.label.set_fontsize(14)
			# end
			for tick in ax.yaxis.get_major_ticks():
				tick.label.set_fontsize(14)
			# end
			
			ax.set_ylim(xLim[0])
			ax.set_xlabel("steps", fontsize=16)
			ax.set_ylabel(r'$\theta$', fontsize=16)
			plt.savefig("UQPlot_" + func + "_" + fileInd + "_Trace.jpg")
#			plt.savefig("PosterPlot_Ackley_Chain_" + width + "_" + fileInd + ".jpg")
		# end
		if( ind2 == 2 ):
			burn = 0
			binC = BinForTargetDist(nBin, nStep-burn, xLim, chainAcc[burn:])
			binC = binC/(1.0*np.sum(binC))
			x = []
			for i in range(nBin):
				x.append( xLim[0,0] + (xLim[0,1]-xLim[0,0])*i/(nBin-1.0) )
			# end
			
			for tick in ax.xaxis.get_major_ticks():
				tick.label.set_fontsize(14)
			# end
			for tick in ax.yaxis.get_major_ticks():
				tick.label.set_fontsize(14)
			# end
			
			ax.plot(x,binC, 'b')
			ax.set_xlabel(r'$\theta$', fontsize=16)
			ax.set_ylabel("posterior distribution", fontsize=16)
			plt.savefig("UQPlot_" + func + "_" + fileInd + "_Posterior.jpg")
#			plt.savefig("PosterPlot_Ackley_Dist_" + width + "_" + fileInd + ".jpg")
		# end
	# end
	
#	plt.tight_layout(w_pad=0.0, h_pad=0.0)
	plt.show()
	
# end

def f( x, a, choice ):
	
	nPar = len(x)-1
	
#	np.random.seed(int(100*np.abs(x[0])+1000000*np.abs(x[1])))
	
	"""
	q = 0
	for i in range(nPar):
		q += x[i]**2
	# end
	"""
	
	# Gaussian
	if( choice == 0 ):
		q = 0
		for i in range(nPar):
			q += x[i]**2
		# end
		y = a[1]*(1.0-np.exp(-0.5*(q/a[2]**2)**(2.0/2.0)))
	# Rosenbrock
	elif( choice == 1 ):
		y = 0
		for i in range(nPar-1):
			y += 100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
		# end
	# Ackley
	elif( choice == 2 ):
		A = 20.0
		B = 4.0
		q1 = 0
		q2 = 0
		for i in range(nPar):
			q1 += x[i]**2
			q2 += np.cos(2*np.pi*x[i])
		# end
		y = A*( 1.0 - np.exp(-0.2*(q1/(1.0*nPar))**0.5) ) + B*( np.e - np.exp(q2/(1.0*nPar)) ) + 10**-15
	# bi-modal
	if( choice == 3 ):
		q1 = 0
		q2 = 0
		for i in range(nPar):
			q1 += (x[i] - a[4])**2
			q2 += (x[i] + a[4])**2
		# end
		y = a[1]*( (1.0-np.exp(-0.5*(q1/a[2]**2)**(a[3]/2.0))) + (1.0-np.exp(-0.5*(q2/a[2]**2)**(a[3]/2.0))) )
	# many minima, gaussian
	elif( choice == 4 ):
		q = 0
		y = 0
		for i in range(nPar-1):
			q += x[i]**2
			y += a[3]*(1.0-np.cos(a[4]*2*np.pi*x[i]))
		# end
		y += a[1]*(1.0-np.exp(-0.5*q/a[3]**2))
	# end
	
	return y
	
# end

def log_likelihood( x, p, choice ):
	
	sig = x[-1]
	
	error = f(x, p, choice)
	
#	ll = -Ovr*np.log(2*np.pi*sig**2)/2 - (Ovr*(RMSE**2)/(2*sig**2))
#	ll = -(Ovr*(RMSE**2)/(2*sig**2))
#	ll = -(nBin**2*(RMSE**2))
	
	# sinkala's
#	ll = -nBin**2*( np.log(1+2*np.pi*sig**2)/2 + error**2/(2*sig**2) )
	ll = -( np.log(2*np.pi*sig**2)/2 + error**2/(2*sig**2) )
	
	return ll, error

def log_prior( x, xLim ):
	
	inRange = 1
	for i in range(len(x)):
		if( not ( xLim[i,0] <= x[i] <= xLim[i,1] ) ):
			inRange = 0
		# end
	# end
	
	if( inRange ):
		return 0
	else:
		return -np.inf
	# end

# end

def log_posterior( x, p, xLim, choice):
	
	pri = log_prior(x, xLim)
	like, error = log_likelihood( x, p, choice )
	
	if np.isfinite(pri):
		return pri + like, error
	else:
		return -np.inf, error
	# end
	
# end

def adapt1( t, a, b, r ):
	return a - (a - b)*(1.0 - np.exp(-r*t))
# end

def adapt2( t, t1, tw, a, b ):
	if( t < t1 ):
		y = a
	elif( t1 <= t and t < t1 + tw ):
#		y = a + a*( 1.0 - np.cos( np.pi*(t-t1)/tw ) )
		y = 2*a - a*np.cos( np.pi*(t-t1)/tw )
	else:
		y = b
	# end
	return y
# end

def metropolis( start, p, xLim, jump_sigma, n, param_toFit, filename, tid, progress, modRate, modAmp, accMod, toMod, choice):
	
	numP = len(param_toFit)
	cov = np.diag(jump_sigma**2)
	zero = np.zeros(len(jump_sigma))
	n_accept = 0.0
	
	modProb = np.array( [1.0/3.0, 1.0/3.0, 1.0/3.0] )
	
	y_i = np.random.uniform(low=0, high=1, size=n)
	jumps = np.random.multivariate_normal(mean=zero,
	                                      cov=cov, 
	                                      size=n)
	# end
	for i in range(n):
		for j in range(len(start)):
			if( jump_sigma[j] == 0.0 ):
				jumps[i][j] = 0.0
	# end
	
	max_lp = -np.inf
	max_ps = start
	
	chain = np.array([start,])
	cur = chain[-1]
	cur_lp, cur_err = log_posterior( start, p, xLim, choice)
	
	isAcc = 1
	
	wFile = open(filename, 'w')
	wFile.write(" ".join(map(str,start))+" "+str(0)+" "+str(isAcc)+" "+str(cur_lp)+" "+str(cur_err)+"\n")
	
	rejects  = 0
	curInc = np.zeros(numP)
	curDec = np.zeros(numP)
	
	# Random walk
	for step in range(n):
#		print curInc
#		print curDec
		
		if( progress ):
			print "step: ", step
		# end
		
		# Get current position of chain
		cur = chain[-1]
		
		if( rejects == 0 or toMod == 0 ):
			curInc = np.zeros(numP)
			curDec = np.zeros(numP)
			
			cand = cur + jumps[step]
		else:
			modProb[2] = adapt2( step, modRate[2], modRate[3], 1.0/3.0, 1.0 )
			modProb[0] = 0.5*(1-modProb[2])
			modProb[1] = modProb[0]
#			print step, modProb
			
			for i in range(numP-1):
				r = np.random.uniform(0,1)
				mod = 1.0
				
				# thin
				if( r <= modProb[0] ):
					curDec[i] += 1
					mod = adapt1( curDec[i], 1, modAmp[0], modRate[0] )
#					mod = 1.0 - (1.0 - modAmp[0])*(1.0 - np.exp(-modRate[0]*curDec[i]))
					
					jumps[step][i] *= mod
					cand[i] = cur[i] + jumps[step][i]
#					print i, curDec[i], mod
				# wide
				elif( r <= modProb[0] + modProb[1] ):
					curInc[i] += 1
					mod = adapt1( curInc[i], 1, modAmp[1], modRate[1] )
#					mod = 1.0 + (modAmp[1] - 1.0)*(1.0 - np.exp(-modRate[1]*curInc[i]))
					
					jumps[step][i] *= mod
					cand[i] = cur[i] + jumps[step][i]
				# fixed
				else:
					cand[i] = cur[i] + jumps[step][i]
				# end
			# end
			cand[-1] = cur[-1] + jumps[step][-1]
		# end
		
		cand_lp, cand_err = log_posterior(cand, p, xLim, choice)
		acc_prob = np.exp(accMod*(cand_lp - cur_lp))
		
		if( y_i[step] <= acc_prob ):
			rejects = 0
			n_accept += 1
			cur = cand
			cur_lp = cand_lp
			cur_err = cand_err
#			print "log-p:\t" + str(cur_lp)
			chain = np.append(chain, [cur,], axis=0)
			isAcc = 1
			#chain.append(cand)
			if( progress ):
				print "err:\t" + str(cur_err)
				print "      accepted " + str(tid)
			# end
		else:
			rejects += 1
			chain = np.append(chain, [cur,], axis=0)
#			chain = np.append(chain, [cand,], axis=0)
			isAcc = 0
#			print "log-p:\t" + str(cand_lp)
			if( progress ):
				print "err:\t" + str(cand_err)
				print "rejected " + str(tid)
			# end
		# end
		
		wFile.write(" ".join(map(str,cand))+" "+str(step+1)+" "+str(isAcc)+" "+str(cand_lp)+" "+str(cand_err)+"\n")
		
		if( cur_lp < max_lp ):
			max_lp = cur_lp
			max_ps = cur
		# end
	# end
	
	wFile.close()
	
	# Acceptance rate
	acc_rate = (1.0*n_accept/n)
		
	return [chain, acc_rate, max_ps, max_lp, jumps]

# end

def getConv( filename, thresh, nStep, nDim ):
	
	stuff  = np.loadtxt(filename)
	
	isConv = 0.0
	curErr = np.inf
	conv = []
	chainOld    = []
	chainFixed  = []
	chainStrict = []
	for i in range(nStep+1):
		if( stuff[i,-1] < thresh ):
			isConv = 1.0
		# end
		conv.append(isConv)
		chainOld.append(stuff[i,0:nDim])
		if( stuff[i,-3] == 1 ):
			chainFixed.append(stuff[i,0:nDim])
		else:
			chainFixed.append(chainFixed[-1])
		# end
		if( stuff[i,-1] < curErr ):
			curErr = stuff[i,-1]
			chainStrict.append(stuff[i,0:nDim])
		else:
			chainStrict.append(chainStrict[-1])
		# end
	# end
	conv = np.array(conv)
	return conv, chainOld, chainFixed, chainStrict
	
# end

def BinForTargetDist(nBin, nStep, xLim, chains):
	
	binCnt = np.zeros(nBin)
	
	xmin = xLim[0][0]
	xmax = xLim[0][1]
	
	for i in range(nStep):
		x = float(chains[i])
		
		ii = int((x - xmin) / (xmax - xmin) * nBin)
		
		if( ii > 0 and ii < nBin ):
		        binCnt[ii] = binCnt[ii] + 1
		# end
	# end
		
	return binCnt

# end

def BinStuff(nBin, nStep, xLim, chains):
	
	binCnt = np.zeros((nBin,nBin))
	
	xmin = xLim[0][0]
	xmax = xLim[0][1]
	ymin = xLim[1][0]
	ymax = xLim[1][1]
	
	xmin = np.min(chains[:,0])
	xmax = np.max(chains[:,0])
	ymin = np.min(chains[:,1])
	ymax = np.max(chains[:,1])
	
	for i in range(nStep):
		x = float(chains[i,0])
		y = float(chains[i,1])
		
		ii = int((x - xmin) / (xmax - xmin) * nBin)
		jj = int((y - ymin) / (ymax - ymin) * nBin)
		
		if( ii > 0 and ii < nBin and jj > 0 and jj < nBin ):
		        binCnt[jj,ii] = binCnt[jj,ii] + 1
		# end
	# end
		
	return binCnt

# end










main()



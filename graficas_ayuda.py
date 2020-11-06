import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import networkx as nx

def makeLogLogFit(xs,ys):
    xsfit = sm.add_constant(np.log(xs))
    # build model and train
    mod = sm.OLS(exog=xsfit,endog=np.log(ys))
    fit = mod.fit()
    # make list of predicted data
    ysfit = [np.exp(y) for y in fit.predict(xsfit)]
    # plotting results
    return fit, ysfit

def plotLogLogFit(xs,ysfit,fit,textbox=False,r1=4):
    plt.plot(xs,ysfit,label="Power law fit")
    textstr = "y = {0} x^({1}) \n R^2 = {2}".format(round(np.exp(fit.params[0]),r1),round(fit.params[1],r1),round(fit.rsquared,8))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.95,0.95, textstr,transform = plt.gca().transAxes,  fontsize=14,va='top',ha="right", bbox=props)
    
def plotLogLog(xs,ys,title,fit=True,norm=True):
    if norm:
        s = sum(ys)
        ys = [y/s for y in ys]
        plt.ylabel("Normalized Frecuency")
    else:
        plt.ylabel("Frecuency")
    plt.plot(xs,ys,label="Data")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(title)
    plt.grid()
    if fit:
        fit, ysfit = makeLogLogFit(xs,ys)
        # plotting results
        plotLogLogFit(xs,ysfit,fit)

def plotBinnedDegreeDistributions(G,title,bins=20,fit=False,norm=True):
    if type(G) == nx.classes.digraph.DiGraph:
        degs = ["degree","in_degree","out_degree"]
    else:
        degs = ["degree"]
    for deg in degs:
        func = getattr(G,deg)
        degreeSequence = sorted([d for n,d in func()])
        m1 = degreeSequence[0]
        m2 = degreeSequence[-1]
        if m1==0:
            m1=1
        A = np.logspace(np.log10(m1),np.log10(m2),bins)
        plt.figure()
        if norm:
            weights = np.ones_like(degreeSequence) / float(len(degreeSequence))
            B = plt.hist(degreeSequence,bins=A,ec="k",weights=weights)
            plt.ylabel("Normalized frecuency")
        else: 
            B = plt.hist(degreeSequence,bins=A,ec="k")
            plt.ylabel("Frecuency")
        indexes = [i for i in range(len(A)-1) if B[0][i] > 0]
        xs = [(A[i]+A[i+1])/2 for i in indexes]
        ys = [B[0][i] for i in indexes]
        fit, ysfit = makeLogLogFit(xs,ys)
        plotLogLogFit(xs,ysfit,fit)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Degree")
        plt.title("{1} \n {0} binned distribution".format(deg.replace("_"," "),title))
        plt.grid()
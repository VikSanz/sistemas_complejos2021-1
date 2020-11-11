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

def plotLogLogFit(xs,ysfit,fit,ax,textbox=False,r1=4):
    ax.plot(xs,ysfit,label="Power law fit")
    textstr = "y = {0} x^({1}) \n R^2 = {2}".format(round(np.exp(fit.params[0]),r1),round(fit.params[1],r1),round(fit.rsquared,8))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95,0.95, textstr,transform = ax.transAxes,  fontsize=14,va='top',ha="right", bbox=props)
    
def plotLogLog(xs,ys,ax,fit=True,norm=True):
    if norm:
        s = sum(ys)
        ys = [y/s for y in ys]
        ax.set_ylabel("Normalized Frecuency")
    else:
        ax.set_ylabel("Frecuency")
    ax.plot(xs,ys,label="Data")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    if fit:
        fit, ysfit = makeLogLogFit(xs,ys)
        # plotting results
        plotLogLogFit(xs,ysfit,fit,ax)


def plotBinnedDegreeDistributions(G,axs,bins=20,fit=False,norm=True,remove_zeros=True):
    if type(G) == nx.classes.digraph.DiGraph:
        degs = ["degree","in_degree","out_degree"]
    else:
        degs = ["degree"]
    for i,deg in enumerate(degs):
        ax = axs.ravel()[i]
        func = getattr(G,deg)
        degreeSequence = sorted([d for n,d in func()])
        m1 = degreeSequence[0]
        m2 = degreeSequence[-1]
        if m1==0:
            m1=1
        A = np.logspace(np.log10(m1),np.log10(m2),bins)
        if norm:
            weights = np.ones_like(degreeSequence) / float(len(degreeSequence))
            B = ax.hist(degreeSequence,bins=A,ec="k",weights=weights)
            ax.set_ylabel("Normalized frecuency")
        else: 
            B = ax.hist(degreeSequence,bins=A,ec="k")
            ax.set_ylabel("Frecuency")
        indexes = [i for i in range(len(A)-1) if B[0][i] > 0]
        xs = [(A[i]+A[i+1])/2 for i in indexes]
        ys = [B[0][i] for i in indexes]
        fit, ysfit = makeLogLogFit(xs,ys)
        plotLogLogFit(xs,ysfit,fit,ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree")
        ax.set_title("{0} \n binned distribution".format(deg.replace("_"," ")))
        ax.grid()

def degree(G):
    return dict(G.degree())

def plotStatistics(G,funcs=[degree,nx.clustering,nx.closeness_centrality,nx.betweenness_centrality,nx.eigenvector_centrality_numpy,nx.pagerank_numpy],axs=None,vals=None):
    if axs is None:
        rows = int(np.ceil(len(funcs)/2))
        fig,axs = plt.subplots(nrows=rows,ncols=2,figsize=(12,12))
    for j,ax in enumerate(axs.ravel()):
        print("calculating {0}".format(funcs[j].__name__))
        if vals is None:
            ys = funcs[j](G).values()
        else:
            ys = vals[j].values()
        ax.hist(ys,bins=40,alpha=0.6,label=G.name,ec="k")
        ax.grid()
        ax.set_ylabel(funcs[j].__name__.replace("numpy","").replace("_","\n"),fontsize=14)
        #plt.xscale("log")
        #plt.yscale("log")
        if j == 1:
            ax.legend(loc="upper right",fontsize=18)

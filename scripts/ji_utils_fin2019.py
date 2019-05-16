"""
================================================================
     Copyright (c) 2018, Yale University
                     All Rights Reserved
================================================================

 NAME : ji_utils_finance.py
 @DATE     Created: 11/23/18 (8:26 AM)
	       Modifications:
	       - 11/23/18: my personal finance library
	                   - Volatility

 @AUTHOR          : Jaime Shinsuke Ide
                    jaime.ide@yale.edu
===============================================================
"""
import logging
import datetime
import pandas as pd
import numpy as np
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import time
import math

import CostCurve # For DYCE result parsing

#import ji_utils_plots as ji
import sys
import os
#sys.path.append(os.path.abspath("/home/jide/projects"))
from ji_utils_plots2019 import *

from scipy import stats

"""
Vectorize to compute:

tic = time.time()
# obs: vectorize more efficient than using df.apply()
df2go['empCost'] = np.vectorize(jifin.compute_empCost)(df2go['ExPriceForBin'], df2go['priceBM'],
                                                         df2go['sign'])
print('... Empirical Cost computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))

"""


def add_tokenID(data, binsize=78):
    '''Function to bin/bucket by token'''
    print('(*) Note: it assumes that all clusters have equal #bins of %d.'%binsize)
    data['tokenID'] = [int(i)//binsize+1 for i in range(data.shape[0])]

    return (data)

def binningModule(data, numBin, on_colname, threshold, method, bin_colname):
    '''Function to bin/bucket the given column on specified method'''
    data = data.copy(deep=True)
    data_truncate = data[abs(data[on_colname]) < threshold]

    if data_truncate.shape[0] > 0:
        data.loc[data[on_colname] < min(data_truncate[on_colname]), on_colname] = min(data_truncate[on_colname])
        data.loc[data[on_colname] > max(data_truncate[on_colname]), on_colname] = max(data_truncate[on_colname])

    data.sort_values(by=[on_colname], inplace=True)
    data = data.reset_index(drop=True)

    if method.lower() == 'size':
        binsize = float(math.ceil(data.shape[0] / numBin - 0.5))
        data[bin_colname] = np.floor(np.divide(data.index, binsize)) + 1
        data.loc[data[bin_colname] > numBin, bin_colname] = numBin

    if (method.lower() == 'length'):
        binlength = (max(data[on_colname]) - min(data[on_colname])) / numBin
        data[bin_colname] = np.floor(np.divide(np.subtract(data[on_colname], min(data[on_colname])), binlength)) + 1
        data.loc[data[on_colname] == max(data[on_colname]), bin_colname] = numBin

    return (data)


def aggregationModule(data, on_colname, by_colname, method, wt_colname=None):
    '''Function to aggregate given column by another column on equal-weighted average or dollar-weighted average with weight specified'''
    if isinstance(on_colname, list) and isinstance(by_colname, list):
        if method.lower() == 'equal':
            print('- Equally weighted average within bin...')
            tmp = data.groupby(by_colname)[on_colname].mean().reset_index()
        elif method.lower() == 'dollar':
            print('- Dollar weighted average within bin...')
            if wt_colname is None:
                raise ValueError('Weight column must be specified if applying dollar-weighting mehtod')
            else:
                tmp = data.groupby(by_colname).apply(wavg, on_colname, wt_colname).reset_index()
                tmp.columns = by_colname + on_colname
        else:
            raise ValueError('Invalid aggregation method')

        # compute Std  divided by sqrt(N)
        sem = data.groupby(by_colname)[on_colname].std() / (data.groupby(by_colname)[on_colname].count().apply(np.sqrt))
        sem = sem.reset_index()

        return (tmp, sem)
    else:
        raise ValueError('Invalid group-by column or value column; must pass as list')


def helper_agg(df2gox, sizeVar, targetVar, quantiles, savepath, savelabel):
    ## 1) Split size by quantile
    SizeThr = str(df2gox[sizeVar].min() - 10)
    for q in quantiles:
        qThr = round(df2gox[sizeVar].quantile(q), 4)
        SizeThr = SizeThr + ":" + str(qThr)
    SizeThr = SizeThr + ":" + str(round(df2gox[sizeVar].max() + 10))
    SizeThr = np.fromstring(SizeThr, dtype=float, sep=":")
    ## 2) Aggregate by size
    ColAgg = [targetVar] + [sizeVar]
    dfagg = CostCurve.CostCurve(df2gox.copy(), None, SizeThr, ColAgg, sizeVar, 'surVola', sizeVar, 0, 0.005,
                                PrintNA=False)
    dfagg.to_csv(os.path.join(savepath, 'table_%s.csv' % (savelabel)))
    print('- Filtering by %s (retrieved %2.2f%%)...' % ('surVola', 100 * dfagg.Count.sum() / df2gox.shape[0]))

    return dfagg

def plot_GoF_fullVersion(df2plot, mylabel, savepath,tablelabel, binstart, binend, sizefloor, sizecap, pdfobj=None, quantiles = [0.2, 0.4, 0.6, 0.8, 0.9], # size quantiles for the aggregation
            binsize=78, binmethod='size', aggrmethod='equal', wt_colname=None):
    '''Plot method to visualize GoF on certain filtrations.'''
    # Default settings
    fsize = 20
    zscore = 1.96
    irangeL = [0, 1, 2] # index of percentiles to plot
    irangeU = [-3, -2, -1]
    fsizetext = 14 # font size of the text

    matplotlib.rcParams.update({'font.size': fsize})

    print("======= GoF:", mylabel)
    print("- Time of the day: bin%d to bin%d"%(binstart,binend))

    # ================= Filtration start ==========================
    truncdata = df2plot.copy()

    truncdata = truncdata[(truncdata.bin >= binstart) & (truncdata.bin <= binend)]
    truncdata = truncdata[(abs(truncdata.sizeShare) >= sizefloor) & (abs(truncdata.sizeShare) <= sizecap)]

    print("--- Filtration ---")
    print("Filtration ratio: ", round(100 - 100 * truncdata.shape[0] / df2plot.shape[0], 1), '%')
    print("A total clusters left:", truncdata.shape[0])

    # imask = df2plot.tradeDate.between(20170101,20170701)
    imask = (truncdata.surVola != 0)
    df2gox = truncdata[imask].copy()
    print('- Excluded %d bins with zero surVola...' % np.sum(df2plot.surVola == 0))
    # df2gox = df2plot.copy()

    # ================= Filtration end ============================

    # ================= Compute logs ==========================
    df2gox['logSurVola'] = df2gox['surVola'].apply(np.log2)
    df2gox['logPred5min'] = df2gox['pred5min'].apply(np.log2)
    df2gox['logPred30min'] = df2gox['pred30min'].apply(np.log2)

    # ================= Aggregate ==========================
    ## 5min
    savelabel = tablelabel + '_vola_aggr5min'
    sizeVar = 'pred5min'
    targetVar = 'surVola'
    dfagg5min = helper_agg(df2gox.copy(), sizeVar, targetVar, quantiles, savepath, savelabel)
    ## 30min
    savelabel = tablelabel + '_vola_aggr30min'
    sizeVar = 'pred30min'
    targetVar = 'surVola'
    dfagg30min = helper_agg(df2gox.copy(), sizeVar, targetVar, quantiles, savepath, savelabel)
    ## log 5min
    savelabel = tablelabel + '_vola_aggr5min'
    sizeVar = 'logPred5min'
    targetVar = 'logSurVola'
    dfaggLog5min = helper_agg(df2gox.copy(), sizeVar, targetVar, quantiles, savepath, savelabel)
    ## log 30min
    savelabel = tablelabel + '_vola_log_aggr30min'
    sizeVar = 'logPred30min'
    targetVar = 'logSurVola'
    dfaggLog30min = helper_agg(df2gox.copy(), sizeVar, targetVar, quantiles, savepath, savelabel)

    # ================= Plot ==========================
    fig = plt.figure(figsize=[20, 10])

    #################### Log scale
    plt.subplot(1, 2, 1)
    realVar = 'logSurVola'
    xymax = 0  # initial value
    xymin = 0  # initial value

    ## 5min
    predVar = 'logPred5min'
    dfx = dfaggLog5min.copy()
    y1 = dfx[realVar]
    x = dfx[predVar]
    xSEM = dfx['StdErr']
    xymax = np.round(np.max([np.abs(x.max()), np.abs(y1.max()), xymax]))  # keep max limit
    xymin = np.min([x.min(), y1.min(), xymin])  # keep max limit
    plt.scatter(x, y1, label='5min', s=70, alpha=0.75, edgecolors='k', facecolors='blue')  # , plt.scatter(x,y1)
    # plt.plot(x, y1, c='b',label='') #, plt.scatter(x,y1)
    ySEM = zscore * xSEM  # Use the predicted SEM for the y-axis...
    plt.fill_between(x, y1 - ySEM, y1 + ySEM, alpha=0.5)
    # Add percentile text
    yyU1 = y1.iloc[irangeU].values
    xxU1 = x.iloc[irangeU].values
    yyL1 = y1.iloc[irangeL].values
    xxL1 = x.iloc[irangeL].values
    # for cx,cy,s in zip(xx,yy,np.array(quantiles)[irange]):
    #     plt.text(cx-0.3,cy+0.05,'%1.3f'%s,fontsize=fsizetext)

    ## 30min
    predVar = 'logPred30min'
    dfx = dfaggLog30min.copy()
    y1 = dfx[realVar]
    x = dfx[predVar]
    xSEM = dfx['StdErr']
    plt.scatter(x, y1, label='30min', s=70, alpha=0.75, edgecolors='r', facecolors='orange')  # , plt.scatter(x,y1)
    # plt.plot(x, y1, c='b',label='') #, plt.scatter(x,y1)
    ySEM = zscore * xSEM  # Use the predicted SEM for the y-axis...
    plt.fill_between(x, y1 - ySEM, y1 + ySEM, alpha=0.5)
    # Add percentile text
    yyU2 = y1.iloc[irangeU].values
    xxU2 = x.iloc[irangeU].values
    yyL2 = y1.iloc[irangeL].values
    xxL2 = x.iloc[irangeL].values

    if xxL1[0]<xxL2[0]:
        for cx, cy, s in zip(xxL1, yyL1, 100*np.array(quantiles)[irangeL]):
            plt.text(cx - 0.3, cy + 0.05, '%2.1f%%' % s, fontsize=fsizetext)
        for cx, cy, s in zip(xxL2, yyL2, 100*np.array(quantiles)[irangeL]):
            plt.text(cx + 0.1, cy - 0.05, '%2.1f%%' % s, fontsize=fsizetext)
    else:
        for cx, cy, s in zip(xxL2, yyL2, 100*np.array(quantiles)[irangeL]):
            plt.text(cx - 0.3, cy + 0.05, '%2.1f%%' % s, fontsize=fsizetext)
        for cx, cy, s in zip(xxL1, yyL1, 100*np.array(quantiles)[irangeL]):
            plt.text(cx + 0.1, cy - 0.05, '%2.1f%%' % s, fontsize=fsizetext)

    if xxU1[-1] < xxU2[-1]:
        for cx, cy, s in zip(xxU1, yyU1, 100*np.array(quantiles)[irangeU]):
            plt.text(cx - 0.3, cy + 0.05, '%2.1f%%' % s, fontsize=fsizetext)
        for cx, cy, s in zip(xxU2, yyU2, 100*np.array(quantiles)[irangeU]):
            plt.text(cx + 0.1, cy - 0.05, '%2.1f%%' % s, fontsize=fsizetext)
    else:
        for cx, cy, s in zip(xxU2, yyU2, 100*np.array(quantiles)[irangeU]):
            plt.text(cx - 0.3, cy + 0.05, '%2.1f%%' % s, fontsize=fsizetext)
        for cx, cy, s in zip(xxU1, yyU1, 100*np.array(quantiles)[irangeU]):
            plt.text(cx + 0.1, cy - 0.05, '%2.1f%%' % s, fontsize=fsizetext)


    ## Others
    xymax = np.round(np.max([np.abs(x.max()), np.abs(y1.max()), xymax]))  # keep max limit
    xymin = np.floor(np.min([x.min(), y1.min(), xymin]))  # keep max limit
    plt.plot([xymin, xymax], [xymin, xymax], label="45 degree", c='gray', alpha=0.5, linewidth=3)
    # plt.title('Costx%s(%s)'%(sizeLabel,mylabel), fontsize=myfontsize)
    plt.ylabel('' + realVar, size=fsize);
    plt.xlabel('' + predVar, size=fsize);
    # set axes range
    plt.xlim(xymin, xymax)
    plt.ylim(xymin, xymax)
    plt.legend()
    plt.title('Predicted v.s. Realized: ' + predVar + '\n%s' % mylabel + ' \nBin range: ' + str(binstart) + '-' + str(
        binend) + '; sizeShare: ' + str(sizefloor) + '-' + str(sizecap), size=fsize)

    ####################### Raw scale
    plt.subplot(1, 2, 2)
    realVar = 'surVola'
    xylim = 0  # initial value

    ## 5min
    predVar = 'pred5min'
    dfx = dfagg5min.copy()
    y1 = dfx[realVar]
    x = dfx[predVar]
    xSEM = dfx['StdErr']
    xylim = np.round(np.max([x.max(), y1.max(), xylim]))  # keep max limit
    plt.scatter(x, y1, label='5min', s=70, alpha=0.75, edgecolors='k', facecolors='blue')  # , plt.scatter(x,y1)
    # plt.plot(x, y1, c='b',label='') #, plt.scatter(x,y1)
    ySEM = zscore * xSEM  # Use the predicted SEM for the y-axis...
    plt.fill_between(x, y1 - ySEM, y1 + ySEM, alpha=0.5)
    # Add percentile text
    yyU1 = y1.iloc[irangeU].values
    xxU1 = x.iloc[irangeU].values
    yyL1 = y1.iloc[irangeL].values
    xxL1 = x.iloc[irangeL].values

    # for cx, cy, s in zip(xx, yy, np.array(quantiles)[irange]):
    #     plt.text(cx - 0.3, cy + 0.05, '%1.3f' % s, fontsize=fsizetext)

    ## 30min
    predVar = 'pred30min'
    dfx = dfagg30min.copy()
    y1 = dfx[realVar]
    x = dfx[predVar]
    xSEM = dfx['StdErr']
    plt.scatter(x, y1, label='30min', s=70, alpha=0.75, edgecolors='r', facecolors='orange')  # , plt.scatter(x,y1)
    # plt.plot(x, y1, c='b',label='') #, plt.scatter(x,y1)
    ySEM = zscore * xSEM  # Use the predicted SEM for the y-axis...
    plt.fill_between(x, y1 - ySEM, y1 + ySEM, alpha=0.5)

    # Add percentile text
    yyU2 = y1.iloc[irangeU].values
    xxU2 = x.iloc[irangeU].values
    yyL2 = y1.iloc[irangeL].values
    xxL2 = x.iloc[irangeL].values

    if xxL1[0]<xxL2[0]:
        for cx, cy, s in zip(xxL1, yyL1, 100*np.array(quantiles)[irangeL]):
            plt.text(cx - 0.3, cy + 0.05, '%2.1f%%' % s, fontsize=fsizetext)
        for cx, cy, s in zip(xxL2, yyL2, 100*np.array(quantiles)[irangeL]):
            plt.text(cx + 0.1, cy - 0.05, '%2.1f%%' % s, fontsize=fsizetext)
    else:
        for cx, cy, s in zip(xxL2, yyL2, 100*np.array(quantiles)[irangeL]):
            plt.text(cx - 0.3, cy + 0.05, '%2.1f%%' % s, fontsize=fsizetext)
        for cx, cy, s in zip(xxL1, yyL1, 100*np.array(quantiles)[irangeL]):
            plt.text(cx + 0.1, cy - 0.05, '%2.1f%%' % s, fontsize=fsizetext)

    if xxU1[-1] < xxU2[-1]:
        for cx, cy, s in zip(xxU1, yyU1, 100*np.array(quantiles)[irangeU]):
            plt.text(cx - 0.3, cy + 0.05, '%2.1f%%' % s, fontsize=fsizetext)
        for cx, cy, s in zip(xxU2, yyU2, 100*np.array(quantiles)[irangeU]):
            plt.text(cx + 0.1, cy - 0.05, '%2.1f%%' % s, fontsize=fsizetext)
    else:
        for cx, cy, s in zip(xxU2, yyU2, 100*np.array(quantiles)[irangeU]):
            plt.text(cx - 0.3, cy + 0.05, '%2.1f%%' % s, fontsize=fsizetext)
        for cx, cy, s in zip(xxU1, yyU1, 100*np.array(quantiles)[irangeU]):
            plt.text(cx + 0.1, cy - 0.05, '%2.1f%%' % s, fontsize=fsizetext)

    ## Others
    xylim = np.round(np.max([x.max(), y1.max(), xylim]))  # keep max limit
    plt.plot([-xylim, xylim], [-xylim, xylim], label="45 degree", c='gray', alpha=0.5, linewidth=3)
    # plt.title('Costx%s(%s)'%(sizeLabel,mylabel), fontsize=myfontsize)
    plt.ylabel('' + realVar, size=fsize);
    plt.xlabel('' + predVar, size=fsize);
    # set axes range
    plt.xlim(0, xylim)
    plt.ylim(0, xylim)
    plt.legend()
    plt.title('Predicted v.s. Realized: ' + predVar + '\n%s' % mylabel + ' \nBin range: ' + str(binstart) + '-' + str(
        binend) + '; sizeShare: ' + str(sizefloor) + '-' + str(sizecap), size=fsize)

    plt.tight_layout()
    #plt.show()

    if pdfobj is not None:
        try:
            pdfobj.savefig(fig)
        except:
            print('(*) Warning: problem saving the PDF file...')


def plot_GoF(df2plot, mylabel, binstart, binend, sizefloor, sizecap, realVar, realFloor, realCap, pdfobj=None,
            binsize=78, binmethod='size', aggrmethod='equal', wt_colname=None):
    '''Plot method to visualize GoF on certain filtrations.'''
    # Assign prefix for variables
    fsize = 20
    matplotlib.rcParams.update({'font.size': fsize})

    print("======= GoF:", mylabel)
    print("- Time of the day: bin%d to bin%d"%(binstart,binend))

    # ================= Filtration start ==========================
    truncdata = df2plot.copy()

    truncdata = truncdata[(truncdata.bin >= binstart) & (truncdata.bin <= binend)]
    truncdata = truncdata[(abs(truncdata.sizeShare) >= sizefloor) & (abs(truncdata.sizeShare) <= sizecap)]
    try:
        truncdata = truncdata[ (truncdata[realVar] >= realFloor) & (truncdata[realVar] <= realCap) ]
    except:
        raise ValueError('Invalid statistic name')

    print("--- Filtration ---")
    print("Filtration ratio: ", round(100 - 100 * truncdata.shape[0] / df2plot.shape[0], 1), '%')
    print("A total clusters left:", truncdata.shape[0])
    # ================= Filtration end ============================
    binningType = 'byToken'
    if binningType == 'byToken':
        # Add tokenID column
        truncdata = jifin.add_tokenID(truncdata,binsize)
        groupColName = 'tokenID'
    else:
        binnum = 20
        truncdata = binningModule(truncdata,numBin=binnum,on_colname=realVar,threshold=10,method='size', bin_colname='mybin')
        groupColName = 'mybin'

    predVar = 'pred5min'
    aggrdata, dfsem = aggregationModule(truncdata, [realVar,predVar], [groupColName],aggrmethod, wt_colname)

    fig = plt.figure(figsize=[20, 10])
    plt.subplot(1, 2, 1)
    dolog = True
    if dolog:
        plt.plot([-2,2],[-2,2], label="45 degree", c='gray',alpha=0.5, linewidth = 5)
        plt.scatter(np.log2(truncdata[predVar]),np.log2(truncdata[realVar]), label="allBins",c='blue',s=3,alpha=0.2)
        plt.scatter(np.log2(aggrdata[predVar]),np.log2(aggrdata[realVar]), label="mean(daily)", s=50, alpha=0.75,edgecolors = 'k',facecolors='blue')
        plt.ylabel('log2:' + realVar.lower().capitalize(), size=fsize);
        plt.xlabel('log2:' + predVar.lower().capitalize(), size=fsize);
        # set axes range
        plt.xlim(-1, 2)
        plt.ylim(-1, 2)
        plt.legend()

    else:
        #plt.plot([-2, 2], [-2, 2], label="45 degree", c='gray', alpha=0.5, linewidth=5)
        plt.scatter(truncdata[predVar], truncdata[realVar], label="allBins", c='blue', s=3, alpha=0.2)
        plt.scatter(aggrdata[predVar], aggrdata[realVar], label="mean(daily)", s=50, alpha=0.75,
                    edgecolors='k', facecolors='blue')
        plt.ylabel(realVar.lower().capitalize(), size=fsize);
        plt.xlabel(predVar.lower().capitalize(), size=fsize);

    plt.title('Realized v.s. Predicted ' + predVar.lower().capitalize() + '\n%s'%mylabel + ' \nBin range: ' + str(binstart) + '-' + str(
                binend) + '; sizeShare: ' + str(sizefloor) + '-' + str(sizecap), size = fsize)

    predVar = 'pred30min'
    aggrdata2, dfsem2 = aggregationModule(truncdata, [realVar, predVar], [groupColName], aggrmethod, wt_colname)
    plt.subplot(1, 2, 2)
    dolog = True
    if dolog:
        plt.plot([-2, 2], [-2, 2], label="45 degree", c='gray', alpha=0.5, linewidth = 5)
        plt.scatter(np.log2(truncdata[predVar]),np.log2(truncdata[realVar]), label="allBins", c='orange', s=3,
                    alpha=0.4)
        plt.scatter(np.log2(aggrdata2[predVar]),np.log2(aggrdata2[realVar]), label="mean(daily)", s=50, alpha=1,
                    edgecolors='r', facecolors='orange')
        plt.ylabel('log2:' + realVar.lower().capitalize(), size = fsize);
        plt.xlabel('log2:' + predVar.lower().capitalize(), size = fsize);
        # set axes range
        plt.xlim(-1, 2)
        plt.ylim(-1, 2)
        plt.legend()

    else:
        #plt.plot([-2, 2], [-2, 2], label="45 degree", c='gray', alpha=0.5, linewidth=5)
        plt.scatter(truncdata[predVar],truncdata[realVar], label="allBins", c='orange', s=3, alpha=0.4)
        plt.scatter(aggrdata2[predVar],aggrdata2[realVar], label="mean(daily)", s=50, alpha=1, edgecolors='r', facecolors='orange')
        plt.ylabel(realVar.lower().capitalize(), size = fsize);
        plt.xlabel(predVar.lower().capitalize(), size = fsize);

    plt.title('Realized v.s. Predicted ' + predVar.lower().capitalize() + '\n%s'%mylabel  + ' \n Bin range: ' + str(binstart) + '-' + str(
            binend) + '; sizeShare: ' + str(sizefloor) + '-' + str(sizecap), size = fsize)

    plt.tight_layout()
    #plt.show()

    if pdfobj is not None:
        pdfobj.savefig(fig)

def p5_helper_load_compute_df2plot30min_ABD2018(df2plotpath,allrange,allHLG,myStrategies = ['SCHEDULED','DARK','IS','OPPORTUNISTIC']):
    """Wrapper to
    - Load df2plot and concatenate
    """

    ## SET DATE RANGES
    #myweeks = [[1,7],[8,15],[16,23],[24,31]]
    iweeks = [1,2,3,4]
    #

    ## SET PATHS
    #dpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data' # Where df2go is loaded from
    #df2plotpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data_df2plot'  # Where df2plot will be saved

    ######## 1) Get df2go
    df2plot = pd.DataFrame()
    print('- Loading from:',df2plotpath)
    for t0t1 in allrange:
        for HLG in allHLG:
            t0 = t0t1[0]
            t1 = t0t1[1]
            slabel = '%d_%d_HLG%d' % (t0, t1, HLG)
            for strategy in myStrategies:
                for week in iweeks:
                    nn = 'df2plot_ABD_HLGlarger_%s_%s_week%d.csv.gz' %(slabel,strategy,week)
                    fname = os.path.join(df2plotpath, nn)
                    print('- %s... ' %(nn),end='')
                    dfx = pd.read_csv(fname) #
                    print('%d trades'%dfx.shape[0])

                    # Convert to 30min bins
                    dfx = jifin.df2go_aggregate(dfx)
                    print('- Total after conversion to 30min: %d trades' % dfx.shape[0])
                    # Compute
                    dfx = jifin.ji_wrapper_compute_CostReturn_etc_30min(dfx)
                    df2plot = df2plot.append(dfx)

    #ji_p5_check_df2plot(df2plot)
    return df2plot

def ji_p5_correct_duration(df2go,cond):
    iupdate = df2go.index[cond]
    nfill = df2go.FillNbrForBin.values
    cbin = df2go.bin.values
    for i in iupdate:
        # print('i:',i)
        # Search back until start of the day (Todo: optimize)
        j = i - 1  # one back
        while True:
            #print('j:',j)
            if nfill[j] != 0:
                # Last trade found
                # duration
                df2go.loc[i,'gapDuration'] = i - j - 1
                break
            else:
                if cbin[j] == 1:
                    # very1stTrade[i] = 1
                    print('** Warning!! ** Reach Bin1 without active bin. Check...')
                    break  # reached the 2nd bin, no more update
                # next iteration
                j += -1
    print('- Correction done: firstTrade bins with duration 0.')
    return df2go

def ji_wrapper_compute_CostReturn_etc_30min(df2go):
    """Wrapper to create and compute all features, including EmpCost and hist volatility etc.
    It will return a df2go."""
    tic0 = time.time()

    # 0) Add sign based on token
    df2go['sign'] = df2go.token.apply(lambda s: 1 if 'Buy' in s else -1)

    # 1) Add historical Volatility (weights)
    df2go['histVola30min'] = df2go.histVolaW*df2go.histVola
    # 2) Add historical Volume
    df2go['histVolu30min'] = df2go.mddv*df2go.histVoluW

    # 3) Estimate realized volatility
    tic = time.time()
    df2go['realVola'] = np.vectorize(compute_vola_maxmin_silent)(df2go['HighPrice'], df2go['LowPrice'],
                                                              df2go['firstMQ'], df2go['lastMQ'])
    if df2go['realVola'].isnull().values.any():
        print('(*) Note: Unchanged price set to 0.01 and division by zero to NaN...')
    print('... MaxMin Volatility computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))

    # 4) Compute surprise of volatility
    tic = time.time()
    df2go['surVola'] = np.vectorize(jifin.my_divide)(df2go['realVola'], df2go['histVola30min'])
    print('... Volatility surprise computed in %2.4f seconds (using vectorize).' % (time.time() - tic))

    ## DEBUGGING
    print('- Number of zero realVola: %2.2f' % (np.sum(df2go.realVola == 0) / df2go.shape[0]))
    if (np.sum(df2go.realVola == 0) / df2go.shape[0])>0.9:
        df2go.head(2000).to_csv('0_df2go_PROBLEM.csv')
        error('ERROR!')

    # # 5) Add the benchmark price, i.e. the previous bin midquote MQ(j-1)
    # df2go = add_priceBM_in_df2go_fast(df2go)

    # 6) Compute the Empirical Cost
    # # 6.1) Handle missing bmPrice (rare but happens when MQ price is missing for first bin of the ...)
    # df2go = fill_missing_priceBM(df2go)
    # # 6.1.1) Fill null after all...
    # if np.sum(df2go.priceBM.isnull()) > 0:
    #     print('- Filling %d null priceBM with zeros...' % (np.sum(df2go.priceBM.isnull())))
    #     df2go.loc[df2go.priceBM.isnull(), 'priceBM'] = 0
    # 6.2) Compute
    tic = time.time()
    # obs: vectorize more efficient than using df.apply()
    df2go['empCost'] = np.vectorize(jifin.compute_empCost_silent)(df2go['ExPriceForBin'], df2go['firstMQ'],
                                                         df2go['sign'])
    nnull = np.sum(df2go.empCost.isnull())
    if nnull > 0:
        print('- Number of empCost filled with NaN: %d...' % nnull)
    print('... Empirical Cost computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))
    #df2go.to_csv('df2go_42cols_Step6.csv')

    # # 6.3) Compute F ang G components
    # tic = time.time()
    # # obs: vectorize more efficient than using df.apply()
    # try:
    #     df2go['Fcost'] = np.vectorize(jifin.compute_F)(df2go['midQuote'], df2go['priceBM'], df2go['vwap'],
    #                                                            df2go['sign'])
    #     df2go['Gcost'] = np.vectorize(jifin.compute_G)(df2go['midQuote'], df2go['priceBM'], df2go['vwap'],
    #                                                    df2go['sign'])
    #     print('... F and G costs computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))
    # except:
    #     print('- PROBLEM: computing F/Gcost. Check 0_df2go_step7.csv...')
    #     df2go.to_csv('0_df2go_step7.csv.gz',compression='gzip')
    #     error('- ERROR ...')

    # 7) Compute participation rate
    tic = time.time()
    df2go['prate'] = np.vectorize(jifin.compute_prate_silent)(df2go['ExShareForBin'], df2go['shareVolume'])
    # Check
    nn = np.sum(df2go.prate == 1)
    if nn > 0:
        print('- Number of prate set to 1 when (exShare > mktVol): %d...' % nn)
    print('... Computed participation rate in %2.4f seconds (using vectorizer).' % (time.time() - tic))

    # # 8) Compute Style
    # tic = time.time()
    # df2go['style'] = np.vectorize(jifin.compute_style)(df2go['ExPriceForBin'], df2go['vwap']
    #                                                  , df2go['priceBM'], df2go['prate'], df2go['sign'])
    # print('... Style computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))
    #
    # # 9) Compute Normalized Style
    # tic = time.time()
    # df2go['normStyle'] = np.vectorize(jifin.compute_style_normalized)(df2go['ExPriceForBin'], df2go['vwap']
    #                                                                   , df2go['HighPrice'], df2go['LowPrice'],
    #                                                                   df2go['prate'], df2go['sign'])
    # print('... Style (normalized) computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))
    #
    # # # 10) Add quadratic volatility from the 15sec MQ
    # # tic = time.time()
    # # df2go = ji_helper_add_quadVola_stdVola_15Sec_in_df2go(df2go, df2go_before)
    # # print('... Quad and Std volatility computed in %2.4f seconds (takes time to load...).' % (time.time() - tic))
    # #
    # # # 11) Compute and add the cumulative and last 30 min volatilities (HL and quadVola)
    # # tic = time.time()
    # # df2go = add_last30_cum_vola(df2go)
    # # print('... Cumulative and last30min volatilities (HL,quadVola) computed in %2.4f seconds.' % (time.time() - tic))
    # #
    # # # 12) Add cumulative Return from the 15sec MQ
    # # tic = time.time()
    # # df2go = ji_helper_add_cumReturn15sec_in_df2go(df2go, df2go_before)
    # # print('... Cumulative return computed in %2.4f seconds (takes time to load...).' % (time.time() - tic))

    # 8) Add Return
    tic = time.time()
    df2go['return'] = np.vectorize(compute_return5min_silent)(df2go['lastMQ'], df2go['firstMQ'])
    df2go['signedReturn'] = np.vectorize(jifin.compute_signedReturn5min_silent)(df2go['lastMQ'], df2go['firstMQ'], df2go['sign'])
    # Checking
    nn = np.sum(df2go['return'].isnull())
    if nn > 0:
        print('(*) Note: Number of missing bmPrice where Return was set to None: %d...' % nn)
    nn = np.sum(df2go['signedReturn'].isnull())
    if nn > 0:
        print('(*) Note: Number of missing bmPrice where signedReturn was set to None: %d...' % nn)
    print('... Return and signedReturn computed in %2.4f seconds (takes time to load...).' % (time.time() - tic))

    # 9) Add First Trade indicator
    execP = df2go['ExPriceForBin'].values
    firstTrade = []
    firstTrade.append(0)  # initialize first bin
    for i in range(len(execP) - 1):
        if (execP[i] == 0) & (execP[i + 1] != 0):
            # print('execP[%d]=%2.4f, val=%2.4f'%(i,execP[i],execP[i+1]))
            firstTrade.append(1)
        else:
            firstTrade.append(0)
    df2go['firstTrade'] = np.array(firstTrade)
    # Correct the first bin: if there is trade, it is the first trade of the day
    df2go.loc[(df2go.bin == 1) & (df2go.ExPriceForBin != 0), 'firstTrade'] = 1
    print('- First trade indicator added.')

    # 10) Add additional feats
    df2go = ji_get_adjusted_milanBM_historicals30(df2go)

    # 10.5) Correct gap duration
    # - a) Fix duration (when there is trade in Bin1): (firstTrade==1) & (very1stTrade==0) & (gapDuration==0)
    cond = (df2go.gapDuration == 0) & (df2go.firstTrade == 1) & (df2go.very1stTrade == 0)
    if np.sum(cond) > 0:
        print('** Fixing! ** 1stTrade after gap with duration ZERO: %d cases...' % np.sum(cond))
        df2go = ji_p5_correct_duration(df2go, cond)

    # 11) Adjustment
    print('Use the adjusted adjBM and historicals and compute:')
    print('- adjNormSignedReturn')
    print('- adjSurVola')
    print('- adjSurVolume')
    df2go['adjSignedReturn'] = np.vectorize(compute_signedReturn5min_silent)(df2go['lastMQ'], df2go['adjBM'],df2go['sign'])
    # Checking
    nn = np.sum(df2go['adjSignedReturn'].isnull())
    if nn > 0:
        print('(*) Note: Number of missing bmPrice where adjSignedReturn was set to None: %d...' % nn)

    # Normalize
    df2go['adjSurVola'] = np.vectorize(jifin.my_divide)(df2go['adjRealVola'], df2go['adjHistVola30min'])
#    df2go['adjSurVolume'] = np.vectorize(jifin.my_divide)(df2go['dollarVolume'], df2go['adjHistVolu5min'])
    df2go['normAdjSignedReturn'] = np.vectorize(jifin.my_divide)(df2go['adjSignedReturn'], df2go['adjHistVola30min']) / 10000 # remove bps

    # 12) Normalize
    # Add regular normalized EmpCost and Return normalized by histVola
    df2go['normSignedReturn'] = np.vectorize(jifin.my_divide)(df2go['signedReturn'], df2go['histVola30min']) / 10000 # remove bps
    # Add EmpCost normalized by histVola as well
    df2go['normEmpCost'] = np.vectorize(jifin.my_divide)(df2go['empCost'], df2go['histVola30min']) / 10000  # remove bps

## DEBUGGING
    print('- Number of zero realVola: %2.2f' % (np.sum(df2go.realVola == 0) / df2go.shape[0]))
    if (np.sum(df2go.realVola == 0) / df2go.shape[0])>0.9:
        df2go.head(2000).to_csv('0_df2go_PROBLEM.csv')
        error('ERROR!')


    print('\n**(TOTAL) df2go created and computed in %2.4f seconds.' % (time.time() - tic0))
    print('******************************************************\n')

    return df2go

def ji_get_adjusted_milanBM_historicals30(df2go): # updated version 2019/04/15
    # Add the milanBM - Working (check 0_df2go_check_milanBM_head2000.xlsx)
    # Checking: 0_df2go_check_milanBM_adjusted_head2000.xlsx
    # Historicals: volatility and volume: sum up weights from the last active bin

    print('- Use the last traded MQ, not the previous one. Sum-up historical vola/volu weights')
    newBM = df2go.firstMQ.copy().values
    newHistVola30min = df2go.histVola30min.copy().values
    newHistVolu30min = df2go.histVolu30min.copy().values
    newRealVola = df2go.realVola.copy().values
    durationFromLast = np.zeros(df2go.shape[0])
    very1stTrade = np.zeros(df2go.shape[0])

    # auxiliary
    bin = df2go.bin.values
    MQ = df2go.firstMQ.values
    maxP = df2go.HighPrice.values
    minP = df2go.LowPrice.values

    shareTraded = df2go.ExShareForBin.values
    close = df2go.close.values # USE THIS AS THIS INFO IS INCLUDED...
    #print('- Using open price instead of close price...')
    #close = df2go.openPrice.values  # placeholder

    nfill = df2go.FillNbrForBin.values
    # historicals
    histVola = df2go.histVola.values
    mddv = df2go.mddv.values
    histVolaW = df2go.histVolaW.values
    histVoluW = df2go.histVoluW.values

    # Compute sizeShare (milan)
    #sizeShare0 = shareTraded / (mddv * 1000000 * histVoluW / close)  # Do not consider gap (convert MDDV to $)
    sizeShare = np.vectorize(compute_sizeShare)(shareTraded, mddv, histVoluW, close)
    sizeShare0 = sizeShare.copy()
    if np.sum(np.isnan(sizeShare))>0:
        print('** Warning! ** %d missing histVoluW... sizeShare set to NaN....'%np.sum(np.isnan(sizeShare)))

    iupdate = df2go.index[(df2go.firstTrade == 1) & (df2go.bin != 1)]

    # # Since bin==1 are left out, set the very1stTrade indicator
    # very1stTrade[(df2go.firstTrade == 1) & (df2go.bin == 1)] = 1

    print('- Number of priceBM to update:', len(iupdate))
    for i in iupdate:
        # print('i:',i)
        # Initialize
        sumVolaW = histVolaW[i] * histVolaW[i]
        sumVoluW = histVoluW[i]
        MQ1 = MQ[i]
        allmaxP = [maxP[i]]
        allminP = [minP[i]]
        # Search back until start of the day (Todo: optimize)
        j = i - 1  # one back
        while True:
            # print('j:',j)
            if bin[j] == 1:
                very1stTrade[i] = 1
                break  # reached the 2nd bin, no more update

            if nfill[j] != 0:
                # Last trade found
                # print('-filled newBM[%d] with: %2.2f'%(i,MQ[j]))
                newBM[i] = MQ[j]  # last active bin price
                newHistVola30min[i] = histVola[i] * np.sqrt(
                    sumVolaW)  # histVola * sqrt(sum(weights^2)) : accumulated weight
                newHistVolu30min[i] = mddv[i] * sumVoluW  # MDDV * sum(shares): accumulated weights
                # compute HL volatility
                MQ0 = MQ[j + 1]  # last.activeBin+1
                newRealVola[i] = jifin.compute_vola_maxmin(np.max(allmaxP), np.min(allminP), MQ0, MQ1)
                # Size (milan) adjusted
                #sizeShare[i] = shareTraded[i] / (mddv[i] * 1000000 * sumVoluW / close[i])  # convert MDDV (mln) to $1
                sizeShare[i] = np.vectorize(compute_sizeShare)(shareTraded[i], mddv[i], sumVoluW, close[i])
                # duration
                durationFromLast[i] = i - j - 1
                break
            else:
                # sum-up weights
                sumVolaW += histVolaW[j] * histVolaW[j]
                sumVoluW += histVoluW[j]
                # keep all max/min
                if (maxP[j] != 0) & (minP[j] != 0):  # fix the high volatility ~2 when minP=0 (2019/04/08)
                    allmaxP.append(maxP[j])
                    allminP.append(minP[j])
                # next iteration
                j += -1

    ## Update the very1stTrade that happened at Bin1
    iupdate = df2go.index[(df2go.firstTrade == 1) & (df2go.bin == 1)]
    for i in iupdate:
        very1stTrade[i] = 1
        #very1stTrade[i + 1:i + 78] = 0  # erase the other ones in the day since there should be only one on the day (IMPORTANT: Assume bins are in order)
        very1stTrade[i + 1:i + 13] = 0  # erase the other ones in the day since there should be only one on the day (IMPORTANT: Assume bins are in order)
    print('- Update very1stTrade that happened at Bin1...')

    # Update dataframe
    df2go['adjBM'] = newBM
    df2go['adjHistVola30min'] = newHistVola30min
    df2go['adjHistVolu30min'] = newHistVolu30min
    df2go['adjRealVola'] = newRealVola
    df2go['sizeShare'] = sizeShare0
    df2go['adjSizeShare'] = sizeShare
    df2go['gapDuration'] = durationFromLast
    df2go['very1stTrade'] = very1stTrade

    return df2go

def df2go_aggregate(df2go):
    """Aggregate variables from 5min to 30min bins
    - Missing LowPrice and HighPrice: they are ignored. However, weights for histVola and histVolu are also ignored together.
    So far checked, those bins have shareVolume=0, what is odd anyway..."""
    print('- Aggregating 5min to 30min bins: %d total bins...'%(df2go.shape[0]))
    tic = time.time()

    ## 1) Create the 30min groups
    df2go['bin30'] = ((df2go.bin-1)//6+1)
    df2go['token'] = df2go.token.apply(lambda s: s.split('_bin')[0]) # remove the bin number from token
    print('- Created bin30 columns and updated token.')

    ## 2) Fix some values to avoid outliers
    df2go[df2go['LowPrice']==0] = np.nan # replace zero by nan to use nanmin which ignores NaN

    # ## 2) Define cols to include and the aggregation to be used
    # # adjBM is the MQ of the last.ActiveBin+1
    # cols2include = ['token', 'tradeDate','bin30',
    #         'relSize','HLG', 'mddv', 'algoType','SpreadCapture', 'binDuration',
    #      'ExPriceForBin', 'ExShareForBin',
    #     'FillNbrForBin', 'midQuote', 'LowPrice', 'HighPrice',
    #    'shareVolume', 'histVolaW', 'histVoluW',
    #                 'pred30min']

    def myavg(x,df2go):
        if np.sum(x==0) == len(x):
            return 0
        else:
            return np.average(x, weights=df2go.loc[x.index, "ExShareForBin"])


    ## 3) Define aggregation function
    f = {'tradeDate':['first'],'histVola':['first'],'closeTm1':['first'],'relSize':['first'],'HLG':['first'], 'mddv':['first'], 'algoType':['first'],'SpreadCapture':['first'], 'binDuration':['first'],
             'SpreadCapture': ['first'], 'binDuration': ['first'],
             #'ExPriceForBin': lambda x: np.average(x, weights=df2go.loc[x.index, "ExShareForBin"]) # weighted average
         'ExPriceForBin': lambda x: myavg(x,df2go)
         ,'ExShareForBin':['sum'],'FillNbrForBin':['sum'],
        'midQuote':['first','last'],
         'LowPrice':lambda x: np.nanmin(df2go.loc[x.index,"LowPrice"]),'HighPrice':['max'],
       'shareVolume':['sum'],
         'histVolaW':lambda x: np.sqrt(np.sum(np.square(x))), 'histVoluW':['sum'],
         'pred30min':['first']}
    colslist = ['token','bin','tradeDate',
                'histVola','close','relSize','HLG', 'mddv', 'algoType','SpreadCapture', 'binDuration',
         'ExPriceForBin', 'ExShareForBin',
        'FillNbrForBin', 'firstMQ','lastMQ', 'LowPrice', 'HighPrice',
       'shareVolume', 'histVolaW', 'histVoluW',
                    'pred30min']

    df2go30min = df2go.groupby(['token','bin30']).agg(f).reset_index()
    print('- Done aggregating: time taken %d minutes.'%((time.time()-tic)//60))
    #print(df2go30min.columns)
    df2go30min.columns = colslist

    return df2go30min


def binToTime(binNumber,binSize,offset=0,endind=0):
    '''Function to turn bin number to time (HH:MM:SS) based on given bin size in seconds'''
    timestamp = str(datetime.timedelta(seconds=int((binNumber+offset)*binSize)))
    if timestamp[0:5] == '1 day':
        timestamp = timestamp[6:]
    if timestamp[0] == ' ':
        timestamp = timestamp[1:]
    if len(timestamp) == 7:
        timestamp = '0'+timestamp
    if endind & (timestamp == '00:00:00'):
        timestamp = '23:59:59'
    return timestamp

def get_table_buckets_SizexDuration(df,valname, sname, dname, sbuckets, dbuckets):
    """Bucket by abs(df[xname])"""
    cc = ['t' + str(d) + ' & $' + str(s) for d in dbuckets for s in sbuckets]
#    dfx = pd.DataFrame(index=df.ccy.unique(), columns=cc)
    dfCostMean = pd.DataFrame(index=df.ccy.unique(), columns=cc)
    dfCostStd = pd.DataFrame(index=df.ccy.unique(), columns=cc)
    # dfFXACEMean = pd.DataFrame(index=df.ccy.unique(),columns=cc)
    for ccy in df.ccy.unique():
        dfccy = df[df.ccy == ccy]
        for d in dbuckets:
            for s in sbuckets:
                ccol = 't' + str(d) + ' & $' + str(s)
                # EmpCost
                dfCostMean.loc[ccy, ccol] = np.mean(dfccy.loc[(dfccy[dname].abs() >= d[0]) & (dfccy[dname].abs() <= d[1]) & \
                                                          (dfccy[sname].abs() >= s[0]) & (
                                                                      dfccy[sname].abs() <= s[1]),valname])
                dfCostStd.loc[ccy, ccol] = np.std(dfccy.loc[(dfccy[dname].abs() >= d[0]) & (dfccy[dname].abs() <= d[1]) & \
                                                        (dfccy[sname].abs() >= s[0]) & (
                                                                    dfccy[sname].abs() <= s[1]),valname])
#    dfx['total'] = dfx.sum(axis=1)
    return dfCostMean, dfCostStd

def helper_plot_SizexDuration(df, tlabel, sbuckets, dbuckets, aggrMethod='Mean', xAbsScale=False):
    #FXACEPairUniverse = ['AUD.USD']
    FXACEPairUniverse = df.index
    # Set DF to use
    dfxx = df.copy()

    ## EmpCost (1 plot per ccy sinze they are in different scales...)
    nccy = len(FXACEPairUniverse)
    f, axs = plt.subplots(nccy, 1, figsize=(10, 7*nccy))
    # for i,ccy in enumerate(dfxx.index):
    for i, ccy in enumerate(FXACEPairUniverse):
        yall = []
        for d in dbuckets:
            y = []
            for s in sbuckets:
                ccol = 't' + str(d) + ' & $' + str(s)
                y.append(dfxx.loc[ccy, ccol])

            plt.subplot(len(FXACEPairUniverse), 1, int(str(i + 1)))
            # by ccy
            if xAbsScale:
                x = [np.mean(i) for i in sbuckets]
            else:
                x = range(len(sbuckets))

            if tlabel == 'nClusters(log)':
                y0 = y
                y = np.log(y0)
                plt.ylabel('log(nClusters)')

            plt.plot(x, y, linewidth=3)
            plt.xlabel('Size(US$ million)', fontsize=16)
            yall.append(y)
            # plt.fill_between(x,y-ystd,y+ystd,color='b',alpha=0.2)
        yall = np.array(yall)
        if aggrMethod == 'Mean':
            # plt.plot(x,yall.mean(axis=0),linewidth=7)
            plt.plot(x, yall.mean(axis=0), '--', linewidth=5)

        plt.title('%s: %s' % (tlabel, ccy), fontsize=20)
        if not xAbsScale:
            plt.xticks(range(len(sbuckets)), sbuckets, fontsize=16, rotation=10)
        else:
            plt.xticks(fontsize=16)
        aux = [str(i) for i in dbuckets]
        aux.append('Mean')
        #print(aux)
        plt.legend(aux, fontsize=16)
    f.subplots_adjust(hspace=.35)
    plt.show()

def get_intDate2weekday(mydate):
    """Given a date as integer yyyymmdd,
    it returns the day of the week.
    0: Monday
    ...
    6: Sunday"""
    dd = datetime.datetime(int(str(mydate)[0:4]),int(str(mydate)[4:6]),int(str(mydate)[6:8]))
    #print(dd)
    return dd.weekday()

def remove_outliers_df_2tails(dfx,percent,vars2rem,showplot=True):
    df = dfx.copy()
    print('Before:',df.shape)
    for i in vars2rem:
        tmax = np.percentile(df[i],100-percent)
        mymax = np.max(df[i])
        tmin = np.percentile(df[i], percent)
        mymin = np.min(df[i])
        # Remove
        y2exclude = (np.abs(df[i].values) < tmin) | (np.abs(df[i].values) > tmax)
        df = df.loc[~y2exclude]
        # t2add = 'excluded %d outliers (min=%2.4f)'%(y2exclude.sum(),mymin)
        t2add = 'excluded %d outliers (min=%2.4f, max=%2.4f)' % (np.sum(y2exclude), mymin, mymax)

        # plot
        if showplot:
            sns.distplot(df[i])
            plt.plot([tmin, tmin], [0,1])
            plt.plot([tmax, tmax], [0, 1])
            plt.text(tmin-0.1, .5,'%s%% treshold:%2.2f'%(percent,tmin), bbox=dict(facecolor='red', alpha=0.5))
            plt.text(tmax-1.5, .75, '%s%% treshold:%2.2f' % (percent,tmax), bbox=dict(facecolor='red', alpha=0.5))
            plt.title('Removing outliers: '+ i +' (%s)'%(t2add))
            plt.show()
    print('After:',df.shape)
    return df

def remove_outliers_df_up(dfx,percent,vars2rem,showplot=True):
    df = dfx.copy()
    print('Before:',df.shape)
    for i in vars2rem:
        t = np.percentile(df[i],100-percent)
        mymax = np.max(df[i])
        # remove
        y2exclude = np.abs(df[i].values)>t
        df = df.loc[~y2exclude]
        #t2add = 'excluded %d outliers (max=%2.4f)'%(y2exclude.sum(),mymax)
        t2add = 'excluded %d outliers (max=%2.4f)' % (np.sum(y2exclude), mymax)
        # plot
        if showplot:
            sns.distplot(df[i])
            plt.plot([t, t], [0,1])
            plt.text(t-1.1, .5,'%s%% treshold:%2.2f'%(percent,t), bbox=dict(facecolor='red', alpha=0.5))
            plt.title('Removing outliers: '+ i +' (%s)'%(t2add))
            plt.show()
    print('After:',df.shape)
    return df

def remove_outliers_df_down(dfx,percent,vars2rem,showplot=True):
    df = dfx.copy()
    print('Before:',df.shape)
    for i in vars2rem:
        t = np.percentile(df[i],percent)
        mymin = np.min(df[i])
        # remove
        y2exclude = np.abs(df[i].values)<t
        df = df.loc[~y2exclude]
        #t2add = 'excluded %d outliers (min=%2.4f)'%(y2exclude.sum(),mymin)
        t2add = 'excluded %d outliers (min=%2.4f)' % (np.sum(y2exclude), mymin)
        if showplot:
            # plot
            sns.distplot(df[i])
            plt.plot([t, t], [0,1])
            plt.text(t-1.1, .5,'%s%% treshold:%2.2f'%(percent,t), bbox=dict(facecolor='red', alpha=0.5))
            plt.title('Removing outliers: '+ i +' (%s)'%(t2add))
            plt.show()
    print('After:',df.shape)
    return df

def get_condranges(ds,condrange):
    if type(condrange) == int:
        ## if too many zeros, it creates noise around zero...
        #print('ds:',ds)
        if np.sum(ds==0)/ds.shape[0] > 1/condrange:
            print('(*) Too many zeros in the condition variable (n=%d)! Fill zeros with small random numbers...'%np.sum(ds==0))
            ds.loc[ds==0] = 0.0001 * np.random.rand(np.sum(ds==0))

        out, bins = pd.qcut(ds, q=condrange, labels=False, retbins=True)
        myranges = [[float('%2.4f'%bins[i]), float('%2.4f'%bins[i + 1])] for i in range(len(out.unique()))]

        #ncond = condrange
    else:
        myranges = condrange
        #ncond = len(condrange)

    return myranges

def run_discretize(ds,condrange,label):
    """Given a DSeries and condrange, it will discretize.
    If int is given, it will split automatically based on quantiles."""
    categories = [] # Use string
    ds_discrete = ds.copy()
    if type(condrange) == int:
        out, bins = pd.qcut(ds, q=condrange, labels=False, retbins=True)
        myranges = [[bins[i], bins[i + 1]] for i in range(len(out.unique()))]
        ncond = condrange
    else:
        myranges = condrange
        ncond = len(condrange)

    for i in range(1, ncond + 1):
        crange = myranges[i - 1]
        if type(crange[0]) != str:
            aux = '%s:%2.6f-%2.6f' % (label,crange[0], crange[1])
            ds_discrete[ds.between(crange[0], crange[1])] = aux

    return ds_discrete

def compute_oneway_anova(Y):
    """Y should be a list of arrays. Dumpy method... up to 5 conditions... add more if necessary
    scypy does not accept list as input..."""
    F,p = 0,0
    if len(Y)== 2:
        F, p = stats.f_oneway(Y[0], Y[1])
    if len(Y) == 3:
        F, p = stats.f_oneway(Y[0], Y[1],Y[2])
    if len(Y) == 4:
        F, p = stats.f_oneway(Y[0], Y[1],Y[2],Y[3])
    if len(Y) == 5:
        F, p = stats.f_oneway(Y[0], Y[1],Y[2],Y[3], Y[4])
    if len(Y) == 6:
        F, p = stats.f_oneway(Y[0], Y[1],Y[2],Y[3], Y[4], Y[5])
    print('** One-way ANOVA: F-value%2.2f, p-value=%2.2e'%(F,p))
    return F,p

def compute_return5min_silent(MQ,priceBM):
    """Compute the return within 5min bin (in bps)
    return = (MQ[j]-MQ[j-1])/MQ[j-1]*10000 = (MQ[j]-BM[j])/BM[j]*10000
    Each row is a sample to compute"""

    if priceBM == 0:
        #print('- Warning! Missing priceBM. Return set to None...')
        return None
    else:
        return (MQ - priceBM) / priceBM * 10000

def compute_signedReturn5min_silent(MQ,priceBM,sign):
    """Compute the return within 5min bin (in bps)
    return = (MQ[j]-MQ[j-1])/MQ[j-1]*SIGN*10000 = (MQ[j]-BM[j])/BM[j]*SIGN*10000
    Each row is a sample to compute"""
    if priceBM == 0:
        #print('- Warning! Missing priceBM. SignedReturn set to None...')
        return None
    else:
        return sign*(MQ-priceBM)/priceBM*10000

def compute_return5min(MQ,priceBM):
    """Compute the return within 5min bin (in bps)
    return = (MQ[j]-MQ[j-1])/MQ[j-1]*10000 = (MQ[j]-BM[j])/BM[j]*10000
    Each row is a sample to compute"""

    if priceBM == 0:
        print('- Warning! Missing priceBM. Return set to None...')
        return None
    else:
        return (MQ - priceBM) / priceBM * 10000

def compute_signedReturn5min(MQ,priceBM,sign):
    """Compute the return within 5min bin (in bps)
    return = (MQ[j]-MQ[j-1])/MQ[j-1]*SIGN*10000 = (MQ[j]-BM[j])/BM[j]*SIGN*10000
    Each row is a sample to compute"""
    if priceBM == 0:
        print('- Warning! Missing priceBM. SignedReturn set to None...')
        return None
    else:
        return sign*(MQ-priceBM)/priceBM*10000

def compute_return15sec(M,nwin):
    """Compute the cumulative return (sum of relative differences)
    return = sum(r), where r[t]=BM[t]-BM[t-1]
    Each row is a sample to compute"""
    return np.sum(np.diff(M)/M[:, :nwin - 1],axis=1)*10000

def compute_vola_quadratic(M,nwin):
    """Compute the quadratic volatility (square root of the sum of relative quadratic differences)
    qvola = sqrt(sum(r**2)), where r[t]=(r[t]-r[t-1])/r[t-1]
    Each row is a sample to compute"""
    #M = dfmerged.iloc[:, 2:nwin+2].values
    #r = np.diff(M) / M[:, :nwin - 1]  # return
    #qv = np.sqrt(np.sum(np.square(r), axis=1))
    return np.sqrt(np.sum(np.square( np.diff(M) / M[:, :nwin - 1] ), axis=1))

def compute_vola_std(M,nwin):
    """Compute the std-based volatility (Anupama's version)
    Each row is a sample to compute"""
    #M = dfmerged.iloc[:, 2:nwin+2].values
    r = np.diff(M) / M[:, :nwin - 1]  # return

    return r.std(axis=1)*np.sqrt(nwin) # Anupama's std-based vola

def compute_prate_silent(exVol,mkVol):
    """Compute the participation rate as: Execution Vol / Market Vol."""
    if mkVol!=0:
        x = exVol/mkVol
        if x>1:
            #print('** Warning! Execution Vol (%2.4f) > Market Vol (%2.4f). prate set to 1... **'%(exVol,mkVol))
            x=1
    else:
        return 0
    return x

def compute_prate(exVol,mkVol):
    """Compute the participation rate as: Execution Vol / Market Vol."""
    if mkVol!=0:
        x = exVol/mkVol
        if x>1:
            print('** Warning! Execution Vol (%2.4f) > Market Vol (%2.4f). prate set to 1... **'%(exVol,mkVol))
            x=1
    else:
        return 0
    return x

def compute_style_silent(exPrice,vwap,bmPrice,prate,sign):
    """Compute the Style as a function of exPrice, vwap, bmPrice, prate and sign (bps). It is signed.
    See RefDoc for details."""
    x = 0.0
    if prate ==1:
        #print('** Warning! prate=1 (exPrice=%2.4f, vwap=%2.4f, bmPrice=%2.4f) --> not possible to estimate Style...**'%(exPrice,vwap,bmPrice))
        return -123456 #
    if bmPrice == 0:
        #print('** Warning! bmPrice=0 (exPrice=%2.4f, vwap=%2.4f, bmPrice=%2.4f) --> not possible to estimate Style...**' % (exPrice, vwap, bmPrice))
        return -1234567  #

    if exPrice!=0: # skip zero execution price
        x = sign*(exPrice-vwap)/(bmPrice*(1-prate))*10000

    return x

def compute_style(exPrice,vwap,bmPrice,prate,sign):
    """Compute the Style as a function of exPrice, vwap, bmPrice, prate and sign (bps). It is signed.
    See RefDoc for details."""
    x = 0.0
    if prate ==1:
        print('** Warning! prate=1 (exPrice=%2.4f, vwap=%2.4f, bmPrice=%2.4f) --> not possible to estimate Style...**'%(exPrice,vwap,bmPrice))
        return 0 #
    if bmPrice == 0:
        print('** Warning! bmPrice=0 (exPrice=%2.4f, vwap=%2.4f, bmPrice=%2.4f) --> not possible to estimate Style...**' % (
            exPrice, vwap, bmPrice))
        return 0  #

    if exPrice!=0: # skip zero execution price
        x = sign*(exPrice-vwap)/(bmPrice*(1-prate))*10000

    return x

def compute_style_normalized_silent(exPrice,vwap,maxP,minP,prate,sign):
    """Compute the Style as a function of exPrice, vwap, bmPrice, prate and sign (bps). It is signed.
    See RefDoc for details."""
    x = 0.0
    if prate ==1:
        #print('** Warning! prate=1 (exPrice=%2.4f, vwap=%2.4f, maxP=%2.4f) --> not possible to estimate Style...**' % (exPrice, vwap,maxP))
        return -123456 #
    #if bmPrice == 0:
    #    print('** Warning! bmPrice=0 (exPrice=%2.4f, vwap=%2.4f, bmPrice=%2.4f) --> not possible to estimate Style...**' % (
    #        exPrice, vwap, bmPrice))
    #    return 0  #
    if exPrice!=0: # skip zero execution price
        x = sign*(exPrice-vwap)/(np.max([maxP-minP,0.01])*(1-prate))

    return x

def compute_style_normalized(exPrice,vwap,maxP,minP,prate,sign):
    """Compute the Style as a function of exPrice, vwap, bmPrice, prate and sign (bps). It is signed.
    See RefDoc for details."""
    x = 0.0
    if prate ==1:
        print('** Warning! prate=1 (exPrice=%2.4f, vwap=%2.4f, maxP=%2.4f) --> not possible to estimate Style...**' % (
            exPrice, vwap,maxP))
        return 0 #
    #if bmPrice == 0:
    #    print('** Warning! bmPrice=0 (exPrice=%2.4f, vwap=%2.4f, bmPrice=%2.4f) --> not possible to estimate Style...**' % (
    #        exPrice, vwap, bmPrice))
    #    return 0  #
    if exPrice!=0: # skip zero execution price
        x = sign*(exPrice-vwap)/(np.max([maxP-minP,0.01])*(1-prate))

    return x

def compute_sizeShare(shareTraded,mddv,histVoluW,close):
    if histVoluW!=0:
        sizeShare = shareTraded / (mddv * 1000000 * histVoluW / close)  # Do not consider gap (convert MDDV to $)
    else:
        #print('** Warning! ** Missing histVoluW... sizeShare set to NaN....')
        sizeShare = np.nan
    return sizeShare

def compute_empCost_silent(exPrice,bmPrice,sign):
    """Compute the Empirical Cost accordingly to execution price (bps). It is signed.
    See RefDoc for details."""
    x = 0.0
    if bmPrice==0:
        #print('- Missing bmPrice. Cost set to NaN...')
        x = np.nan
    elif exPrice!=0: # skip zero execution price
        x = sign*(exPrice-bmPrice)/bmPrice*10000

    return x

def compute_empCost(exPrice,bmPrice,sign):
    """Compute the Empirical Cost accordingly to execution price (bps). It is signed.
    See RefDoc for details."""
    x = 0.0
    if bmPrice==0:
        print('- Missing bmPrice. Cost set to ZERO...')
    elif exPrice!=0: # skip zero execution price
        x = sign*(exPrice-bmPrice)/bmPrice*10000

    return x

def compute_F(p1,p0,vwap,sign):
    """Compute the F Cost accordingly to initial nad final MQ prices. It is signed.
    vwap is used just to check if there was some trade.
    midQuote (thus priceBM) are not reliable indicator since they happen to be non-zero for non-traded bins...
    See RefDoc for details."""
    if (vwap!=0) & (p0!=0):
        x = sign*0.5*(p1-p0)/p0*10000
    else:
        return 0
    return x

def compute_G(p1,p0,vwap,sign):
    """Compute the G Cost accordingly to initial nad final MQ prices, and vwap. It is signed.
    See RefDoc for details."""
    if (vwap!= 0) & (p0!=0):
        x = sign*(vwap-(p1+p0)/2)/p0*10000
    else:
        return 0
    return x

def compute_empCostMaxMin(exPrice,bmPrice,maxP,minP,sign):
    """Compute the Empirical Cost accordingly to execution price (bps). It is signed.
    Normalize with maxP-minP instead of bmPrice
    See RefDoc for details."""
    x = 0.0
    if exPrice!=0: # skip zero execution price
        x = sign*(exPrice-bmPrice)/np.max([maxP-minP,0.01])

    return x

def compute_FMaxMin(p1,p0,vwap,maxP,minP,sign):
    """Compute the F Cost accordingly to initial nad final MQ prices. It is signed.
    vwap is used just to check if there was some trade.
    midQuote (thus priceBM) are not reliable indicator since they happen to be non-zero for non-traded bins...
    See RefDoc for details."""
    if vwap!=0:
        x = sign*0.5*(p1-p0)/np.max([maxP-minP,0.01])
    else:
        return 0
    return x

def compute_GMaxMin(p1,p0,vwap,maxP,minP,sign):
    """Compute the G Cost accordingly to initial nad final MQ prices, and vwap. It is signed.
    See RefDoc for details."""
    if vwap != 0:
        x = sign*(vwap-(p1+p0)/2)/np.max([maxP-minP,0.01])
    else:
        return 0
    return x

def compute_vola_maxmin_silent(maxP,minP,firstMQ,lastMQ):
    """Given the max, min and MQs compute the ITG traditional maxmin or highlow estimate.
       Assume that input are single values, not arrays."""
    #tic = time.time()

    x = 0.0
    if (maxP - minP != 0) & (maxP!=0) & (minP!=0): # updated: 2019/04/08
        if firstMQ!=0: # corner case with MQ=0... ignore it
            x = (np.max([maxP,firstMQ,lastMQ])-np.min([minP,firstMQ,lastMQ]))/((maxP+minP)/2)
        else:
            x = (maxP - minP) / ((maxP + minP) / 2)
    elif (maxP==0) | (minP==0): # updated: 2019/04/15:
        #print('** Warning ** Volatility set to zero: maxP=%2.2f, min=%2.2f, firstMQ=%2.2f, lastMQ=%2.2f' % (maxP, minP, firstMQ, lastMQ))
        return np.nan   # zero to NaN instead of zero
    else: # - setting 0.01 as the lowest price variation when maxP==minP
        if ((np.max([maxP, firstMQ, lastMQ]) - np.min([minP, firstMQ, lastMQ]))==0):  # corner case without change in price
            x = 0.01 / ((maxP + minP) / 2)
            #print('** Warning! ** Unchanged price --> set 0.01 --> Vola=%2.4f: maxP=%2.2f, min=%2.2f, firstMQ=%2.2f, lastMQ=%2.2f'%(x,maxP, minP, firstMQ, lastMQ))
        return x

    return x

def compute_vola_maxmin(maxP,minP,firstMQ,lastMQ):
    """Given the max, min and MQs compute the ITG traditional maxmin or highlow estimate.
       Assume that input are single values, not arrays."""
    #tic = time.time()

    #if maxP-minP!=0:
    x = 0.0
    if (maxP - minP != 0) & (maxP!=0) & (minP!=0): # updated: 2019/04/08
        if firstMQ!=0: # corner case with MQ=0... ignore it
            x = (np.max([maxP,firstMQ,lastMQ])-np.min([minP,firstMQ,lastMQ]))/((maxP+minP)/2)
        else:
            x = (maxP - minP) / ((maxP + minP) / 2)
    elif (maxP==0) | (minP==0): # updated: 2019/04/15:
        print('** Warning ** Volatility set to zero: maxP=%2.2f, min=%2.2f, firstMQ=%2.2f, lastMQ=%2.2f' % (maxP, minP, firstMQ, lastMQ))
        return np.nan   # zero to NaN instead of zero
    else: # - setting 0.01 as the lowest price variation when maxP==minP
        if ((np.max([maxP, firstMQ, lastMQ]) - np.min([minP, firstMQ, lastMQ]))==0):  # corner case without change in price
            x = 0.01 / ((maxP + minP) / 2)
            print('** Warning! ** Unchanged price --> set 0.01 --> Vola=%2.4f: maxP=%2.2f, min=%2.2f, firstMQ=%2.2f, lastMQ=%2.2f'%(x,maxP, minP, firstMQ, lastMQ))
        #print('** Warning ** Volatility set to zero: maxP=%2.2f, min=%2.2f, MQ=%2.2f '%(maxP,minP,firstMQ))
        #if (maxP == 0) | (minP == 0):
        #    print('maxP or minP is ZERO ')
        return x

    #print('MaxMin Volatility computed in %s seconds.'%(time.time()-tic))
    #print('- realVola=%2.6f, where maxP=%2.2f, min=%2.2f, MQ=%2.2f '%(x,maxP,minP,firstMQ))
    return x

def my_divide(realized,historical):
    """Very simple function to divide efficiently."""
    if historical==0:
        print('- Missing denominator... return set to None...')
        return None
    else:
        return realized/historical

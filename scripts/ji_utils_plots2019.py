"""
================================================================
     Copyright (c) 2018, Yale University
                     All Rights Reserved
================================================================

 NAME : ji_utils_plots.py
 @DATE     Created: 12/4/18 (11:22 AM)
	       Modifications:
	       - 12/4/18:

 @AUTHOR          : Jaime Shinsuke Ide
                    jaime.ide@yale.edu
===============================================================
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ji_utils_fin2019 as jifin
import seaborn as sns
import datetime
import CostCurve
import os
import pandas as pd

from scipy.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


global cont


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_two_axes(npa1,npa2,ylabel1,ylabel2):
    '''Plot figure with two axes'''
    fig, ax1 = plt.subplots(figsize=(13,5))

    ax1.plot(range(len(npa1)), npa1, 'b-')
    ax1.set_xlabel('xscale')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel(ylabel1, color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(range(len(npa2)), npa2, 'r-')
    ax2.set_ylabel(ylabel2, color='r')
    ax2.tick_params('y', colors='r')

    # ax2.plot(range(len(npa2)), 2*npa2, 'g-')
    # ax2.set_ylabel('third', color='g')
    #ax2.tick_params('y', colors='g')

    fig.tight_layout()
    plt.show()

def plot_crossCorr_df2go(dfx, condvar,title='Cross-correlation'):
    """Plot cross correlation scatter plot.
    """
    global cont
    global nconditions

    ## 1) Scatter plot
    plotScatter = False
    if plotScatter:
        sns.pairplot(dfx, hue=condvar, diag_kind='kde',
                     plot_kws={'alpha': 0.1, 's': 20, 'edgecolor': 'k'},
                     height=4)

    ## 2) Annotate correlations
    # Create a pair grid instance
    #grid = sns.PairGrid(data= dfx, vars = cols2include, hue = 'condition',
                        #hue_kws = {'alpha': 0.4},
    grid = sns.PairGrid(data=dfx, hue=condvar,
                        height = 4)
    # Map the plots to the locations
    #grid = grid.map_upper(plt.scatter, alpha = 0.5)
    #grid = grid.map_upper(plt.scatter, alpha = 0.5)
    #grid = grid.map_upper(corr_txt)
    grid = grid.map_offdiag(plt.scatter, alpha = 0.5)
    cont = 0
    nconditions = len(dfx[condvar].unique())
    #grid = grid.map_offdiag(annotate_corr_txt,kwargs={'cont':0})
    grid = grid.map_offdiag(annotate_corr_txt2)
    #grid = grid.map_lower(sns.kdeplot ,  cmap="Blues_d")
    print('- Diagonals: histogram')
    grid = grid.map_diag(plt.hist, bins = 20, histtype="step", linewidth=4)
    #grid = grid.map_diag(sns.kdeplot)
    grid = grid.add_legend()

    plt.subplots_adjust(top=0.9)
    grid.fig.suptitle(title,fontsize = 20)  # can also get the figure from plt.gcf()

    for i in dfx[condvar].unique():
        print('Nbr of ',i,':',np.sum(dfx[condvar]==i))

def annotate_corr_txt2(x, y,**kwargs):
    global cont
    global nconditions
    # Calculate the value
    #coef = np.corrcoef(x, y)[0][1]
    coef, pvalue = pearsonr(x,y)
    # Make the label
    #label = r'$\rho$ = ' + str(round(coef, 2)) + ' cont=%d'%cont
    label = r'$\rho_{%d}$ = %s, p=%1.2e'%(cont,str(round(coef, 2)),pvalue)
    #print('label: ',label)
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy=(0.1, 0.95-0.1*cont), size=20, xycoords=ax.transAxes)
    cont+=1
    if cont%nconditions == 0:
        cont = 0

def helper_get_Aggregate(df2gox, sizeVar, targetVar, quantiles):
    ## 1) Split size by quantile
    SizeThr = "0"
    for q in quantiles:
        qThr = round(df2gox[sizeVar].quantile(q), 4)
        SizeThr = SizeThr + ":" + str(qThr)
    SizeThr = SizeThr + ":1000000000"
    SizeThr = np.fromstring(SizeThr, dtype=float, sep=":")

    ## 2) Aggregate by size
    ColAgg = ['normEmpCost', 'normSignedReturn', 'normAdjSignedReturn', 'prate', 'adjSizeShare']

    # Cost (to get the StdErr as well, not only the average (StdErr is computed on the col2filter)
    dfagg = CostCurve.CostCurve(df2gox.copy(), None, SizeThr, ColAgg, sizeVar, targetVar, sizeVar, 0, 0.01, PrintNA=False)

    return dfagg

def helper_plot_Cost_Return_bySize_v2(df2gox,sizeVar,sizeLabel, mylabel, quantiles = [0.3, 0.6, 0.9, 0.97], dur = [0,0]):
    """Same as the previous version but now the bucketing is done for each group"""
    # Defaults
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    myfontsize = 16
    #quantiles = [0.3, 0.6, 0.9, 0.97]
    zscore = 1.96  # CI: 95%
    # zscore = 1.645 # CI: 90%
    includeSEM = True

    ################################## ajdSizeShare ##########################################################
    targetVar = 'normAdjSignedReturn'
    targetVarLabel = 'normAdjSignReturn'
#    sizeVar = 'adjSizeShare'
#    sizeLabel = 'size (nShare/histADV)'

    # Cost (to get the StdErr as well, not only the average (StdErr is computed on the col2filter)
    dfagg_all_cost = helper_get_Aggregate(df2gox.copy(), sizeVar, 'normEmpCost', quantiles)
    if dur == [0,0]:
        dfagg_very1stTrade_cost = helper_get_Aggregate(df2gox[(df2gox.very1stTrade == 1)].copy(), sizeVar, 'normEmpCost', quantiles)
        dfagg_not1stTrade_cost = helper_get_Aggregate(df2gox[(df2gox.firstTrade != 1)].copy(), sizeVar, 'normEmpCost', quantiles)
    else: # there is no 1stTrade exclusive that has duration 0
        dfagg_1stTrade_cost = helper_get_Aggregate(df2gox[(df2gox.firstTrade == 1) & (df2gox.very1stTrade != 1)].copy(),
                                                   sizeVar, 'normEmpCost', quantiles)
    print('- Filtering by %s...' % 'normEmpCost')

    # Return
    dfagg_all = helper_get_Aggregate(df2gox.copy(), sizeVar, targetVar, quantiles)
    if dur == [0, 0]:
        dfagg_very1stTrade = helper_get_Aggregate(df2gox[(df2gox.very1stTrade == 1)].copy(), sizeVar, targetVar, quantiles)
        dfagg_not1stTrade = helper_get_Aggregate(df2gox[(df2gox.firstTrade != 1)].copy(), sizeVar, targetVar, quantiles)
    else:
        dfagg_1stTrade = helper_get_Aggregate(df2gox[(df2gox.firstTrade == 1) & (df2gox.very1stTrade != 1)].copy(),
                                              sizeVar, targetVar, quantiles)

    print('- Filtering by %s...' % targetVar)

    cols2print = ['normEmpCost','normAdjSignedReturn','StdErr',sizeVar,'Count']
    print('** Aggregate all-trades ** \n',dfagg_all[cols2print])
    if dur == [0, 0]:
        print('** Aggregate Very1st-trades ** \n', dfagg_very1stTrade[cols2print])
        print('** Aggregate Not1st-trades ** \n', dfagg_not1stTrade[cols2print])
    else:
        print('** Aggregate 1st-trades ** \n', dfagg_1stTrade[cols2print])

    # Plot
    fig = plt.figure(figsize=(15, 4))

    ## Cost
    x1 = dfagg_all_cost[sizeVar]
    if dur == [0, 0]:
        x3 = dfagg_very1stTrade_cost[sizeVar]
        x4 = dfagg_not1stTrade_cost[sizeVar]
    else:
        x2 = dfagg_1stTrade_cost[sizeVar]

    plt.subplot(1, 2, 1)

    y1 = dfagg_all_cost['normEmpCost']
    if dur == [0, 0]:
        y3 = dfagg_very1stTrade_cost['normEmpCost']
        y4 = dfagg_not1stTrade_cost['normEmpCost']
    else:
        y2 = dfagg_1stTrade_cost['normEmpCost']

    plt.plot(x1, y1, label='All-trades', linewidth=3) #, plt.scatter(x,y1)
    if dur == [0, 0]:
        plt.plot(x3, y3, label='Very 1st-trades', linewidth=3) #, plt.scatter(x,y3)
        plt.plot(x4, y4, label='Not 1st-trades', linewidth=3)
    else:
        plt.plot(x2, y2, label='1st-trades (exclusive)', linewidth=3)  # , plt.scatter(x,y2)

    if includeSEM:
        ySEM = zscore * dfagg_all_cost['StdErr']
        plt.fill_between(x1, y1 - ySEM, y1 + ySEM, color='b', alpha=0.2)
        if dur == [0, 0]:
            ySEM = zscore * dfagg_very1stTrade_cost['StdErr']
            plt.fill_between(x3, y3 - ySEM, y3 + ySEM, color='g', alpha=0.2)
            ySEM = zscore * dfagg_not1stTrade_cost['StdErr']
            plt.fill_between(x4, y4 - ySEM, y4 + ySEM, color='r', alpha=0.2)
        else:
            ySEM = zscore * dfagg_1stTrade_cost['StdErr']
            plt.fill_between(x2, y2 - ySEM, y2 + ySEM, color='orange', alpha=0.2)

    plt.title('Cost x Size (%s)'%mylabel, fontsize=myfontsize)
    plt.ylabel('normEmpCost', fontsize=myfontsize-2)
    plt.xlabel(sizeLabel, fontsize=myfontsize-2)
    plt.legend()

    ## Return
    plt.subplot(1, 2, 2)
    x1 = dfagg_all[sizeVar]
    if dur == [0, 0]:
        x3 = dfagg_very1stTrade[sizeVar]
        x4 = dfagg_not1stTrade[sizeVar]
    else:
        x2 = dfagg_1stTrade[sizeVar]

    y1 = dfagg_all['normAdjSignedReturn']
    if dur == [0, 0]:
        y3 = dfagg_very1stTrade['normAdjSignedReturn']
        y4 = dfagg_not1stTrade['normAdjSignedReturn']
    else:
        y2 = dfagg_1stTrade['normAdjSignedReturn']

    plt.plot(x1, y1, label='All-trades', linewidth=3)#, plt.scatter(x,y1)
    if dur == [0, 0]:
        plt.plot(x3, y3, label='Very 1st-trades', linewidth=3)
        plt.plot(x4, y4, label='Not 1st-trades', linewidth=3)
    else:
        plt.plot(x2, y2, label='1st-trades (exclusive)', linewidth=3)

    if includeSEM:
        ySEM = zscore * dfagg_all['StdErr']
        plt.fill_between(x1, y1 - ySEM, y1 + ySEM, color='b', alpha=0.2)
        if dur == [0, 0]:
            ySEM = zscore * dfagg_very1stTrade['StdErr']
            plt.fill_between(x3, y3 - ySEM, y3 + ySEM, color='g', alpha=0.2)
            ySEM = zscore * dfagg_not1stTrade['StdErr']
            plt.fill_between(x4, y4 - ySEM, y4 + ySEM, color='r', alpha=0.2)
        else:
            ySEM = zscore * dfagg_1stTrade['StdErr']
            plt.fill_between(x2, y2 - ySEM, y2 + ySEM, color='orange', alpha=0.2)

    plt.title('Return x Size (%s)'%mylabel, fontsize=myfontsize)
    plt.ylabel('normReturn (adjusted)', fontsize=myfontsize - 2)
    plt.xlabel(sizeLabel, fontsize=myfontsize-2)
    plt.legend()

def helper_plot_Cost_Return_bySize_alldur_v2(df2gox,sizeVar,sizeLabel, mylabel, quantiles = [0.3, 0.6, 0.9, 0.97], multipleBins = False, durrange = [[1,3],[4,77]]):
    # Defaults
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    myfontsize = 16
    #quantiles = [0.3, 0.6, 0.9, 0.97]
    zscore = 1.96  # CI: 95%
    # zscore = 1.645 # CI: 90%
    includeSEM = True

    ################################## ajdSizeShare ##########################################################
    targetVar = 'normAdjSignedReturn'
    targetVarLabel = 'normAdjSignReturn'
#    sizeVar = 'adjSizeShare'
#    sizeLabel = 'size (nShare/histADV)'
    ## 1) Split size by quantile
    SizeThr = "0"
    for q in quantiles:
        qThr = round(df2gox[sizeVar].quantile(q), 4)
        SizeThr = SizeThr + ":" + str(qThr)
    SizeThr = SizeThr + ":1000000000"
    SizeThr = np.fromstring(SizeThr, dtype=float, sep=":")

    ## 2) Aggregate by size
    #ColAgg = ['normEmpCost', 'normSignedReturn', 'normAdjSignedReturn', 'prate', 'adjSizeShare']
    ColAgg = ['normEmpCost', 'normSignedReturn', 'normAdjSignedReturn', 'prate', 'adjSizeShare']
    if sizeVar not in ColAgg:
        ColAgg += [sizeVar]

    # Cost (to get the StdErr as well, not only the average (StdErr is computed on the col2filter)
    dfagg_all_cost = CostCurve.CostCurve(df2gox.copy(), None, SizeThr, ColAgg, sizeVar, 'normEmpCost', sizeVar, 0, 0.01,PrintNA=False)
    dfagg_very1stTrade_cost = CostCurve.CostCurve(df2gox[(df2gox.very1stTrade == 1)].copy(), None, SizeThr, ColAgg, sizeVar,
                                                  'normEmpCost', sizeVar, 0, 0.01,PrintNA=False)
    dfagg_not1stTrade_cost = CostCurve.CostCurve(df2gox[(df2gox.firstTrade != 1)].copy(),
                                             None, SizeThr, ColAgg, sizeVar, 'normEmpCost', sizeVar, 0, 0.01,
                                             PrintNA=False)
    dfagg_1stTrade_cost_1_3 = CostCurve.CostCurve(df2gox[(df2gox.firstTrade == 1) & (df2gox.very1stTrade != 1) & (df2gox.gapDuration.isin(durrange[0]))].copy(),
                                          None, SizeThr, ColAgg, sizeVar, 'normEmpCost', sizeVar, 0, 0.01,
                                          PrintNA=False)
    dfagg_1stTrade_cost_4_77 = CostCurve.CostCurve(
        df2gox[(df2gox.firstTrade == 1) & (df2gox.very1stTrade != 1) & (df2gox.gapDuration.isin(durrange[1]))].copy(),
        None, SizeThr, ColAgg, sizeVar, 'normEmpCost', sizeVar, 0, 0.01,
        PrintNA=False)

    print('- Filtering by %s...' % 'normEmpCost')

    # Return
    dfagg_all = CostCurve.CostCurve(df2gox.copy(), None, SizeThr, ColAgg, sizeVar, targetVar, sizeVar, 0, 0.01,
                                    PrintNA=False)
    dfagg_very1stTrade = CostCurve.CostCurve(df2gox[df2gox.very1stTrade == 1].copy(), None, SizeThr, ColAgg, sizeVar,
                                             targetVar, sizeVar, 0, 0.01, PrintNA=False)
    dfagg_not1stTrade = CostCurve.CostCurve(df2gox[(df2gox.firstTrade != 1)].copy(),
                                            None, SizeThr, ColAgg, sizeVar, targetVar, sizeVar, 0, 0.01,
                                             PrintNA=False)
    dfagg_1stTrade_1_3 = CostCurve.CostCurve(df2gox[(df2gox.firstTrade == 1) & (df2gox.very1stTrade != 1) & (df2gox.gapDuration.isin(durrange[0]))].copy(), None,
                                         SizeThr, ColAgg, sizeVar,
                                         targetVar, sizeVar, 0, 0.01, PrintNA=False)
    dfagg_1stTrade_4_77 = CostCurve.CostCurve(
        df2gox[(df2gox.firstTrade == 1) & (df2gox.very1stTrade != 1) & (df2gox.gapDuration.isin(durrange[1]))].copy(), None,
        SizeThr, ColAgg, sizeVar,
        targetVar, sizeVar, 0, 0.01, PrintNA=False)

    dur0label = '%s'%durrange[0]
    dur1label = '%s'% durrange[1]

    print('- Filtering by %s...' % targetVar)
    cols2print = ['normEmpCost','normAdjSignedReturn','StdErr',sizeVar,'Count']
    print('** Aggregate all-trades ** \n',dfagg_all[cols2print])
    print('** Aggregate Very1st-trades ** \n', dfagg_very1stTrade[cols2print])
    print('** Aggregate Not1st-trades ** \n', dfagg_not1stTrade[cols2print])
    print('** Aggregate 1st-trades dur=%s** \n'%dur0label, dfagg_1stTrade_1_3[cols2print])
    print('** Aggregate 1st-trades dur=%s ** \n'%dur1label, dfagg_1stTrade_4_77[cols2print])

    # Plot
    fig = plt.figure(figsize=(15, 4))

    ## Cost
    plt.subplot(1, 2, 1)
    # x-axis
    x1 = dfagg_all[sizeVar].copy()
    x3 = dfagg_very1stTrade[sizeVar].copy()
    x4 = dfagg_not1stTrade[sizeVar].copy()
    x2a = dfagg_1stTrade_1_3[sizeVar].copy()
    x2b = dfagg_1stTrade_4_77[sizeVar].copy()
    if multipleBins == False:
        x2a, x2b, x3, x4 = x1, x1, x1, x1 # Fixed x-axis

    # y-axis
    y1 = dfagg_all['normEmpCost']
    y3 = dfagg_very1stTrade['normEmpCost']
    y4 = dfagg_not1stTrade['normEmpCost']
    y2a = dfagg_1stTrade_1_3['normEmpCost']
    y2b = dfagg_1stTrade_4_77['normEmpCost']

    plt.plot(x1, y1, label='All-trades', linewidth=3) #, plt.scatter(x,y1)
    plt.plot(x3, y3, label='Very 1st-trades dur=[0,0]', linewidth=3, color='g') #, plt.scatter(x,y3)
    plt.plot(x4, y4, label='Not 1st-trades dur=[0,0]', linewidth=3,color='r')
    #plt.plot(x2a, y2a, label='1st-trades (exclusive) dur=SHORT ** \n', linewidth=3, color='orange')  # , plt.scatter(x,y2)
    #plt.plot(x2b, y2b, label='1st-trades (exclusive) dur=LONG ** \n', linewidth=3, color='cyan')  # , plt.scatter(x,y2)
    plt.plot(x2a, y2a, label='1st-trades (exclusive) dur=%s ** \n'%dur0label, linewidth=3, color='orange')  # , plt.scatter(x,y2)
    plt.plot(x2b, y2b, label='1st-trades (exclusive) dur=%s ** \n'%dur1label, linewidth=3, color='cyan')  # , plt.scatter(x,y2)

    if includeSEM:
        ySEM = zscore * dfagg_all_cost['StdErr']
        plt.fill_between(x1, y1 - ySEM, y1 + ySEM, color='b', alpha=0.2)
        ySEM = zscore * dfagg_very1stTrade_cost['StdErr']
        plt.fill_between(x3, y3 - ySEM, y3 + ySEM, color='g', alpha=0.2)
        ySEM = zscore * dfagg_not1stTrade_cost['StdErr']
        plt.fill_between(x4, y4 - ySEM, y4 + ySEM, color='r', alpha=0.2)
        ySEM = zscore * dfagg_1stTrade_cost_1_3['StdErr']
        plt.fill_between(x2a, y2a - ySEM, y2a + ySEM, color='orange', alpha=0.2)
        ySEM = zscore * dfagg_1stTrade_cost_4_77['StdErr']
        plt.fill_between(x2b, y2b - ySEM, y2b + ySEM, color='cyan', alpha=0.2)

    plt.title('Cost x Size (%s)'%mylabel, fontsize=myfontsize)
    plt.ylabel('normEmpCost', fontsize=myfontsize-2)
    plt.xlabel(sizeLabel, fontsize=myfontsize-2)
    plt.legend()

    ## Return
    plt.subplot(1, 2, 2)
    y1 = dfagg_all['normAdjSignedReturn']
    y3 = dfagg_very1stTrade['normAdjSignedReturn']
    y4 = dfagg_not1stTrade['normAdjSignedReturn']
    y2a = dfagg_1stTrade_1_3['normAdjSignedReturn']
    y2b = dfagg_1stTrade_4_77['normAdjSignedReturn']

    plt.plot(x1, y1, label='All-trades', linewidth=3)#, plt.scatter(x,y1)
    plt.plot(x3, y3, label='Very 1st-trades dur=[0,0]', linewidth=3, color='g')
    plt.plot(x4, y4, label='Not 1st-trades dur=[0,0]', linewidth=3, color='r')
    plt.plot(x2a, y2a, label='1st-trades (exclusive) dur=%s ** \n'%dur0label, linewidth=3, color='orange')
    plt.plot(x2b, y2b, label='1st-trades (exclusive) dur=%s ** \n'%dur1label, linewidth=3, color='cyan')

    if includeSEM:
        ySEM = zscore * dfagg_all['StdErr']
        plt.fill_between(x1, y1 - ySEM, y1 + ySEM, color='b', alpha=0.2)
        ySEM = zscore * dfagg_very1stTrade['StdErr']
        plt.fill_between(x3, y3 - ySEM, y3 + ySEM, color='g', alpha=0.2)
        ySEM = zscore * dfagg_not1stTrade['StdErr']
        plt.fill_between(x4, y4 - ySEM, y4 + ySEM, color='r', alpha=0.2)
        ySEM = zscore * dfagg_1stTrade_1_3['StdErr']
        plt.fill_between(x2a, y2a - ySEM, y2a + ySEM, color='orange', alpha=0.2)
        ySEM = zscore * dfagg_1stTrade_4_77['StdErr']
        plt.fill_between(x2b, y2b - ySEM, y2b + ySEM, color='cyan', alpha=0.2)

    plt.title('Return x Size (%s)'%mylabel, fontsize=myfontsize)
    plt.ylabel('normReturn (adjusted)', fontsize=myfontsize - 2)
    plt.xlabel(sizeLabel, fontsize=myfontsize-2)
    plt.legend()


def helper_plot_Cost_Return_bySize_alldur(df2gox,sizeVar,sizeLabel, mylabel, quantiles = [0.3, 0.6, 0.9, 0.97]):
    # Defaults
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    myfontsize = 16
    #quantiles = [0.3, 0.6, 0.9, 0.97]
    zscore = 1.96  # CI: 95%
    # zscore = 1.645 # CI: 90%
    includeSEM = True

    ################################## ajdSizeShare ##########################################################
    targetVar = 'normAdjSignedReturn'
    targetVarLabel = 'normAdjSignReturn'
#    sizeVar = 'adjSizeShare'
#    sizeLabel = 'size (nShare/histADV)'
    ## 1) Split size by quantile
    SizeThr = "0"
    for q in quantiles:
        qThr = round(df2gox[sizeVar].quantile(q), 4)
        SizeThr = SizeThr + ":" + str(qThr)
    SizeThr = SizeThr + ":1000000000"
    SizeThr = np.fromstring(SizeThr, dtype=float, sep=":")

    ## 2) Aggregate by size
    ColAgg = ['normEmpCost', 'normSignedReturn', 'normAdjSignedReturn', 'prate', 'adjSizeShare']

    # Cost (to get the StdErr as well, not only the average (StdErr is computed on the col2filter)
    dfagg_all_cost = CostCurve.CostCurve(df2gox.copy(), None, SizeThr, ColAgg, sizeVar, 'normEmpCost', sizeVar, 0, 0.01,PrintNA=False)
    dfagg_very1stTrade_cost = CostCurve.CostCurve(df2gox[(df2gox.very1stTrade == 1)].copy(), None, SizeThr, ColAgg, sizeVar,
                                                  'normEmpCost', sizeVar, 0, 0.01,PrintNA=False)
    dfagg_not1stTrade_cost = CostCurve.CostCurve(df2gox[(df2gox.firstTrade != 1)].copy(),
                                             None, SizeThr, ColAgg, sizeVar, 'normEmpCost', sizeVar, 0, 0.01,
                                             PrintNA=False)
    dfagg_1stTrade_cost = CostCurve.CostCurve(df2gox[(df2gox.firstTrade == 1) & (df2gox.very1stTrade != 1)].copy(),
                                          None, SizeThr, ColAgg, sizeVar, 'normEmpCost', sizeVar, 0, 0.01,
                                          PrintNA=False)

    print('- Filtering by %s...' % 'normEmpCost')

    # Return
    dfagg_all = CostCurve.CostCurve(df2gox.copy(), None, SizeThr, ColAgg, sizeVar, targetVar, sizeVar, 0, 0.01,
                                    PrintNA=False)
    dfagg_very1stTrade = CostCurve.CostCurve(df2gox[df2gox.very1stTrade == 1].copy(), None, SizeThr, ColAgg, sizeVar,
                                             targetVar, sizeVar, 0, 0.01, PrintNA=False)
    dfagg_not1stTrade = CostCurve.CostCurve(df2gox[(df2gox.firstTrade != 1)].copy(),
                                            None, SizeThr, ColAgg, sizeVar, targetVar, sizeVar, 0, 0.01,
                                             PrintNA=False)
    dfagg_1stTrade = CostCurve.CostCurve(df2gox[(df2gox.firstTrade == 1) & (df2gox.very1stTrade != 1)].copy(), None,
                                         SizeThr, ColAgg, sizeVar,
                                         targetVar, sizeVar, 0, 0.01, PrintNA=False)

    print('- Filtering by %s...' % targetVar)
    cols2print = ['normEmpCost','normAdjSignedReturn','StdErr',sizeVar,'Count']
    print('** Aggregate all-trades ** \n',dfagg_all[cols2print])
    print('** Aggregate Very1st-trades ** \n', dfagg_very1stTrade[cols2print])
    print('** Aggregate Not1st-trades ** \n', dfagg_not1stTrade[cols2print])
    print('** Aggregate 1st-trades ** \n', dfagg_1stTrade[cols2print])

    # Plot
    fig = plt.figure(figsize=(15, 4))

    ## Cost
    x = dfagg_all[sizeVar]
    plt.subplot(1, 2, 1)

    y1 = dfagg_all['normEmpCost']
    y3 = dfagg_very1stTrade['normEmpCost']
    y4 = dfagg_not1stTrade['normEmpCost']
    y2 = dfagg_1stTrade['normEmpCost']

    plt.plot(x, y1, label='All-trades', linewidth=3) #, plt.scatter(x,y1)
    plt.plot(x, y3, label='Very 1st-trades', linewidth=3, color='g') #, plt.scatter(x,y3)
    plt.plot(x, y4, label='Not 1st-trades', linewidth=3,color='r')
    plt.plot(x, y2, label='1st-trades (exclusive)', linewidth=3, color='orange')  # , plt.scatter(x,y2)

    if includeSEM:
        ySEM = zscore * dfagg_all_cost['StdErr']
        plt.fill_between(x, y1 - ySEM, y1 + ySEM, color='b', alpha=0.2)
        ySEM = zscore * dfagg_very1stTrade_cost['StdErr']
        plt.fill_between(x, y3 - ySEM, y3 + ySEM, color='g', alpha=0.2)
        ySEM = zscore * dfagg_not1stTrade_cost['StdErr']
        plt.fill_between(x, y4 - ySEM, y4 + ySEM, color='r', alpha=0.2)
        ySEM = zscore * dfagg_1stTrade_cost['StdErr']
        plt.fill_between(x, y2 - ySEM, y2 + ySEM, color='orange', alpha=0.2)

    plt.title('Cost x Size (%s)'%mylabel, fontsize=myfontsize)
    plt.ylabel('normEmpCost', fontsize=myfontsize-2)
    plt.xlabel(sizeLabel, fontsize=myfontsize-2)
    plt.legend()

    ## Return
    plt.subplot(1, 2, 2)
    y1 = dfagg_all['normAdjSignedReturn']
    y3 = dfagg_very1stTrade['normAdjSignedReturn']
    y4 = dfagg_not1stTrade['normAdjSignedReturn']
    y2 = dfagg_1stTrade['normAdjSignedReturn']

    plt.plot(x, y1, label='All-trades', linewidth=3)#, plt.scatter(x,y1)
    plt.plot(x, y3, label='Very 1st-trades', linewidth=3, color='g')
    plt.plot(x, y4, label='Not 1st-trades', linewidth=3, color='r')
    plt.plot(x, y2, label='1st-trades (exclusive)', linewidth=3, color='orange')

    if includeSEM:
        ySEM = zscore * dfagg_all['StdErr']
        plt.fill_between(x, y1 - ySEM, y1 + ySEM, color='b', alpha=0.2)
        ySEM = zscore * dfagg_very1stTrade['StdErr']
        plt.fill_between(x, y3 - ySEM, y3 + ySEM, color='g', alpha=0.2)
        ySEM = zscore * dfagg_not1stTrade['StdErr']
        plt.fill_between(x, y4 - ySEM, y4 + ySEM, color='r', alpha=0.2)
        ySEM = zscore * dfagg_1stTrade['StdErr']
        plt.fill_between(x, y2 - ySEM, y2 + ySEM, color='orange', alpha=0.2)

    plt.title('Return x Size (%s)'%mylabel, fontsize=myfontsize)
    plt.ylabel('normReturn (adjusted)', fontsize=myfontsize - 2)
    plt.xlabel(sizeLabel, fontsize=myfontsize-2)
    plt.legend()

def helper_plot_Cost_Return_bySize(df2gox,sizeVar,sizeLabel, mylabel, quantiles = [0.3, 0.6, 0.9, 0.97], dur=None):
    # Defaults
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    myfontsize = 16
    #quantiles = [0.3, 0.6, 0.9, 0.97]
    zscore = 1.96  # CI: 95%
    # zscore = 1.645 # CI: 90%
    includeSEM = True

    ################################## ajdSizeShare ##########################################################
    targetVar = 'normAdjSignedReturn'
    targetVarLabel = 'normAdjSignReturn'
#    sizeVar = 'adjSizeShare'
#    sizeLabel = 'size (nShare/histADV)'
    ## 1) Split size by quantile
    SizeThr = "0"
    for q in quantiles:
        qThr = round(df2gox[sizeVar].quantile(q), 4)
        SizeThr = SizeThr + ":" + str(qThr)
    SizeThr = SizeThr + ":1000000000"
    SizeThr = np.fromstring(SizeThr, dtype=float, sep=":")

    ## 2) Aggregate by size
    #ColAgg = ['normEmpCost', 'normSignedReturn', 'normAdjSignedReturn', 'prate', 'adjSizeShare']
    ColAgg = ['normEmpCost', 'normSignedReturn', 'normAdjSignedReturn', 'prate', 'adjSizeShare']+[sizeVar]
    # Cost (to get the StdErr as well, not only the average (StdErr is computed on the col2filter)
    dfagg_all_cost = CostCurve.CostCurve(df2gox.copy(), None, SizeThr, ColAgg, sizeVar, 'normEmpCost', sizeVar, 0, 0.01,PrintNA=False)
    if dur == [0, 0]:
        dfagg_very1stTrade_cost = CostCurve.CostCurve(df2gox[(df2gox.very1stTrade == 1)].copy(), None, SizeThr, ColAgg, sizeVar,
                                                      'normEmpCost', sizeVar, 0, 0.01,PrintNA=False)
        dfagg_not1stTrade_cost = CostCurve.CostCurve(df2gox[(df2gox.firstTrade != 1)].copy(),
                                                 None, SizeThr, ColAgg, sizeVar, 'normEmpCost', sizeVar, 0, 0.01,
                                                 PrintNA=False)
    else:
        dfagg_1stTrade_cost = CostCurve.CostCurve(df2gox[(df2gox.firstTrade == 1) & (df2gox.very1stTrade != 1)].copy(),
                                              None, SizeThr, ColAgg, sizeVar, 'normEmpCost', sizeVar, 0, 0.01,
                                              PrintNA=False)

    print('- Filtering by %s...' % 'normEmpCost')

    # Return
    dfagg_all = CostCurve.CostCurve(df2gox.copy(), None, SizeThr, ColAgg, sizeVar, targetVar, sizeVar, 0, 0.01,
                                    PrintNA=False)
    if dur == [0, 0]:
        dfagg_very1stTrade = CostCurve.CostCurve(df2gox[df2gox.very1stTrade == 1].copy(), None, SizeThr, ColAgg, sizeVar,
                                                 targetVar, sizeVar, 0, 0.01, PrintNA=False)
        dfagg_not1stTrade = CostCurve.CostCurve(df2gox[(df2gox.firstTrade != 1)].copy(),
                                                None, SizeThr, ColAgg, sizeVar, targetVar, sizeVar, 0, 0.01,
                                                 PrintNA=False)
    else:
        dfagg_1stTrade = CostCurve.CostCurve(df2gox[(df2gox.firstTrade == 1) & (df2gox.very1stTrade != 1)].copy(), None,
                                             SizeThr, ColAgg, sizeVar,
                                             targetVar, sizeVar, 0, 0.01, PrintNA=False)

    print('- Filtering by %s...' % targetVar)
    cols2print = ['normEmpCost','normAdjSignedReturn','StdErr',sizeVar,'Count']
    print('** Aggregate all-trades ** \n',dfagg_all[cols2print])
    if dur == [0, 0]:
        print('** Aggregate Very1st-trades ** \n', dfagg_very1stTrade[cols2print])
        print('** Aggregate Not1st-trades ** \n', dfagg_not1stTrade[cols2print])
    else:
        print('** Aggregate 1st-trades ** \n', dfagg_1stTrade[cols2print])

    # Plot
    fig = plt.figure(figsize=(15, 4))

    ## Cost
    x = dfagg_all[sizeVar]
    plt.subplot(1, 2, 1)

    y1 = dfagg_all['normEmpCost']
    if dur == [0, 0]:
        y3 = dfagg_very1stTrade['normEmpCost']
        y4 = dfagg_not1stTrade['normEmpCost']
    else:
        y2 = dfagg_1stTrade['normEmpCost']

    plt.plot(x, y1, label='All-trades', linewidth=3) #, plt.scatter(x,y1)
    if dur == [0, 0]:
        plt.plot(x, y3, label='Very 1st-trades', linewidth=3, color='g') #, plt.scatter(x,y3)
        plt.plot(x, y4, label='Not 1st-trades', linewidth=3,color='r')
    else:
        plt.plot(x, y2, label='1st-trades (exclusive)', linewidth=3, color='orange')  # , plt.scatter(x,y2)

    if includeSEM:
        ySEM = zscore * dfagg_all_cost['StdErr']
        plt.fill_between(x, y1 - ySEM, y1 + ySEM, color='b', alpha=0.2)
        if dur == [0, 0]:
            ySEM = zscore * dfagg_very1stTrade_cost['StdErr']
            plt.fill_between(x, y3 - ySEM, y3 + ySEM, color='g', alpha=0.2)
            ySEM = zscore * dfagg_not1stTrade_cost['StdErr']
            plt.fill_between(x, y4 - ySEM, y4 + ySEM, color='r', alpha=0.2)
        else:
            ySEM = zscore * dfagg_1stTrade_cost['StdErr']
            plt.fill_between(x, y2 - ySEM, y2 + ySEM, color='orange', alpha=0.2)

    plt.title('Cost x Size (%s)'%mylabel, fontsize=myfontsize)
    plt.ylabel('normEmpCost', fontsize=myfontsize-2)
    plt.xlabel(sizeLabel, fontsize=myfontsize-2)
    plt.legend()

    ## Return
    plt.subplot(1, 2, 2)
    y1 = dfagg_all['normAdjSignedReturn']
    if dur == [0, 0]:
        y3 = dfagg_very1stTrade['normAdjSignedReturn']
        y4 = dfagg_not1stTrade['normAdjSignedReturn']
    else:
        y2 = dfagg_1stTrade['normAdjSignedReturn']

    plt.plot(x, y1, label='All-trades', linewidth=3)#, plt.scatter(x,y1)
    if dur == [0, 0]:
        plt.plot(x, y3, label='Very 1st-trades', linewidth=3, color='g')
        plt.plot(x, y4, label='Not 1st-trades', linewidth=3, color='r')
    else:
        plt.plot(x, y2, label='1st-trades (exclusive)', linewidth=3, color='orange')

    if includeSEM:
        ySEM = zscore * dfagg_all['StdErr']
        plt.fill_between(x, y1 - ySEM, y1 + ySEM, color='b', alpha=0.2)
        if dur == [0, 0]:
            ySEM = zscore * dfagg_very1stTrade['StdErr']
            plt.fill_between(x, y3 - ySEM, y3 + ySEM, color='g', alpha=0.2)
            ySEM = zscore * dfagg_not1stTrade['StdErr']
            plt.fill_between(x, y4 - ySEM, y4 + ySEM, color='r', alpha=0.2)
        else:
            ySEM = zscore * dfagg_1stTrade['StdErr']
            plt.fill_between(x, y2 - ySEM, y2 + ySEM, color='orange', alpha=0.2)

    plt.title('Return x Size (%s)'%mylabel, fontsize=myfontsize)
    plt.ylabel('normReturn (adjusted)', fontsize=myfontsize - 2)
    plt.xlabel(sizeLabel, fontsize=myfontsize-2)
    plt.legend()

def helper_plot_Y_byX(df2gox,yVar, yLabel, sizeVar,sizeLabel, mylabel, quantiles = [0.3, 0.6, 0.9, 0.97]):
    """Plot Y versus a X variable specified by use"""
    # Defaults
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    myfontsize = 16
    #quantiles = [0.3, 0.6, 0.9, 0.97]
    zscore = 1.96  # CI: 95%
    # zscore = 1.645 # CI: 90%
    includeSEM = True


    targetVar = yVar
    targetVarLabel = yLabel

    # Check null values in the sizeVar
    if np.sum(df2gox[sizeVar].isnull()) > 0:
        print('(*) Removing null %s (n=%d)...'%(sizeVar,np.sum(df2gox[sizeVar].isnull())))
        df2gox = df2gox[~df2gox[sizeVar].isnull()]

    ## 1) Split size by quantile
    SizeThr = str(df2gox[sizeVar].min()-1)
    #SizeThr = "0"
    for q in quantiles:
        qThr = round(df2gox[sizeVar].quantile(q), 4)
        SizeThr = SizeThr + ":" + str(qThr)
    SizeThr = SizeThr + ":"+str(round(df2gox[sizeVar].max()+1))
    #SizeThr = SizeThr + ":100000"
    SizeThr = np.fromstring(SizeThr, dtype=float, sep=":")
    print('- Bucket thresholds:',SizeThr)
    ## 2) Aggregate by size
    #ColAgg = ['normEmpCost', 'normSignedReturn', 'normAdjSignedReturn', 'prate', 'adjSizeShare']
    ColAgg = ['Count']
    if sizeVar not in ColAgg:
        ColAgg += [sizeVar]
    if targetVar not in ColAgg:
        ColAgg += [targetVar]
    # Cost (to get the StdErr as well, not only the average (StdErr is computed on the col2filter)
    dfagg_all = CostCurve.CostCurve(df2gox.copy(), None, SizeThr, ColAgg, sizeVar, targetVar, sizeVar, 0, 0.01,PrintNA=False)
    print('- Filtering by %s...'%targetVar)
    print('- Aggregating by %s (min=%2.2f, max=%2.2f)...' %(sizeVar,df2gox[sizeVar].min(),df2gox[sizeVar].max()))

    cols2print = ColAgg
    print('** Aggregate all-trades ** \n',dfagg_all[cols2print])

    # Plot
    fig = plt.figure(figsize=(12, 6))

    ## Cost
    x = dfagg_all[sizeVar]
    y1 = dfagg_all[targetVar]
    plt.plot(x, y1, linewidth=3) #, plt.scatter(x,y1)

    if includeSEM:
        ySEM = zscore * dfagg_all['StdErr']
        plt.fill_between(x, y1 - ySEM, y1 + ySEM, color='b', alpha=0.2)

    plt.title('%s x %s (%s)'%(yLabel,sizeLabel,mylabel), fontsize=myfontsize)
    plt.ylabel(yLabel, fontsize=myfontsize-2)
    plt.xlabel(sizeLabel, fontsize=myfontsize-2)
#    plt.legend()


def helper_plot_Return_bySize_byGapDist_withCondition(df2gox,sizeVar,sizeLabel, condVar, condLabel, condrange, mylabel, lastBin=78, quantiles = [0.3, 0.6, 0.9, 0.97], savepath = './', savelabel='', pdfobj=None):
    """Plot Cost and Return versus a X variable specified by use"""
    # Defaults

    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    myfontsize = 16
    #quantiles = [0.3, 0.6, 0.9, 0.97]
    zscore = 1.96  # CI: 95%
    # zscore = 1.645 # CI: 90%
    includeSEM = True

    # Check null values in the sizeVar
    if np.sum(df2gox[sizeVar].isnull()) > 0:
        print('(*) Removing null %s (n=%d)...' % (sizeVar, np.sum(df2gox[sizeVar].isnull())))
        df2gox = df2gox[~df2gox[sizeVar].isnull()]
    # Check null values in the sizeVar
    if np.sum(df2gox[condVar].isnull()) > 0:
        print('(*) Removing null %s (n=%d)...' % (condVar, np.sum(df2gox[condVar].isnull())))
        df2gox = df2gox[~df2gox[condVar].isnull()]

    # Check null values in the Returns
    vars2check = ['normSignedReturn', 'normAdjSignedReturn','surVola','pred30min','sizeLastTraded','gapDistance']
    for xvar in vars2check:
        if (np.sum(df2gox[xvar].isnull()) > 0):
            # xvar = 'normAdjSignedReturn'
            print('(*) Removing null %s (n=%d)...' % (xvar, np.sum(df2gox[xvar].isnull())))
            df2gox = df2gox[~df2gox[xvar].isnull()]

    ################################## ajdSizeShare ##########################################################
    targetVar = 'normSignedReturn'
    targetVarLabel = 'normSignReturn'
#    sizeVar = 'adjSizeShare'
#    sizeLabel = 'size (nShare/histADV)'

    # Get condition ranges
    print('Condrange:',condrange)
    #print(df2gox[condVar].unique())
    condranges = jifin.get_condranges(df2gox[condVar].copy(),condrange)

    # Plot
    fig = plt.figure(figsize=(15, 10))
    # For the return bar
    dfAvgReturn = pd.DataFrame()

    for i, condrange in enumerate(condranges):
        imask = df2gox[condVar].between(condrange[0],condrange[1])
        print('\n--- Number of clusters for range [%2.2f, %2.2f]: %d'%(condrange[0],condrange[1],np.sum(imask)))

        # 0) Return average (use a single bucket)
        sizeVarDist = 'gapDistance'
        ColAgg0 = ['normSignedReturn', 'gapDistance']
        dfagg_return = CostCurve.CostCurve(df2gox[imask].copy(), None, [0,lastBin], ColAgg0, sizeVarDist, targetVar, sizeVarDist, 0,0.01, PrintNA=False)
        print('- Return average: \n', dfagg_return[['normSignedReturn','StdErr','Count']])
        dfAvgReturn = dfAvgReturn.append(dfagg_return[['normSignedReturn','StdErr','Count']])

        ## 1) Split size by quantile
        #SizeThr = "0"
        SizeThr = str(df2gox[sizeVar].min() - 10)
        for q in quantiles:
            qThr = round(df2gox.loc[imask,sizeVar].quantile(q), 4)
            SizeThr = SizeThr + ":" + str(qThr)
        #SizeThr = SizeThr + ":1000000000"
        SizeThr = SizeThr + ":" + str(round(df2gox[sizeVar].max() + 10))
        SizeThr = np.fromstring(SizeThr, dtype=float, sep=":")
        #print('- SizeThr:',SizeThr)

        ## 2) Aggregate by size
        ColAgg = ['normSignedReturn', 'normAdjSignedReturn', 'gapDistance']
        if sizeVar not in ColAgg:
            ColAgg += [sizeVar]

        # Return by gapDistance
        sizeVarDist = 'gapDistance'
        if lastBin ==13:
            SizeThrDist = [0, 1, 2, 3, 4, 5, 13] # 30min bin
        if lastBin ==78:
            SizeThrDist = [0, 1, 2, 6, 10, 20, 40, 78] # 5min bin: 5,10,30,60, 120, 240
        dfagg_all_dist = CostCurve.CostCurve(df2gox[imask].copy(), None, SizeThrDist, ColAgg, sizeVarDist, targetVar, sizeVarDist, 0,0.01, PrintNA=False)
        dfagg_all_dist.to_csv(os.path.join(savepath, 'table_Return_ByGapDist_%s_range%d.csv' % (savelabel.split('Size_')[1], i)))
        print('- Filtering by %s...' % targetVar)

        # Return by Size
        dfagg_all = CostCurve.CostCurve(df2gox[imask].copy(), None, SizeThr, ColAgg, sizeVar, targetVar, sizeVar, 0, 0.01,PrintNA=False)
        dfagg_all.to_csv(os.path.join(savepath,'table_Return_%s_range%d.csv'%(savelabel,i)))
        print('- Filtering by %s...' % targetVar)

        cols2print = ['normSignedReturn','StdErr',sizeVar,'gapDistance','Count']
        print('- Aggregate table (by sizeShare): \n',dfagg_all[cols2print])
        print('- Aggregate table (by gapDistance): \n', dfagg_all_dist[cols2print])

        ## Return by gapDistance (duration)
        x = dfagg_all_dist[sizeVarDist]
        plt.subplot(2, 2, 1)
        y1 = dfagg_all_dist['normSignedReturn']

        plt.plot(x, y1, label='%s:[%1.2f,%1.2f]'%(condLabel,condrange[0],condrange[1]), linewidth=3)#, plt.scatter(x,y1)

        if includeSEM:
            ySEM = zscore * dfagg_all_dist['StdErr']
            plt.fill_between(x, y1 - ySEM, y1 + ySEM, alpha=0.2)

        plt.title('Returnx%s(%s)'%('gapDistance',mylabel), fontsize=myfontsize)
        plt.ylabel('normReturn', fontsize=myfontsize - 2)
        plt.xlabel('gapDistance (bin)', fontsize=myfontsize-2)
        plt.legend()


        ## Return
        x = dfagg_all[sizeVar]
        plt.subplot(2, 2, 2)
        y1 = dfagg_all['normSignedReturn']

        plt.plot(x, y1, label='%s:[%1.2f,%1.2f]'%(condLabel,condrange[0],condrange[1]), linewidth=3)#, plt.scatter(x,y1)

        if includeSEM:
            ySEM = zscore * dfagg_all['StdErr']
            plt.fill_between(x, y1 - ySEM, y1 + ySEM, alpha=0.2)

        plt.title('Returnx%s(%s)'%(sizeLabel,mylabel), fontsize=myfontsize)
        plt.ylabel('normReturn', fontsize=myfontsize - 2)
        plt.xlabel(sizeLabel, fontsize=myfontsize-2)
        plt.legend()

    dfAvgReturn['volaRanges'] = condranges
    dfAvgReturn.to_csv(os.path.join(savepath, 'table_AvgReturn_%s.csv'% (savelabel.split('Size_')[1])))

    print('-- Average Return --\n',dfAvgReturn)

    ## Plot Bar with Avg Returns
    means = dfAvgReturn.normSignedReturn.values  # Mean Data
    stds = dfAvgReturn.StdErr.values  # Standard deviation Data
    peakval = ['%2.3f'%i for i in means]  # String array of means
    myticks = ['[%1.2f,%1.2f]'%(condrange[0], condrange[1]) for condrange in condranges ]

    ind = np.arange(len(means))
    width = 0.35
    #colours = ['red', 'blue', 'green', 'yellow']

    plt.subplot(2, 2, 3)
    plt.title('Average Return', size=20)
    for i in range(len(means)):
        # plt.bar(ind[i],means[i],width,color=colours[i],align='center',yerr=stds[i],ecolor='k')
        plt.bar(ind[i], means[i], width, align='center', yerr=stds[i], # , color=colours[i]
                error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=2))

    plt.ylabel('normSignedReturn',size=16)
    plt.xlabel('%s'%condLabel, size=16)
    plt.xticks(ind, myticks)

    def autolabel(bars, peakval):
        for ii, bar in enumerate(bars):
            height = bars[ii]
            plt.text(ind[ii] + 0.3, height, '%s' % (peakval[ii]), ha='center', va='bottom', size=16)

    autolabel(means, peakval)
    plt.tight_layout()
    plt.show()

    if pdfobj is not None:
        pdfobj.savefig(fig)
    #plt.legend()


def helper_plot_Cost_Return_byX_withCondition(df2gox,sizeVar,sizeLabel, condVar, condLabel, condrange, mylabel, quantiles = [0.3, 0.6, 0.9, 0.97], savepath = './', savelabel='', pdfobj=None):
    """Plot Cost and Return versus a X variable specified by use"""
    # Defaults

    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    myfontsize = 16
    #quantiles = [0.3, 0.6, 0.9, 0.97]
    zscore = 1.96  # CI: 95%
    # zscore = 1.645 # CI: 90%
    includeSEM = True

    # Check null values in the sizeVar
    if np.sum(df2gox[sizeVar].isnull()) > 0:
        print('(*) Removing null %s (n=%d)...' % (sizeVar, np.sum(df2gox[sizeVar].isnull())))
        df2gox = df2gox[~df2gox[sizeVar].isnull()]
    # Check null values in the sizeVar
    if np.sum(df2gox[condVar].isnull()) > 0:
        print('(*) Removing null %s (n=%d)...' % (condVar, np.sum(df2gox[condVar].isnull())))
        df2gox = df2gox[~df2gox[condVar].isnull()]
    # Check null values in the Returns
    vars2check = ['normEmpCost', 'normSignedReturn', 'normAdjSignedReturn', 'prate', 'adjSizeShare']
    for xvar in vars2check:
        if (np.sum(df2gox[xvar].isnull()) > 0):
            # xvar = 'normAdjSignedReturn'
            print('(*) Removing null %s (n=%d)...' % (xvar, np.sum(df2gox[xvar].isnull())))
            df2gox = df2gox[~df2gox[xvar].isnull()]

    ################################## ajdSizeShare ##########################################################
    targetVar = 'normAdjSignedReturn'
    targetVarLabel = 'normAdjSignReturn'
#    sizeVar = 'adjSizeShare'
#    sizeLabel = 'size (nShare/histADV)'

    # Get condition ranges
    print('Condrange:',condrange)
    print('df2gox shape:', df2gox.shape)
    #print(df2gox[condVar].head())
    condranges = jifin.get_condranges(df2gox[condVar],condrange)

    # Plot
    fig = plt.figure(figsize=(15, 4))

    for i, condrange in enumerate(condranges):
        imask = df2gox[condVar].between(condrange[0],condrange[1])
        print('--- Number of clusters for range [%2.2f, %2.2f]: %d'%(condrange[0],condrange[1],np.sum(imask)))

        ## 1) Split size by quantile
        #SizeThr = "0"
        SizeThr = str(df2gox[sizeVar].min() - 10)
        for q in quantiles:
            qThr = round(df2gox.loc[imask,sizeVar].quantile(q), 4)
            SizeThr = SizeThr + ":" + str(qThr)
        #SizeThr = SizeThr + ":1000000000"
        SizeThr = SizeThr + ":" + str(round(df2gox[sizeVar].max() + 10))
        SizeThr = np.fromstring(SizeThr, dtype=float, sep=":")

        ## 2) Aggregate by size
        ColAgg = ['normEmpCost', 'normSignedReturn', 'normAdjSignedReturn', 'prate', 'adjSizeShare']
        if sizeVar not in ColAgg:
            ColAgg += [sizeVar]
        # Cost (to get the StdErr as well, not only the average (StdErr is computed on the col2filter)
        dfagg_all_cost = CostCurve.CostCurve(df2gox[imask].copy(), None, SizeThr, ColAgg, sizeVar, 'normEmpCost', sizeVar, 0, 0.01,PrintNA=False)
        dfagg_all_cost.to_csv(os.path.join(savepath,'table_Cost_%s_range%d.csv'%(savelabel,i)))
        print('- Filtering by %s...' % 'normEmpCost')

        # Return
        dfagg_all = CostCurve.CostCurve(df2gox[imask].copy(), None, SizeThr, ColAgg, sizeVar, targetVar, sizeVar, 0, 0.01,PrintNA=False)
        dfagg_all.to_csv(os.path.join(savepath,'table_Return_%s_range%d.csv'%(savelabel,i)))
        print('- Filtering by %s...' % targetVar)

        cols2print = ['normEmpCost','normAdjSignedReturn','StdErr',sizeVar,'Count']
        print('- Aggregate table: \n',dfagg_all[cols2print])

        ## Cost
        x = dfagg_all[sizeVar]
        plt.subplot(1, 2, 1)

        y1 = dfagg_all['normEmpCost']

        plt.plot(x, y1, label='%s:[%1.2f,%1.2f]'%(condLabel,condrange[0],condrange[1]), linewidth=3) #, plt.scatter(x,y1)

        if includeSEM:
            ySEM = zscore * dfagg_all_cost['StdErr']
            plt.fill_between(x, y1 - ySEM, y1 + ySEM, alpha=0.2)

        plt.title('Costx%s(%s)'%(sizeLabel,mylabel), fontsize=myfontsize)
        plt.ylabel('normEmpCost', fontsize=myfontsize-2)
        plt.xlabel(sizeLabel, fontsize=myfontsize-2)
        plt.legend()

        ## Return
        plt.subplot(1, 2, 2)
        y1 = dfagg_all['normAdjSignedReturn']

        plt.plot(x, y1, label='%s:[%1.2f,%1.2f]'%(condLabel,condrange[0],condrange[1]), linewidth=3)#, plt.scatter(x,y1)

        if includeSEM:
            ySEM = zscore * dfagg_all['StdErr']
            plt.fill_between(x, y1 - ySEM, y1 + ySEM, alpha=0.2)

        plt.title('Returnx%s(%s)'%(sizeLabel,mylabel), fontsize=myfontsize)
        plt.ylabel('normReturn (adjusted)', fontsize=myfontsize - 2)
        plt.xlabel(sizeLabel, fontsize=myfontsize-2)
        plt.legend()
    plt.tight_layout()

    if pdfobj is not None:
        pdfobj.savefig(fig)
    #plt.legend()

def helper_plot_Cost_Return_byX(df2gox,sizeVar,sizeLabel, mylabel, quantiles = [0.3, 0.6, 0.9, 0.97]):
    """Plot Cost and Return versus a X variable specified by use"""
    # Defaults

    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    myfontsize = 16
    #quantiles = [0.3, 0.6, 0.9, 0.97]
    zscore = 1.96  # CI: 95%
    # zscore = 1.645 # CI: 90%
    includeSEM = True

    # Check null values in the sizeVar
    if np.sum(df2gox[sizeVar].isnull()) > 0:
        print('(*) Removing null %s (n=%d)...' % (sizeVar, np.sum(df2gox[sizeVar].isnull())))
        df2gox = df2gox[~df2gox[sizeVar].isnull()]

    # Check null values in the Returns
    vars2check = ['normEmpCost', 'normSignedReturn', 'normAdjSignedReturn', 'prate', 'adjSizeShare']
    for xvar in vars2check:
        if (np.sum(df2gox[xvar].isnull())>0):
            #xvar = 'normAdjSignedReturn'
            print('(*) Removing null %s (n=%d)...' % (xvar, np.sum(df2gox[xvar].isnull())))
            df2gox = df2gox[~df2gox[xvar].isnull()]

    ################################## ajdSizeShare ##########################################################
    targetVar = 'normAdjSignedReturn'
    targetVarLabel = 'normAdjSignReturn'
#    sizeVar = 'adjSizeShare'
#    sizeLabel = 'size (nShare/histADV)'
    ## 1) Split size by quantile
    #SizeThr = "0"
    SizeThr = str(df2gox[sizeVar].min() - 10)
    for q in quantiles:
        qThr = round(df2gox[sizeVar].quantile(q), 4)
        SizeThr = SizeThr + ":" + str(qThr)
    SizeThr = SizeThr + ":1000000000"
    SizeThr = np.fromstring(SizeThr, dtype=float, sep=":")
    #print('- SizeThr:', SizeThr)
    ## 2) Aggregate by size
    ColAgg = ['normEmpCost', 'normSignedReturn', 'normAdjSignedReturn', 'prate', 'adjSizeShare']
    if sizeVar not in ColAgg:
        ColAgg += [sizeVar]
    print('- ColAgg:', ColAgg)
    # Cost (to get the StdErr as well, not only the average (StdErr is computed on the col2filter)
    dfagg_all_cost = CostCurve.CostCurve(df2gox.copy(), None, SizeThr, ColAgg, sizeVar, 'normEmpCost', sizeVar, 0, 0.01,PrintNA=False)
    print('- Filtering by %s...' % 'normEmpCost')

    # Return
    dfagg_all = CostCurve.CostCurve(df2gox.copy(), None, SizeThr, ColAgg, sizeVar, targetVar, sizeVar, 0, 0.01,PrintNA=False)
    print('- Filtering by %s...' % targetVar)

    cols2print = ['normEmpCost','normAdjSignedReturn','StdErr',sizeVar,'Count']
    print('** Aggregate all-trades ** \n',dfagg_all[cols2print])

    #print(dfagg_all)

    # Plot
    fig = plt.figure(figsize=(15, 4))

    ## Cost
    x = dfagg_all[sizeVar]
    plt.subplot(1, 2, 1)

    y1 = dfagg_all['normEmpCost']

    plt.plot(x, y1, label='All-trades', linewidth=3) #, plt.scatter(x,y1)

    if includeSEM:
        ySEM = zscore * dfagg_all_cost['StdErr']
        plt.fill_between(x, y1 - ySEM, y1 + ySEM, color='b', alpha=0.2)

    plt.title('Cost x %s (%s)'%(sizeLabel,mylabel), fontsize=myfontsize)
    plt.ylabel('normEmpCost', fontsize=myfontsize-2)
    plt.xlabel(sizeLabel, fontsize=myfontsize-2)
    plt.legend()

    ## Return
    plt.subplot(1, 2, 2)
    y1 = dfagg_all['normAdjSignedReturn']

    plt.plot(x, y1, label='All-trades', linewidth=3)#, plt.scatter(x,y1)

    if includeSEM:
        ySEM = zscore * dfagg_all['StdErr']
        plt.fill_between(x, y1 - ySEM, y1 + ySEM, color='b', alpha=0.2)

    plt.title('Return x %s (%s)'%(sizeLabel,mylabel), fontsize=myfontsize)
    plt.ylabel('normReturn (adjusted)', fontsize=myfontsize - 2)
    plt.xlabel(sizeLabel, fontsize=myfontsize-2)
    plt.legend()


def helper_plot_bars_Y_byBuckets(Y,sizex,allbins,condVar,size_buckets,daytime,nsplit,myflag):
    """Helper to plot Y x {size}
    Y: can be any variable. dtype: Series
    size: can be any variable. dtype: Series
    allbins: bin info of each row. dtype: Series
    condVar: condition variable. dtype: Series
    nsplit: number of splits of the condition var (equally divided)
    daytime: [bin_start,bin_end]. List with 2 numbers
    size_buckets: list with the ranges
    myflag: dictionary with flag values
    """
    print('* NOTE: Excluded Y equal to ZERO: %d (out %d)'%(np.sum(Y == 0),len(Y)))
    imask = (Y != 0)
    Y = Y[imask]
    size = sizex[imask]
    condVar = condVar[imask]
    allbins = allbins[imask]

    # If no size_backets is specified, only the number of buckets
    if type(size_buckets)==int:
        # Split condVar into bins (equally distributed)
        #out, bins = pd.qcut(size, q=size_buckets, labels=False, retbins=True)
        out, bins = pd.qcut(size, q=size_buckets, labels=False, retbins=True) # Exclude no Exec
        size_buckets = [[float('%2.6f'%bins[i]), float('%2.6f'%bins[i + 1])] for i in range(len(out.unique()))]

    flag = argparse.Namespace(**myflag)
    # Add daytime info to title (assume 78 bins of 5 min)
    # time.strftime("%H:%M:%S", time.gmtime(0))
    usStart = 34200 # 9:30 in seconds
    t0 = time.strftime("%H:%M", time.gmtime((daytime[0]-1)*5*60+usStart)) # 5min bins
    t1 = time.strftime("%H:%M", time.gmtime((daytime[1])*5*60+usStart))
    flag.title = flag.title+' (Time:%s-%s)'%(t0,t1)

    if type(nsplit)==int:
        # Split condVar into bins (equally distributed)
        #out,bins = pd.qcut(condVar,q=nsplit,labels=False,retbins= True)
        out, bins = pd.qcut(condVar, q=nsplit, labels=False, retbins=True) # exclude zeros of Y (no execution...)
        mybins = [[bins[i],bins[i+1]] for i in range(len(out.unique()))]
        ncond = nsplit
    else:
        mybins = nsplit
        ncond = len(nsplit)

    # holders
    yall = []
    ystd = []
    ntrades = []
    print('- Size buckets:',size_buckets)
    #print('out:',out.unique()[::-1])
    #print('mybins:',mybins)
    #for ccategory,cbin in zip(out.unique()[::-1],mybins):

    binlabels = []
    for i in range(1, ncond + 1):
        crange = mybins[i - 1]
        if type(crange[0]) != str:
            condmask = condVar.between(crange[0], crange[1])
        else:
            condmask = (condVar == crange)
            binlabels.append(crange)

        # Given a condVar category
        #print('- Condition:',i-1,cbin)
        print('- Condition:', i - 1, crange)
        #condmask = (out==ccategory)
        yallx = []
        ystdx = []
        ntradesx = []
        for i in size_buckets:
            #print('- Bucket:',i)
            imask = (size.between(i[0],i[1])) & (allbins.between(daytime[0],daytime[1])) & condmask
            #print(size[imask])
            ds = Y[imask]
            yallx.append(ds.mean())
            ystdx.append(ds.std())
            ntradesx.append(len(ds))
        #print('yall:',yallx)
        #print('ystd:',ystdx)
        #print('ntrades:',ntradesx)
        yall.append(yallx)
        ystd.append(ystdx)
        ntrades.append(ntradesx)

    # If conditional variable is numeric and a range is specified
    if not binlabels:
        binlabels = mybins

    others = matlablike()
    others.ntrades = ntrades
    others.xlabel2 = '#trades'
    ji.plot_bars_buckets_withTable(yall,ystd, size_buckets, binlabels,flag,others)

def plot_bars_buckets_withTable(yall,ystd, size_buckets, mybins,flag, others, pdfobj=None):
    # Optional
    ntrades = others.ntrades

    mytitle = '%s (condition: %s)' % (flag.title, flag.condlabel)

    fsize = 19
    # General settings
    bwidth = 0.25
    fig = plt.figure(figsize=(20, 10))
    plt.rc('font', size=fsize)
    mycolors = ["r", "g", "b", "y", "m", "peachpuff", "fuchsia"]  # Some of the colors

    # For the table
    columns = ['%s: %s-%s' % (flag.xlabel, i[0], i[1]) for i in size_buckets]
    rows = []
    rowcolors = []
    cell_text = []
    for cond, values in enumerate(yall):
        xx = np.arange(len(values))
        myxtick = ['%s:%s-%s' % (flag.xlabel, i[0], i[1]) for i in size_buckets]

        # If a range is given instead of strings
        if type(mybins[0]) != str:
            binlabel = '%2.4f-%2.4f' % (mybins[cond][0], mybins[cond][1])
        else:
            binlabel = mybins[cond]

        plt.bar(xx + (cond - 1) * bwidth, values, bwidth, alpha=0.8, color=mycolors[cond],
                label='%s#%d:%s' % (flag.condlabel, cond, binlabel))
        # plt.bar(xx + bwidth, values0, bwidth, label=mylabels.g2)
        #fig.autofmt_xdate()
        fig.autofmt_xdate(bottom=0.1, rotation=3, ha='right')


        # Prepare table
        # rows.append('%s#%d (Std)'%(flag.condlabel,cond))
        rows.append(others.xlabel2+'(n=%d)'%np.sum(ntrades[cond]))
        rows.append('%s#%d (Mean+-Std)' % (flag.condlabel, cond))
        # cell_text.append(['%2.4f' % (x) for x in ystd[cond]])
        cell_text.append(['%d' % (x) for x in ntrades[cond]])
        cell_text.append(['%2.4f(+-%2.4f)' % (x, y) for x, y in zip(values, ystd[cond])])
        rowcolors.append(mycolors[cond])
        rowcolors.append(mycolors[cond])

    ## INCLUDE TABLE

    # Reverse colors and text labels to display the last value at the top.
    rows = rows[::-1]
    rowcolors = rowcolors[::-1]
    cell_text.reverse()
    nrows = len(rows)

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          fontsize=fsize,
                          rowLabels=rows,
                          rowColours=rowcolors,
                          colLabels=columns,
                          bbox=[0, -0.4-0.05*(nrows-6), 1, 0.3],
                          loc='bottom')

    ## change cell properties
    table_props = the_table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells:
        # cell.set_width(0.2)
        cell.set_height(0.03)
        cell.set_fontsize(fsize)
        cell.set_alpha(0.5)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0)
    # space2add = '\n'*round(1.25*len(rows))
    space2add = ''
    plt.xlabel('%s Size Buckets (%s)' % (space2add, flag.xlabel))
    plt.ylabel(flag.ylabel)
    plt.xticks(xx, myxtick)
    # lt.xticks([])

    plt.title(mytitle)
    plt.legend()
    plt.show()

    try:
        #flag.saveplot = False
        #savelabel = 'HLG'+flag.ylabel.split('HLG')[1]
        #now = datetime.datetime.now()
        #sname = 'fig_%s_by_%s_%s' % (flag.title, flag.condlabel, savelabel)
        #if flag.saveplot:
        #    fig.savefig('%s.png'%sname,bbox_inches='tight')
        #    print('Saved figure:',sname)
        if pdfobj is not None:
            plt.show()
            pdfobj.savefig(fig, bbox_inches="tight")
            #pdfobj.savefig(the_table)
            print('Figure saved: %s %s %s' % (flag.title, flag.condlabel))
    except:
        print('Did not save...')
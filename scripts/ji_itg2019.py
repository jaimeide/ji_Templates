"""
================================================================
     Copyright (c) 2019, VIRTU Financials Inc
                     All Rights Reserved
================================================================

 NAME : ji_virtu2019.py
 @DATE     Created: 11/21/18 (10:48 AM)
	       Modifications:
	       - 11/21/18: functions to access flat files
	       - 05/01/2019: couple of helpers and wrapper for the Volatility Prediction project

 @AUTHOR          : Jaime S Ide
                    jaime.ide@virtu.com or jaime.ide@yale.edu
===============================================================
"""

import logging
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import os
import csv
import glob
import subprocess

import argparse
import pyodbc #, subprocess
import ji_utils_fin2019 as jifin
import ji_utils_plots2019 as ji

import time
import argparse
import itertools
from scipy import stats
from pathlib import Path
from scipy.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

log = logging.getLogger()

class matlablike():
    pass

## Global variables (For production: convert to json
path_tickDataUSD15sec = '/local/data/itg/fe/prod/data1/jide-local/Data/tickData_USA_IntradayStats15Sec'

##############################################################################################
# ----------------------------------------- DYCE ----------------------------------------
##############################################################################################

''' -- Spliting the proccess --
id = 0
DyCEcmd = []

for (subplan, substrat) in zip(np.array_split(Plan,5),np.array_split(Strat,5)):
    fp = './plan'+str(id)+'.dat'
    fs = './strat'+str(id)+'.dat'
    subplan.to_csv(fp,header = False, index = False)
    substrat.to_csv(fs,header = False, index = False)
    cmd = f"/DyCE/bin/DyCE -p {fp} -S USERSTRAT -c -u {fs}  -o ./ --gzip &"
    print(cmd)
    DyCEcmd.append(cmd)
    id += 1

def CallProc(cmd):
    proc = subprocess.Popen(cmd, shell=True)
    out, err = proc.communicate()
    return(out, err)

with Pool(args.nProcs) as pool:
    results = pool.map(CallProc, DyCEcmd)
'''


def dyce_load_returns_file(fname, var2look):
    # myccys = ['EURUSD','AUDUSD','EURJPY','EURGBP','USDMXN','USDHKD']
    #     mydates = [20170522,20180430]
    #     mysizes = [20,200,1000,10000]

    myfont = 20
    # binstart = 12 * 8
    #     myrange = range(2,10)

    ## 1) LOAD
    #dfall = pd.DataFrame()
    dfxout = pd.read_csv(fname)
    #cols2include = ['token', 'binIndex', var2look]
    #df2look = dfxout.loc[dfxout.strategyName == strategy, cols2include]
    cols2include = ['token','strategyName', 'binIndex', var2look]
    df2look = dfxout[cols2include]

    # Stack into bins
    dfvals = df2look.set_index(['token','strategyName']).groupby(['token','strategyName'])[var2look].apply(
        lambda df: df.reset_index(drop=True)).unstack().reset_index()

    ## 2) Extract conditions from the token
    dfvals['ccy'] = dfvals.token.apply(lambda s: s[:6]).values
    dfvals['trdate'] = dfvals.token.apply(lambda s: int(s[7:15])).values
    dfvals['size'] = dfvals.token.apply(lambda s: int(s[30:])).values

    print('- File loaded:', var2look)
    return dfvals

def dyce_load_returns(ID, Xs, strategy, var2look):
        #     myccys = ['EURUSD','AUDUSD','EURJPY','EURGBP','USDMXN','USDHKD']
        #     mydates = [20170522,20180430]
        #     mysizes = [20,200,1000,10000]

    myfont = 20
    #binstart = 12 * 8
    #     myrange = range(2,10)

    ## 1) LOAD
    dfall = pd.DataFrame()
    cols2include = ['token', 'binIndex', var2look]
    for x in Xs:
        dfxout = pd.read_csv(os.path.join('Out_' + ID, 'returns_' + x + '.csv'))
        df2look = dfall.append(dfxout.loc[dfxout.strategyName == strategy, cols2include])
        # Merge
        # dfres = pd.merge(inputfile,dfxout[dfxout.strategyName==strategy],on=['token'])
        # dfreturns = dfall.append(dfres)
    del dfxout
    # Stack into bins
    dfvals = df2look.set_index(['token']).groupby(['token'])[var2look].apply(
        lambda df: df.reset_index(drop=True)).unstack().reset_index()

    ## 2) Extract conditions from the token
    dfvals['ccy'] = dfvals.token.apply(lambda s: s[:6]).values
    dfvals['trdate'] = dfvals.token.apply(lambda s: int(s[7:15])).values
    dfvals['size'] = dfvals.token.apply(lambda s: int(s[30:])).values

    print('- File loaded:', var2look)
    return dfvals

def dyce_plot_scheduling_plusMC(dfvals,dfvolu,dfvola,dfspread,strategy,var2look, myccys, mydates, mysizes, myrange, binstart, savelabel,binend=288,addMC=True):
    #     myccys = ['EURUSD','AUDUSD','EURJPY','EURGBP','USDMXN','USDHKD']
    #     mydates = [20170522,20180430]
    #     mysizes = [20,200,1000,10000]

    # Adjust to the engine bin index
    binstart+=1
    binend += 1

    myfont = 20
    #binstart = 12 * 8

    pdf = matplotlib.backends.backend_pdf.PdfPages("Figures/results_%s.pdf" % savelabel)

    for ccy in myccys:
        for date in mydates:
            for size in mysizes:
                df2plot = dfvals[(dfvals.ccy == ccy) & (dfvals.trdate == date) & (dfvals['size'] == size)]
                tmp = df2plot.replace(0, np.NaN).copy()
                #fig,ax1 = plt.figure(figsize=(20, 7))
                fig, ax1 = plt.subplots(figsize=(15, 8))
                # for row in range(tmp.shape[0]):
                for row in myrange:
                    token = tmp.iloc[row, 0]
                    #if int(token[23:28])-int(token[16:21]) > 5*60: # Do not plot dur=5min (to keep plot scale)
                    i0 = str(datetime.timedelta(seconds=int(token[16:21])))[:-3]
                    i1 = str(datetime.timedelta(seconds=int(token[23:28])))[:-3]
                    mylabel = i0 + '-' + i1
                    if var2look=='binShares':
                        vals2plot = tmp.iloc[row, binstart:binend] / 1000000
                        myylabel = var2look + '(million)'
                    if (var2look=='returnEndTotal') | (var2look=='returnStartTotal'):
                        vals2plot = tmp.iloc[row, binstart:binend]
                        myylabel = var2look + '(bps)'

                    #ax1.plot(np.arange(binstart, binend), vals2plot,'o-',markersize=7,linewidth=3, label=mylabel,mfc='none')
                    ax1.bar(np.arange(binstart, binend), vals2plot,alpha=0.8)

                    #ax1.scatter(np.arange(binstart, binend), vals2plot,s=50, facecolors='none')

                    #print('Vals2plot:',vals2plot)

                    # plt.plot(np.arange(binstart,289),tmp.iloc[row,binstart:289],label=mylabel)
                    # plt.plot(np.arange(binstart,289),tmp.iloc[row,binstart:289]/1000000,label=tmp.iloc[row,0])

                    plt.xticks(np.arange(binstart, binend, 12),
                               [jifin.binToTime(i-1, binSize=300)[:5] for i in np.arange(binstart, binend, 12)], rotation=20,
                               #[jifin.binToTime(i, binSize=300) for i in np.arange(binstart, binend, 12)], rotation=15,
                               fontsize=myfont)
                    plt.yticks(fontsize=myfont)
                    plt.xlabel('time of day',fontsize=myfont)

                    #plt.ylabel(myylabel, fontsize=myfont)
                    ax1.set_ylabel(myylabel, fontsize=myfont)

                    plt.title('Scheduling (%s) %d: %s size=%d(million)' % (strategy, date, ccy, size), fontsize=myfont)
                    ax1.legend(fontsize=myfont)
                    #ax1.set_ylim(0,np.max(vals2plot)+1)
                    ax1.set_ylim(auto=True)


                if addMC:
                    ## Plot volume and volatility
                    ax2 = ax1.twinx()
                    volume = dfvolu[dfvolu.token==token].iloc[:,1:].values # Get from the last used token (It does not matter which duration)
                    vola = dfvola[dfvola.token == token].iloc[:,1:].values
                    spread = dfspread[dfspread.token == token].iloc[:, 1:].values
                    ax2.plot(np.arange(binstart, binend), 10*volume[0][binstart:binend],'b--', label='10*volume')
                    ax2.plot(np.arange(binstart, binend), vola[0][binstart:binend], 'r--', label='volatility')
                    ax2.plot(np.arange(binstart, binend), spread[0][binstart:binend]*1000, 'c--', label='1000*spread')
                    ax2.plot(np.arange(binstart, binend), np.power(volume[0][binstart:binend],2/3)/vola[0][binstart:binend]/10, 'g--', label='$0.1*\\frac{volu^{2/3}}{vola}$' )
                    ax2.set_ylabel('weights',fontsize=myfont)
                    #ax2.plot(volume[0][binstart:288], 'r--', label='volume')
                    #ax2.legend(fontsize=myfont,loc=3)
                    ax2.legend(fontsize=myfont)

                plt.yticks(fontsize=myfont)
                plt.tight_layout()
                plt.show()

                pdf.savefig(fig)


    #plt.savefig('results_%s.pdf'%savelabel)

    #pdf = matplotlib.backends.backend_pdf.PdfPages("results_%s.pdf"%savelabel)
    #for fig in range(1,10):  ## will open an empty extra figure :(
    #    pdf.savefig(fig)
    pdf.close()


def dyce_plot_scheduling_plusVoluVola(dfvals,dfvolu,dfvola,strategy,var2look, myccys, mydates, mysizes, myrange, binstart = 12 * 8):
    #     myccys = ['EURUSD','AUDUSD','EURJPY','EURGBP','USDMXN','USDHKD']
    #     mydates = [20170522,20180430]
    #     mysizes = [20,200,1000,10000]

    myfont = 20
    #binstart = 12 * 8

    for ccy in myccys:
        for date in mydates:
            for size in mysizes:
                df2plot = dfvals[(dfvals.ccy == ccy) & (dfvals.trdate == date) & (dfvals['size'] == size)]
                tmp = df2plot.replace(0, np.NaN).copy()
                #fig,ax1 = plt.figure(figsize=(20, 7))
                fig, ax1 = plt.subplots(figsize=(20, 7))
                # for row in range(tmp.shape[0]):
                for row in myrange:
                    token = tmp.iloc[row, 0]
                    i0 = str(datetime.timedelta(seconds=int(token[16:21])))[:-3]
                    i1 = str(datetime.timedelta(seconds=int(token[23:28])))[:-3]
                    mylabel = i0 + '-' + i1
                    if var2look=='binShares':
                        vals2plot = tmp.iloc[row, binstart:288] / 1000000
                        myylabel = var2look + '(million)'
                    if var2look=='returnEndTotal':
                        vals2plot = tmp.iloc[row, binstart:288]
                        myylabel = var2look + '(bps)'

                    ax1.plot(np.arange(binstart, 288), vals2plot, label=mylabel)
                    # plt.plot(np.arange(binstart,289),tmp.iloc[row,binstart:289],label=mylabel)
                    # plt.plot(np.arange(binstart,289),tmp.iloc[row,binstart:289]/1000000,label=tmp.iloc[row,0])

                    plt.xticks(np.arange(binstart, 288, 12),
                               [jifin.binToTime(i, binSize=300) for i in np.arange(binstart, 288, 12)], rotation=15,
                               fontsize=myfont)
                    plt.yticks(fontsize=myfont)
                    plt.xlabel('time of day',fontsize=myfont)

                    #plt.ylabel(myylabel, fontsize=myfont)
                    ax1.set_ylabel(myylabel, fontsize=myfont)

                    plt.title('Scheduling (%s) %d: %s size=%d(million)' % (strategy, date, ccy, size), fontsize=myfont)
                    ax1.legend(fontsize=myfont);

                ## Plot volume and volatility
                ax2 = ax1.twinx()
                volume = dfvolu[dfvolu.token==token].iloc[:,1:].values # Get from the last used token (It does not matter which duration)
                vola = dfvola[dfvola.token == token].iloc[:,1:].values
                ax2.plot(np.arange(binstart, 288), 10*volume[0][binstart:288],'b--', label='10*volume')
                ax2.plot(np.arange(binstart, 288), vola[0][binstart:288], 'r--', label='volatility')
                ax2.plot(np.arange(binstart, 288), np.power(volume[0][binstart:288],2/3)/vola[0][binstart:288]/10, 'g--', label='$0.1*\\frac{volu^{2/3}}{vola}$')
                ax2.set_ylabel('weights',fontsize=myfont)
                #ax2.plot(volume[0][binstart:288], 'r--', label='volume')
                ax2.legend(fontsize=myfont,loc=2)

                plt.yticks(fontsize=myfont)
                plt.show()

def dyce_plot_scheduling(dfvals,strategy,var2look, myccys, mydates, mysizes, myrange, binstart = 12 * 8):
    #     myccys = ['EURUSD','AUDUSD','EURJPY','EURGBP','USDMXN','USDHKD']
    #     mydates = [20170522,20180430]
    #     mysizes = [20,200,1000,10000]

    myfont = 20
    #binstart = 12 * 8

    for ccy in myccys:
        for date in mydates:
            for size in mysizes:
                df2plot = dfvals[(dfvals.ccy == ccy) & (dfvals.trdate == date) & (dfvals['size'] == size)]
                tmp = df2plot.replace(0, np.NaN).copy()
                plt.figure(figsize=(20, 7))
                # for row in range(tmp.shape[0]):
                for row in myrange:
                    token = tmp.iloc[row, 0]
                    i0 = str(datetime.timedelta(seconds=int(token[16:21])))[:-3]
                    i1 = str(datetime.timedelta(seconds=int(token[23:28])))[:-3]
                    mylabel = i0 + '-' + i1
                    if var2look=='binShares':
                        vals2plot = tmp.iloc[row, binstart:289] / 1000000
                        myylabel = var2look + '(million)'
                    if var2look=='returnEndTotal':
                        vals2plot = tmp.iloc[row, binstart:289]
                        myylabel = var2look + '(bps)'

                    #plt.plot(np.arange(binstart, 289), vals2plot, label=mylabel)
                    plt.bar(np.arange(binstart, 289), vals2plot, label=mylabel,alpha=0.8)
                    # plt.plot(np.arange(binstart,289),tmp.iloc[row,binstart:289],label=mylabel)
                    # plt.plot(np.arange(binstart,289),tmp.iloc[row,binstart:289]/1000000,label=tmp.iloc[row,0])
                    plt.xticks(np.arange(binstart, 288, 12),
                               [jifin.binToTime(i, binSize=300)[:5] for i in np.arange(binstart, 288, 12)], rotation=15,
                               fontsize=myfont)
                    plt.xlabel('time of day',fontsize=myfont)
                    plt.ylabel(myylabel, fontsize=myfont)
                    plt.title('Scheduling (%s) %d: %s size=%d(million)' % (strategy, date, ccy, size), fontsize=myfont)
                    plt.legend(fontsize=myfont);

                plt.yticks(fontsize=myfont)
                plt.show()


def dyce_load_intradayData_file(fname,var2look='binVolume'):
    ## 1) LOAD
    cols2include = ['token','binIndex',var2look]
    df2look = pd.read_csv(fname)
    # Stack into bins
    dfvals = df2look.set_index(['token']).groupby(['token'])[var2look].apply(lambda df: df.reset_index(drop=True)).unstack().reset_index()

    return dfvals


def dyce_load_intradayData(ID,Xs,var2look='binVolume'):
    ## 1) LOAD
    dfall = pd.DataFrame()
    cols2include = ['token','binIndex',var2look]
    for x in Xs:
        dfxout = pd.read_csv(os.path.join('Out_'+ID,'intradayData_'+x+'.csv'))
        df2look = dfall.append(dfxout.loc[:,cols2include])

    del dfxout
    # Stack into bins
    dfvals = df2look.set_index(['token']).groupby(['token'])[var2look].apply(lambda df: df.reset_index(drop=True)).unstack().reset_index()

    return dfvals


def pz_GetDyCECalData(RunDir,Xs):
    #BasicData = pd.read_csv(RunDir + "/basicData.csv.gz",
    #                        usecols=['token', 'HLG', 'lg', 'event', 'closePrice', 'MDDV', 'endShares',
    #                                 'GcalibrationCoefficient', 'betaSpread'])
    CostData = pd.read_csv(RunDir + "/costAndRisk_"+Xs[0]+".csv",
                           usecols=['token', 'strategyName', 'totalCostFinal', 'SpreadCost', 'FCost', 'GCost',
                                    'MOCCost', 'totalCostConcave', 'endOfOrderImpact'])
    ReturnData = pd.read_csv(RunDir + "/returns_"+Xs[0]+".csv",
                             usecols=['token', 'binIndex', 'binShares', 'returnEndTotal', 'returnEndDueToMOC',
                                      'returnInstantDueToF'])
    IntraDayData = pd.read_csv(RunDir + "/intradayData_"+Xs[0]+".csv", usecols=['token', 'binIndex', 'binADVforFG'])
    #pp = pd.read_csv(RunDir + "/postProcess.csv.gz", usecols=['token', 'Z', 'tau', 'phi', 'mu', 'chi'])

    BinList = np.sort(ReturnData.binIndex.unique())
    MPOReturnCol = ['MPOReturn' + str(i) for i in BinList]
    NetReturnCol = ['NetReturn' + str(i) for i in BinList]
    FInsCol = ['FIns' + str(i) for i in BinList]
    FMOCCol = ['FMOC' + str(i) for i in BinList]

    ReturnData = ReturnData.merge(IntraDayData, on=['token', 'binIndex'])
    ReturnData['SWPrate'] = ReturnData.binShares * ReturnData.binShares / ReturnData.binADVforFG
    Group = ReturnData.groupby('token')['binShares', 'SWPrate'].sum()
    Group.SWPrate /= Group.binShares
    Group.reset_index(inplace=True)
    CostData = CostData.merge(Group, how='left', on='token')
    ReturnData.drop(columns=['binShares', 'binADVforFG', 'SWPrate'], inplace=True)
    ReturnData = ReturnData.pivot_table(index='token', columns='binIndex').reset_index()
    ReturnData.columns = ['token'] + FMOCCol + MPOReturnCol + FInsCol
    #CostData = CostData.merge(BasicData, how='left', on='token')
    CostData = CostData.merge(ReturnData, how='left', on='token')
    #CostData = CostData.merge(pp, how='left', on=['token'])
    return CostData.copy()

# Use the utility below for the aggregation
#    http: // git - real - sdla / Research / MPO / blob / dev / Calibration / scripts / CostCurve.py
# ---------------------------------------------------------------------------------------


###############################################
## P3: FXACE (2019) - extension of EmpHelper_ji
###############################################
def fx_getVolumeDist(pairname,dateint,dpath = '/local/data/itg/fe/prod/data1/HistoricalFiles/FX/1.0/DailyStats/Volume/distributions/data'):
    vlmdist = pd.read_csv(os.path.join(dpath,'fxDistribution_Volume_1.0_'+str(dateint)+'.dat.gz'),
        compression='gzip', skiprows=range(0, 7), header=None)
    #tmp = vlmdist[(vlmdist.iloc[:, 0] == pairname)&(vlmdist.iloc[:, 5] == percentile)]
    myfont = 16
    tmp = vlmdist[(vlmdist.iloc[:, 0] == pairname)]
    plt.figure(figsize=[20, 5])
    for row in range(tmp.shape[0]):
        plt.plot(np.arange(1, 289), tmp.iloc[row, 3:], label=tmp.iloc[row, 1])
        #plt.plot(np.arange(1, 289), tmp.iloc[row, 6:], label=tmp.iloc[row, 1])
    plt.xticks(np.arange(1, 288, 50), [jifin.binToTime(i, binSize=300) for i in np.arange(1, 288, 50)], rotation=15,fontsize=myfont)
    plt.xlabel('time of day',fontsize=myfont)
    plt.ylabel('volume weight',fontsize=myfont)
    plt.title('Volume distribution: ' + pairname,fontsize=myfont)
    plt.legend(fontsize=myfont)

    return tmp

def fx_getVolaDist(pairname,dateint,percentile = 50,dpath = '/local/data/itg/fe/prod/data1/HistoricalFiles/FX/1.0/DailyStats/Vol/distributions/data'):
    vlmdist = pd.read_csv(os.path.join(dpath,'fxDistribution_Vol_1.0_'+str(dateint)+'.dat.gz'),
        compression='gzip', skiprows=range(0, 7), header=None)
    myfont = 16
    #tmp = vlmdist[vlmdist.iloc[:, 0] == pairname]
    print('- Using monthly profile with percentile:', percentile)
    tmp = vlmdist[(vlmdist.iloc[:, 0] == pairname) & (vlmdist.iloc[:, 5] == percentile)]
    plt.figure(figsize=[20, 5])
    for row in range(tmp.shape[0]):
        #plt.plot(np.arange(1, 289), tmp.iloc[row, 6:], label=tmp.iloc[row, 1])
        plt.plot(np.arange(1, 289), tmp.iloc[row, 6:], label=tmp.iloc[row, 3])

    plt.xticks(np.arange(1, 288, 50), [jifin.binToTime(i, binSize=300) for i in np.arange(1, 288, 50)], rotation=15,fontsize=myfont)
    plt.xlabel('time of day',fontsize=myfont)
    plt.ylabel('volatility weight',fontsize=myfont)
    plt.title('Volatility distribution: ' + pairname,fontsize=myfont)
    plt.legend(fontsize=myfont)

    return tmp


def p3_load_ECN_trades(ccypairs,trdates):
    """ Load ECN trade data and convert to the format used in Danlu's scripts.
    - execution price --> spot
    - BaseVolume converted to million (base currency)"""
    print('- Loading ECN trades...')
    dfall = pd.DataFrame()
    for trdate in trdates:
        sdate = str(trdate)
        cdate = datetime.datetime(int(sdate[:4]), int(sdate[4:6]), int(sdate[6:]))
        print('%s-'%sdate,end='')
        for ccy in ccypairs:
            #print('-Loading ECN trades ',cdate)
            dfx = ji_read_FX_ECN_trades(ccy[:3]+'.'+ccy[3:], cdate)
            #print(dfx)
            #if not dfx.empty:
            dfx['ccy'] = ccy
            dfx['trDate'] = trdate
            dfall = dfall.append(dfx)
    print('')
    dfall.rename(columns={'Timestamp':'stamp','ExecutionPrice':'spot','BaseVolume':'tradeSize','Side':'side'}, inplace=True)

    ## Convert to appropriate format
    # 1) Remove the milisec field in stamp
    dfall.stamp = dfall.stamp.apply(lambda s: s[:8])
    # 2) Side with "Sell" and "Buy"
    dfall.replace({'S':'Sell','B':'Buy'}, inplace=True)
    # 3) Trade sizes in milllion and US$
    dfall.tradeSize = dfall.tradeSize/1000000
    dfall['tradeSizeUSD'] = dfall.tradeSize * dfall.spot # Quick approximation for now...
    print('- Use the spot price to convert tradeSize to USD (quick solution)...')
    # 4) Create timeBinCourse (5min)
    dfall['timeBinCoarse'] = dfall.stamp.apply(lambda s: (3600*int(s[:2])+60*int(s[3:5])+int(s[6:]))//300 + 1) # 5min
    dfall['timeBinFine'] = dfall.stamp.apply(lambda s: (3600 * int(s[:2]) + 60 * int(s[3:5]) + int(s[6:])) // 15 + 1) # 15sec

    return dfall.drop(columns=['id']).reset_index(drop=True)



###############################################
## P5: AlgoDB Cost Prediction
###############################################


#########################################
## P5.b: Prediction with Sklearn

def helper_analyze_dfres_weights(dfres, model='ridge'):
    # From: data_train.X.columns
    selectedCols = ['sign', 'lg', 'histVola', 'binDuration', 'bin', 'ExPriceForBin',
     'ExShareForBin', 'FillNbrForBin', 'midQuote', 'LowPrice', 'HighPrice',
     'vwap', 'spread', 'histVola5min', 'histVolu5min', 'realVola', 'surVola',
     'priceBM', 'empCost', 'prate', 'normStyle', 'quadVola',
     'last30quadVola', 'cumquadVola', 'dollarVolume', 'surVolume', 'FGcost',
     'weekDay']
    selectedCols = ['w_%s' % i for i in selectedCols]

    #
    dfw = pd.DataFrame(columns=['HLG','bin','train_r2','test_r2'] + selectedCols)
    cont = 0
    if model=='ridge':
        coeflabel, r2label = 'coef_ridge', 'R2_rid'
    elif model=='rf':
        coeflabel, r2label = 'vim_rf', 'R2_RF'
    elif model=='xgb':
        coeflabel, r2label = 'vim_xgb', 'R2_XGB'

    for index, row in dfres.iterrows():
        wall = [float(i) for i in row[coeflabel].split(' ')]
        print('wall(len=%d):'%len(wall), wall)
        dfw.loc[cont, 'HLG'] = row['LG']
        dfw.loc[cont, 'bin'] = row['bin']
        dfw.loc[cont, 'train_r2'] = row[r2label]
        dfw.loc[cont, 'test_r2'] = row[r2label+'_Test']
        dfw.loc[cont, selectedCols[0]:selectedCols[-1]] = np.array(wall)
        cont += 1

    return dfw

def helper_prepare_df2pred(df2pred, options):
    data = matlablike()
    df = df2pred.copy()

    # Set filters
    myLG = options.LG
    myBins = options.bin
    #isETF = options.isETF
    #EventType = options.EventType
    remove_outliers = options.remove_outliers
    mypercentile = options.outlier_percent
    start_date = options.start_date
    end_date = options.end_date
    target = options.target_column
    normType = options.normType # 0: do nothing, 1: normalize by (x-mean)/std, 2: Quantile

    # 1) Select date range
    print('Selected period:', start_date, ' to ', end_date)
    mask_date = (df.tradeDate >= start_date) & (df.tradeDate <= end_date)
    df = df.loc[mask_date]

    # 2) Filter columns
    # Basic data to consider
    """ # (too many...)
        selectedCols = ['sign','tradeDate','relSize','lg','HLG',
       'mddv', 'histVola','openPrice','binDuration','bin',
       'ExPriceForBin','ExShareForBin', 'FillNbrForBin', 'midQuote',
       'LowPrice', 'HighPrice', 'vwap', 'spread', 'TradeWeightedSpread',
       'shareVolume', 'histVolaW', 'histVola5min', 'histVoluW', 'histVolu5min',
       'realVola', 'surVola', 'priceBM', 'empCost', 'Fcost', 'Gcost', 'prate',
       'style', 'normStyle', 'quadVola', 'stdVola', 'last30HLVola',
       'cumHLVola', 'last30quadVola', 'cumquadVola', 'empCostMM', 'FcostMM',
       'GcostMM', 'dollarVolume', 'surVolume', 'FGcost', 'empCost_next5min']
    """
    selectedCols = ['sign', 'tradeDate', 'lg', 'HLG',
                    'histVola', 'binDuration', 'bin',
                    'ExPriceForBin', 'ExShareForBin', 'FillNbrForBin', 'midQuote',
                    'LowPrice', 'HighPrice', 'vwap', 'spread',
                    'histVola5min', 'histVolu5min',
                    'realVola', 'surVola', 'priceBM', 'empCost', 'prate',
                    'normStyle', 'quadVola', 'last30quadVola', 'cumquadVola',
                    'dollarVolume', 'surVolume', 'FGcost', 'empCost_next5min']

    df = df.loc[:, selectedCols]

    # 3) Filter rows
    pairs = [(i, j) for i in myLG for j in myBins]
    for iLG, iBin in pairs:
        print('Sample size (HLG=%d,Bin=%d): %d' % (iLG, iBin, len(df[(df.HLG == iLG) & (df.bin == iBin)])))

    # a) Select bins
    df = df[df.bin.isin(myBins)]
    # b) Select LG
    df = df[df.HLG.isin(myLG)]
    # display(df)

    # c) Select EventType
    #df = df[df.EventType.isin(EventType)]
    #print('Included EventType:', df.EventType.isin(EventType).sum(), EventType)
    # d) Select isETF
    #df = df[df.IsETF == isETF]

    # e) Exclude outliers
    if remove_outliers:
        vars2rem = [target]
        df =jifin.remove_outliers_df_down(df,percent=mypercentile,vars2rem=vars2rem, showplot=False)
        df =jifin.remove_outliers_df_up(df,percent=mypercentile,vars2rem=vars2rem, showplot=False)

    # 4) Fill NaN if some
    if df.isnull().sum().sum() > 0:
        df = df.fillna(0)
        print('Replaced %d NaN with zeros...' % (df.isnull().sum().sum()))

    # 5) Add weekDay (day of the week 0:Monday... 6:Sunday
    print('- Added day of the week as feature...')
    df['weekDay'] = df2pred.tradeDate.apply(jifin.get_intDate2weekday)

    # 6) Standardize (specially useful for interpretation of regression weights
    if normType==1:
        print('- Features and target standardized: (x-mean)/std ...')
        scaled_features = StandardScaler().fit_transform(df)
        #scaled_target_values = StandardScaler().fit_transform(target_values.reshape(-1, 1)) # Not working properly...
        df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    if normType==2:
        print('- Features and target Quantile Normalizer(output:Gaussian) ...')
        scaled_features = QuantileTransformer().fit_transform(df)
        df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    if normType==3: # TODO: not working. Problem with NaN values...
        print('- Apply log to volatility and volume related (non-neg) feats ...')
        iselected = ['histVola','midQuote',
                     'ExPriceForBin', 'ExShareForBin',
                    'LowPrice', 'HighPrice', 'vwap', 'spread',
                    'histVola5min', 'histVolu5min',
                    'realVola', 'surVola', 'priceBM',
                    'quadVola', 'last30quadVola', 'cumquadVola',
                    'dollarVolume', 'surVolume']
        df[iselected] = df[iselected].apply(np.log).fillna(0)

    # 7) Drop off unused cols
    scaled_target_values = df[target].values # get target columns before dropoff
    df = df.drop(columns=[target, 'tradeDate','HLG'])

    # 8) Prepare the data to return
    #display(df.head())
    #selectedReg = ['SurpriseCumu', 'Surprise30Min', 'SurpriseCumu.PrevDay']
    data = matlablike()
    data.options = options
    data.y = scaled_target_values
    data.X = df
    data.X_labels = data.X.columns
    data.y_label = target
    print('Final number of samples:', len(data.y))
    return data

#########################################
## P5.a: Analysis Cost x Size x Volatility

def p5_wrapper_examine_Set1(df2go,imask,xcostName,xcostLabel,saveflag,excludeZeroY=False):
    print('####################################################################################')
    print('Selected trades (using imask): %d (%2.2f%%)' % (np.sum(imask), 100 * np.sum(imask) / df2go.shape[0]))
    print('####################################################################################')

    print('------ Examined variable: %s --------'%xcostLabel)
    print(df2go.loc[imask,xcostName].describe())

    print('##############################################################################')
    print('################## (I)   %s  x Size ########################################' % xcostLabel)
    print('############################################################################## \n')
    p5_wrapper_examine_CostxSize(df2go[imask], costName=xcostName, costLabel=xcostLabel, plotHist=True,
                                     perc2exclude=0.5, excludeZeroY=excludeZeroY, saveflag=saveflag)

    print('###############################################################################################')
    print('################## (II)  %s  x  RealVola    #################################################' % xcostLabel)
    print('###############################################################################################/n')
    p5_wrapper_examine_CostxVola(df2go[imask], xcostName, xcostLabel, 'realVola', plotHist=False, perc2exclude=0.5,
                                     excludeZeroY=excludeZeroY, saveflag=saveflag)

    print('#######################################################################################################')
    print(
        '################## (III) %s  x  SurVola               ###############################################' % xcostLabel)
    print('#######################################################################################################/n')
    p5_wrapper_examine_CostxVola(df2go[imask], xcostName, xcostLabel, 'surVola', plotHist=False, perc2exclude=0.5,
                                     excludeZeroY=excludeZeroY, saveflag=saveflag)

    print('###############################################################################################')
    print('################## (IV)  %s  x  relSize    #################################################' % xcostLabel)
    print('###############################################################################################/n')
    p5_wrapper_examine_CostxrelSize(df2go[imask], xcostName, xcostLabel, 'relSize', plotHist=False,
                                        perc2exclude=0.5, excludeZeroY=excludeZeroY, saveflag=saveflag)

    print('###############################################################################################')
    print('################## (V)  %s  x  Dollar Volume    ############################################' % xcostLabel)
    print('###############################################################################################/n')
    p5_wrapper_examine_CostxrelSize(df2go[imask], xcostName, xcostLabel, 'dollarVolume', plotHist=False,
                                        perc2exclude=0.5, excludeZeroY=excludeZeroY, saveflag=saveflag)

    print('#######################################################################################################')
    print(
        '################## (VI) %s  x  Surprise Volume         ###############################################' % xcostLabel)
    print('#######################################################################################################/n')
    p5_wrapper_examine_CostxrelSize(df2go[imask], xcostName, xcostLabel, 'surVolume', plotHist=False,
                                        perc2exclude=0.5, excludeZeroY=excludeZeroY, saveflag=saveflag)

def p5_wrapper_examine_crossCorr(df2go,imask,costName,tlabel,perc2exclude = 1):
    """Wrapper for a set of cross correlation maps.
    costName: choose the cost to analyse 'empCost','Fcost', etc
    """

    ## 2) Create df to crossCorrelate
    dfx = df2go[imask].copy()

    ## 0) Make sure to exclude trades without client execution
    dfx = df2go[df2go.empCost != 0].copy()
    print('- Excluded bins without client execution: %d (%d%%)' % (np.sum(df2go.empCost == 0), 100 * np.sum(df2go.empCost == 0) / df2go.shape[0]))
    # Outlier
    dfx = jifin.remove_outliers_df_down(dfx, percent=perc2exclude / 2, vars2rem=[costName])
    dfx = jifin.remove_outliers_df_up(dfx, percent=perc2exclude / 2, vars2rem=[costName])

    dfx['logSpread'] = dfx.spread.apply(np.log)
    dfx['logQuadVola'] = dfx.quadVola.apply(np.log)
    dfx['logVolume'] = dfx.shareVolume.apply(np.log)
    # dfx['logEmpCost'] = dfx.empCost.apply(np.log)

    ## 3a) Create var to condition: bin
    condranges = [[1, 20], [20, 40], [40, 70], [70, 78]]
    condvarLabel = 'Bin'  # prefix of discretized values
    condvar = 'bin'  # column to hue
    # Discretize condition var
    dfx['condition'] = jifin.run_discretize(df2go.loc[imask, condvar], condranges, condvarLabel)
    ## 3b) Plot
    # cols2include = ['logSpread','logQuadVola','empCost','normStyle','condition']
    cols2include = ['logSpread', 'logQuadVola', 'logVolume', costName, 'condition']
    mytitle = 'Cross correlation (%s, Condition:%s)' %(tlabel, condvarLabel)
    ji.plot_crossCorr_df2go(dfx[cols2include], condvar='condition', title=mytitle)

    ## 4a) Create var to condition: normStyle
    condranges = 4
    condvarLabel = 'normStyle'  # prefix of discretized values
    condvar = 'normStyle'  # column to hue
    # Discretize condition var
    dfx['condition'] = jifin.run_discretize(df2go.loc[imask, condvar], condranges, condvarLabel)
    ## 4b) Plot
    # cols2include = ['logSpread','logQuadVola','empCost','normStyle','condition']
    cols2include = ['logSpread', 'logQuadVola', 'logVolume', costName, 'condition']
    mytitle = 'Cross correlation (%s, Condition:%s)' % (tlabel, condvarLabel)
    ji.plot_crossCorr_df2go(dfx[cols2include], condvar='condition', title=mytitle)

def p5_wrapper_examine_CostxVolume(df2go,costName,costLabel, plotHist=True,perc2exclude = 1):
    """Wrapper to repeat a set of analyses."""

    ## 0) Make sure to exclude trades without client execution
    dfx = df2go[df2go.empCost != 0].copy()
    print('- Excluded bins without client execution: %d (%d%%)' % (np.sum(df2go.empCost == 0), 100 * np.sum(df2go.empCost == 0) / df2go.shape[0]))
    # Outlier
    dfx = jifin.remove_outliers_df_down(dfx, percent=perc2exclude / 2, vars2rem=[costName])
    dfx = jifin.remove_outliers_df_up(dfx, percent=perc2exclude / 2, vars2rem=[costName])

    ## Commons parameters
    xName = 'dollarVolume' # Choose the x-axis variable

    ## III) NormStyle
    condName = 'normStyle'
    cflag = {'nsplit': 4, 'daytime': [22, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName,xName, cflag)

    ## III.b) Time of the day
    condName = 'bin'
    cflag = {'nsplit': 4, 'daytime': [2, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag)

    ## IV) binDuration
    condName = 'binDuration'
    # nsplit: is for the condVar
    #condrange = [[1, 40], [40, 70], [70, 78]]
    condrange = [[1, 6], [6, 18], [18, 40], [40, 78]]
    cflag = {'nsplit': condrange, 'daytime': [2, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag)

    ## V) Spread (time weighted)
    condName = 'spread'
    cflag = {'nsplit': 3, 'daytime': [22, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag)

    ## VI) Spread (time weighted)
    condName = 'TradeWeightedSpread'
    cflag = {'nsplit': 3, 'daytime': [22, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag)

    ## VII) FillNbr
    condName = 'FillNbrForBin'
    cflag = {'nsplit': 3, 'daytime': [22, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag)

    ## VIII) Strategy
    condName = 'algoStrategy'
    condrange = ['VWAP:100', 'Volume Participation:100', 'TWAP:100']
    cflag = {'nsplit': condrange, 'daytime': [22, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag)


def p5_wrapper_examine_CostxVola(df2go,costName,costLabel,volaType, plotHist=True,perc2exclude = 1, excludeZeroY=True, saveflag=None):
    """Wrapper to repeat a set of analyses."""

    daytime = [22, 77]  # bin range to evaluate 1 (9:30).. 78(16:00)
    #daytime = [1, 78]  # bin range to evaluate 1 (9:30).. 78(16:00)

    # Create PDF if requested
    if saveflag is not None:
        pdfobj = matplotlib.backends.backend_pdf.PdfPages(
            os.path.join(saveflag.outDir, "fig_%s_x_%s_%s.pdf" % (costName,volaType, saveflag.label)))

    ## 0) Make sure to exclude trades without client execution
    dfx = df2go[df2go.empCost != 0].copy()
    print('- Excluded bins without client execution: %d (%d%%)' % (np.sum(df2go.empCost == 0), 100 * np.sum(df2go.empCost == 0) / df2go.shape[0]))
    # Outlier
    dfx = jifin.remove_outliers_df_down(dfx, percent=perc2exclude / 2, vars2rem=[costName])
    dfx = jifin.remove_outliers_df_up(dfx, percent=perc2exclude / 2, vars2rem=[costName])

    ## Commons parameters
    xName = volaType # Choose the x-axis variable

    ## III) NormStyle
    condName = 'normStyle'
    cflag = {'nsplit': 4, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName,xName, cflag, pdfobj)

    ## III.b) Time of the day
    condName = 'bin'
    cflag = {'nsplit': 4, 'daytime': [2, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag, pdfobj)

    ## IV) binDuration
    condName = 'binDuration'
    # nsplit: is for the condVar
    condrange = [[1, 40], [40, 70], [70, 78]]
    cflag = {'nsplit': condrange, 'daytime': [2, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag, pdfobj)

    ## V) Spread (time weighted)
    condName = 'spread'
    cflag = {'nsplit': 3, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag, pdfobj)

    ## VI) Spread (time weighted)
    condName = 'TradeWeightedSpread'
    cflag = {'nsplit': 3, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag, pdfobj)

    ## VII) FillNbr
    condName = 'FillNbrForBin'
    cflag = {'nsplit': 3, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag, pdfobj)

    ## VIII) Strategy
    condName = 'algoStrategy'
    condrange = ['VWAP:100', 'Volume Participation:100', 'TWAP:100']
    cflag = {'nsplit': condrange, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag, pdfobj)

    if saveflag is not None:
        pdfobj.close()

def p5_wrapper_examine_CostxrelSize(df2go,costName,costLabel,volaType, plotHist=True,perc2exclude = 1, excludeZeroY=True, saveflag=None):
    """Wrapper to repeat a set of analyses."""

    daytime = [22, 77]  # bin range to evaluate 1 (9:30).. 78(16:00)
    #daytime = [1, 78]  # bin range to evaluate 1 (9:30).. 78(16:00)

    # Create PDF if requested
    if saveflag is not None:
        pdfobj = matplotlib.backends.backend_pdf.PdfPages(
            os.path.join(saveflag.outDir, "fig_%s_x_%s_%s.pdf" % (costName,volaType, saveflag.label)))

    ## 0) Make sure to exclude trades without client execution
    dfx = df2go[df2go.empCost != 0].copy()
    print('- Excluded bins without client execution: %d (%d%%)' % (np.sum(df2go.empCost == 0), 100 * np.sum(df2go.empCost == 0) / df2go.shape[0]))
    # Outlier
    dfx = jifin.remove_outliers_df_down(dfx, percent=perc2exclude / 2, vars2rem=[costName])
    dfx = jifin.remove_outliers_df_up(dfx, percent=perc2exclude / 2, vars2rem=[costName])

    ## Commons parameters
    xName = volaType # Choose the x-axis variable

    ## I) Realized Vola (HL)
    condName = 'realVola'
    cflag = {'nsplit':3, 'daytime':[22, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}   # bin range to evaluate 1 (9:30).. 78(16:00)
    #p5_helper2wrapper_xprate(dfx, costName, costLabel, condName,cflag, pdfobj)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag, pdfobj)

    ## II) Realized Vola Surprise (HL)
    condName = 'surVola'
    cflag = {'nsplit': 3, 'daytime': [22, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}   # bin range to evaluate 1 (9:30).. 78(16:00)
    #p5_helper2wrapper_xprate(dfx, costName, costLabel, condName, cflag, pdfobj)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag, pdfobj)

    ## III) NormStyle
    condName = 'normStyle'
    cflag = {'nsplit': 4, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName,xName, cflag, pdfobj)

    ## III.b) Time of the day
    condName = 'bin'
    cflag = {'nsplit': 4, 'daytime': [2, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag, pdfobj)

    ## IV) binDuration
    condName = 'binDuration'
    # nsplit: is for the condVar
    condrange = [[1, 40], [40, 70], [70, 78]]
    cflag = {'nsplit': condrange, 'daytime': [2, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag, pdfobj)

    ## V) Spread (time weighted)
    condName = 'spread'
    cflag = {'nsplit': 3, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag, pdfobj)

    # ## VI) Spread (time weighted)
    # condName = 'TradeWeightedSpread'
    # cflag = {'nsplit': 3, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    # p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag, pdfobj)
    #
    # ## VII) FillNbr
    # condName = 'FillNbrForBin'
    # cflag = {'nsplit': 3, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    # p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag, pdfobj)

    ## VIII) Strategy
    condName = 'algoStrategy'
    condrange = ['VWAP:100', 'Volume Participation:100', 'TWAP:100']
    cflag = {'nsplit': condrange, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xVar(dfx, costName, costLabel, condName, xName, cflag, pdfobj)

    if saveflag is not None:
        pdfobj.close()


def p5_helper2wrapper_xVar(dfx,costName,costLabel,condName,xName,dicflag, pdfobj=None):
    """nsplit: is for the condVar. It can be integer, a range or string values"""

    # Get flag vars
    flag = argparse.Namespace(**dicflag)
    daytime = flag.daytime
    nsplit = flag.nsplit
    excludeZeroY = flag.excludeZeroY

    allbins = dfx.bin
    # 0) Distribution
    plotHist = False
    if flag.plotHist:
        if type(nsplit)==int:
            plotHist = True
        elif type(nsplit[0]) != str:
            plotHist = True

    print('## Condition Variable:',condName)
    print(dfx[condName].describe())
    if plotHist:
        sns.distplot(dfx[condName],bins=15)
        plt.title('Distribution of %s' % condName)
        print('%s>0: %d bins(5min)' % (condName, np.sum(dfx[condName] > 0)))
        print('%s<0: %d bins(5min)' % (condName, np.sum(dfx[condName] < 0)))

    # 1) Define the value to examine
    Y = dfx[costName]
    #size = dfx.prate # CHOOSE: normalized size
    #size_buckets = [[0,0.025],[0.025,0.05],[0.05,0.1],[0.1,1]] # depends on the size metric
    size = dfx[xName]  # CHOOSE
    size_buckets = 4
    #size_buckets = [[0, 0.025], [0.025, 0.05], [0.05, 0.1], [0.1, 1]]  # depends on the size metric

    # 2) CondVar
    condVar = dfx[condName] # CHOOSE
    #nsplit = 3 # number of categories for condVar
    #nsplit = [[1,40],[40,70],[70,78]]  # set manually
    allbins = dfx.bin
    #daytime = [22,77] # bin range to evaluate 1 (9:30).. 78(16:00)
    #daytime = [22,77] # bin range to evaluate 1 (9:30).. 78(16:00)
    #myflag = {'title':'EmpCost x Size','ylabel':'EmpCost(bps)', 'condlabel':condName,'xlabel':'prate'}
    myflag = {'title': '%s x %s'%(costName,xName), 'ylabel': '%s' % costLabel, 'condlabel': condName, 'xlabel': xName}
    # 3) Plot
    p5_helper_plot_Y_byBuckets(Y,size,allbins,condVar,size_buckets,daytime,nsplit,myflag, excludeZeroY, pdfobj)

def p5_helper_retrieveStrategy(df2plot,strat,sperc):
    print('- Number of %s: %d out of %d'%(strat,np.sum(df2plot.algoType.str.contains(strat + ':')),df2plot.shape[0]))
    df2plot = df2plot[df2plot.algoType.str.contains(strat + ':')]
    allStrat = df2plot.algoType.values
    allperc = []
    imask = []
    for s in allStrat:
        if int(s.split(strat + ':')[1].split(';')[0]) >= sperc:
            imask.append(True)
        else:
            imask.append(False)

    print('- Total %s retrieved: %d out %d' % (strat, np.sum(imask), len(imask)))
    return df2plot[imask]

def p5_helper_append_df2plot30min_ABD2017(df2plot2018,df2plotpath,allHLG,myStrategies = ['SCHEDULED','DARK','IS','OPPORTUNISTIC']):
    """Wrapper to
    - Load df2plot from 2017, concatenate with 2018 and save by Strategy and HLG
    """


    ######## 1) Get df2go
    #df2plot = pd.DataFrame()
    print('- Loading from:',df2plotpath)
    for HLG in allHLG:
        print('\n================================== Processing HLG=%d ======================================'%HLG)
        nn = 'df2plot30min_2017_HLGlarger_ABD_Q1Q2_HLG%d.csv.gz' % (HLG)
        fname = os.path.join(df2plotpath, nn)
        print('- %s... ' % (nn))
        df2plot2017 = pd.read_csv(fname) #
        nn = 'df2plot30min_2017_HLGlarger_ABD_Q3Q4_HLG%d.csv.gz' % (HLG)
        fname = os.path.join(df2plotpath, nn)
        print('- %s... ' % (nn))
        df2plot2017 = df2plot2017.append(pd.read_csv(fname))  #

        for strategy in myStrategies:
            print('----------------- Strategy: %s ------------------' % strategy)
            sperc = 0.7  # NOTE: assume that all strategies were already filtered at perc>0.7
            ## 1) Retrieve from 2017
            df2plot = p5_helper_retrieveStrategy(df2plot2017, strategy, sperc)
            imask = (df2plot.HLG == HLG)
            df2plot = df2plot[imask]
            ## 2) Retrieve from 2018
            df2018x = p5_helper_retrieveStrategy(df2plot2018, strategy, sperc)
            imask = (df2018x.HLG==HLG)
            df2plot = df2plot.append(df2018x[imask])

            print('- Total loaded 2017+2018: %d trades' % df2plot.shape[0])
            # Consistency check
            ji_p5_check_df2plot(df2plot)

            nn = 'df2plot30min_ABD_HLGlarger_2017_2018_%s_HLG%d.csv.gz' %(strategy,HLG)
            fname = os.path.join(df2plotpath, nn)
            print('- Saving %s...'%fname)
            df2plot.to_csv(fname,index=False,compression='gzip')


def p5_helper_append_df2plot_ABD2017(df2plot2018,df2plotpath, cols2include,allHLG,myStrategies = ['SCHEDULED','DARK','IS','OPPORTUNISTIC']):
    """Wrapper to
    - Load df2plot from 2017, concatenate with 2018 and save by Strategy and HLG
    """


    ######## 1) Get df2go
    #df2plot = pd.DataFrame()
    print('- Loading from:',df2plotpath)
    for HLG in allHLG:
        print('\n================================== Processing HLG=%d ======================================'%HLG)
        nn = 'df2plot_2017_HLGlarger_ABD_Q1Q2_HLG%d.csv.gz' % (HLG)
        fname = os.path.join(df2plotpath, nn)
        print('- %s... ' % (nn))
        df2plot2017 = pd.read_csv(fname)[cols2include]  #
        nn = 'df2plot_2017_HLGlarger_ABD_Q3Q4_HLG%d.csv.gz' % (HLG)
        fname = os.path.join(df2plotpath, nn)
        print('- %s... ' % (nn))
        df2plot2017 = df2plot2017.append(pd.read_csv(fname)[cols2include])  #

        for strategy in myStrategies:
            print('----------------- Strategy: %s ------------------' % strategy)
            sperc = 0.7  # NOTE: assume that all strategies were already filtered at perc>0.7
            ## 1) Retrieve from 2017
            df2plot = p5_helper_retrieveStrategy(df2plot2017, strategy, sperc)
            imask = (df2plot.HLG == HLG)
            df2plot = df2plot.loc[imask, cols2include]
            ## 2) Retrieve from 2018
            df2018x = p5_helper_retrieveStrategy(df2plot2018, strategy, sperc)
            imask = (df2018x.HLG==HLG)
            df2plot = df2plot.append(df2018x.loc[imask,cols2include])

            print('- Total loaded 2017+2018: %d trades' % df2plot.shape[0])
            # Consistency check
            ji_p5_check_df2plot(df2plot)

            nn = 'df2plot_ABD_HLGlarger_2017_2018_%s_HLG%d.csv.gz' %(strategy,HLG)
            fname = os.path.join(df2plotpath, nn)
            print('- Saving %s...'%fname)
            df2plot.to_csv(fname,index=False,compression='gzip')


def p5_helper_load_df2plot30min_ABD2018(df2plotpath,allrange,allHLG,myStrategies = ['SCHEDULED','DARK','IS','OPPORTUNISTIC']):
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
                    df2plot = df2plot.append(dfx[cols2include])

    ji_p5_check_df2plot(df2plot)
    return df2plot

def p5_helper_load_df2plot_April9(df2plotpath, cols2include,allrange,allHLG,myStrategies = ['SCHEDULED','DARK','IS','OPPORTUNISTIC']):
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
                    df2plot = df2plot.append(dfx[cols2include])

    ji_p5_check_df2plot(df2plot)
    return df2plot

def p5_wrapper_runDyCE_save_df2plot_April8(allrange,allHLG,myStrategies = ['SCHEDULED','DARK','IS','OPPORTUNISTIC'],allperc=[80,71,80,80], runDyCE=True, isTest=False):
    """Wrapper to
    - Load df2go
    - Generate inputplan
    - Run DyCE
    - Generate prediction and save df2plot"""

    #allrange = [[20180101, 20180131]]
    #allHLG = [0]

    ## SET STRATEGIES
    #myStrategies = ['SCHEDULED']
    #allperc = [80]
    #myStrategies = ['SCHEDULED','DARK','IS','OPPORTUNISTIC']
    #allperc = [80,71,80,80]

    ## SET DATE RANGES
    if isTest:
        #myweeks = [[1, 7]]
        myweeks = [[1, 7], [8, 15], [16, 23], [24, 31]]
    else:
        myweeks = [[1,7],[8,15],[16,23],[24,31]]
    #

    ## SET PATHS
    dpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data' # Where df2go is loaded from
    planpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/Plans'  # Where input plans will be saved
    df2plotpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data_df2plot'  # Where df2plot will be saved
    # newfe-s18: DyCE location and working directory
    DyCEpath = "/local/data/itg/fe/prod/data1/jide-local/DyCE/ref/DyCE/bin/DyCE"
    ppath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5'
    # feres-s12
    # DyCEpath = "/local/data/itg/fe/prod/data1/jide-local/MPO/DyCE/bin/DyCE"
    # spath = '/local/data/itg/fe/prod/data1/jide-local/P5/Out'

    ######## 1) Get df2go
    for t0t1 in allrange:
        for HLG in allHLG:
            t0 = t0t1[0]
            t1 = t0t1[1]
            slabel = '%d_%d_HLG%d' % (t0, t1, HLG)
            fname = os.path.join(dpath, 'df2go_62cols_ABD_HLGlarger_%s.csv.gz' % (slabel))
            print('- Loading: %s...' % (fname))
            df2go = pd.read_csv(fname)  #

            for strategy, sperc in zip(myStrategies, allperc):
                print('==================================================================================')
                print('=== Range: %d-%d HLG=%d Strategy:%s ==='%(t0,t1,HLG,strategy))
                print('==================================================================================')

                ## a) Filter by Strategy
                # sperc = 80 # 50% percentile
                df2gox = p5_helper_retrieveStrategy(df2go.copy(), strategy, sperc)

                if runDyCE:
                    ## b) Create input plan for engine
                    print('(*) OBS1: Use the "bin" time as the starttime, not the parent-order start time (to simplify)')
                    print('(*) OBS2: All bins included. Bin1 will be used to predict at Bin2 etc.')
                    # print('(*) OBS2: No prediction for the bin1. So, simply exclude all first bin (DyCE does not generate)...')
                    # print('(*) OBS3: Start time is shifted 5min back to generate prediction for the current time...')

                    inputfile = pd.DataFrame({
                        'token': df2gox['#key'] + '_bin' + df2gox['bin'].astype(str),
                        # 'token':df2go['tradeDate'].astype(str)+'_'+df2go['ticker']+'_'+df2go['clientMoniker']+'_bin'+df2go['bin'].astype(str)+'_size'+df2go['ExShareForBin'].astype(str),
                        'ISO': 'USA',
                        'date': df2gox['tradeDate'],  #
                        'ticker': df2gox['ticker'],
                        # 'starttime':((df2gox['bin']-1)*300+34200-300).astype(str)+'s', # SHIFT START TIME BY 5MIN FOR PREDICTION!
                        # 'endtime':df2gox['lastFillTime'].astype(str)+'s',   # [ ] CHECK WITH KONSTANTIN WHICH END TIME TO USE...
                        'starttime': ((df2gox['bin'] - 1) * 300 + 34200).astype(str) + 's',  # NO SHIFT
                        'endtime': '57600s',  # SET TO THE LAST BIN...
                        'startshare': 0,
                        'endshare': 100  # Use a fixed amount
                        # 'endshare':df2gox['ExShareForBin'] # [ ] CHECK WITH KONSTANTIN WHICH SIZE TO USE... CORRECT? Does it matter?
                    })
                    print('- Total number of loaded clusters:', inputfile.shape[0])

                    ## c) SPLIT BY WEEK AND SAVE
                    # spath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/Plans'
                    datalabel = slabel
                    # myweeks = [[1,7],[8,15],[16,23],[24,31]]
                    onlydates = inputfile.date.astype(str).apply(lambda s: s[-2:])
                    for i, week in enumerate(myweeks):
                        plan = inputfile[onlydates.astype(int).between(week[0], week[1])]
                        fname = os.path.join(planpath, 'input_ABD_HLGlarger_%s_%s_week%d.plan' % (datalabel, strategy, i + 1))
                        print('- Saving %s...' % fname)
                        plan.reset_index(drop=True).to_csv(fname, index=False, header=False)

                        ## d) DyCE: Run the input.plan
                    #     # newfe-s18
                    #     DyCEpath = "/local/data/itg/fe/prod/data1/jide-local/DyCE/ref/DyCE/bin/DyCE"
                    #     ppath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5'

                    #     # feres-s12
                    #     #DyCEpath = "/local/data/itg/fe/prod/data1/jide-local/MPO/DyCE/bin/DyCE"
                    #     #spath = '/local/data/itg/fe/prod/data1/jide-local/P5/Out'

                spath = os.path.join(ppath, 'Out')

                if runDyCE:
                    for i, week in enumerate(myweeks):
                        tic = time.time()
                        planfile = os.path.join(ppath, 'Plans',
                                                'input_ABD_HLGlarger_%s_%s_week%d.plan' % (datalabel, strategy, i + 1))
                        mylabel = 'ABD_HLGlarger_%s_%s_week%d' % (datalabel, strategy, i + 1)
                        cmd = f'{DyCEpath} -p {planfile} -S VWAP -onlyAnalytics -c -o {spath} -l {mylabel}'
                        try:
                            print('- Running DyCE: %s' % cmd)
                            subprocess.call(cmd, shell = True)
                            fileList = []
                            fileList.append(glob.glob(os.path.join(spath, 'intraday*%s.*' % mylabel)))
                            fileList.append(glob.glob(os.path.join(spath, 'basicData*%s.*' % mylabel)))
                            fileList.append(glob.glob(os.path.join(spath, 'costAndRisk*%s.*' % mylabel)))
                            fileList.append(glob.glob(os.path.join(spath, 'postProcess*%s.*' % mylabel)))
                            print('- Extra files removed...')
                            for f in fileList:
                                if f:
                                    subprocess.call('rm %s' % f[0], shell=True)
                            print('- Time taken to compute DyCE %dmin.' % ((time.time() - tic) // 60))
                            outpath = spath
                            try:
                                ji_wrapper_save_df2plot_volaPrediction_fromDf2go(df2gox.copy(), outpath, df2plotpath, mylabel)
                            except:
                                print('** Warning ** Failed to save df2plot.. Check DyCE output...')
                        except:
                            print('** Warning ** Failed to run DyCE:',cmd)
                            print('**  - df2plot not saved for:',mylabel)
                            print('**  - Check why...')
                else:
                    datalabel = slabel
                    outpath = spath
                    for i, week in enumerate(myweeks):
                        tic = time.time()
                        mylabel = 'ABD_HLGlarger_%s_%s_week%d' % (datalabel, strategy, i + 1)
                        #try:
                        ji_wrapper_save_df2plot_volaPrediction_fromDf2go(df2gox.copy(), outpath, df2plotpath,
                                                                             mylabel)
                        #except:
                        #    print('** Warning ** Failed to save df2plot.. Check DyCE output...')


            print('=========================== DONE ===================================================')


def p5_wrapper_runDyCE_save_df2plot_April5(allrange, allHLG, myStrategies=['SCHEDULED', 'DARK', 'IS', 'OPPORTUNISTIC'],
                                           allperc=[80, 71, 80, 80]):
    """Wrapper to
    - Load df2go
    - Generate inputpla
    - Run DyCE
    - Generate prediction and save df2plot"""

    # allrange = [[20180101, 20180131]]
    # allHLG = [0]

    ## SET STRATEGIES
    # myStrategies = ['SCHEDULED']
    # allperc = [80]
    # myStrategies = ['SCHEDULED','DARK','IS','OPPORTUNISTIC']
    # allperc = [80,71,80,80]

    ## SET DATE RANGES
    myweeks = [[1, 7], [8, 15], [16, 23], [24, 31]]
    # myweeks = [[1, 7]]

    ## SET PATHS
    dpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data'  # Where df2go is loaded from
    planpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/Plans'  # Where input plans will be saved
    df2plotpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data_df2plot'  # Where df2plot will be saved
    # newfe-s18: DyCE location and working directory
    DyCEpath = "/local/data/itg/fe/prod/data1/jide-local/DyCE/ref/DyCE/bin/DyCE"
    ppath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5'
    # feres-s12
    # DyCEpath = "/local/data/itg/fe/prod/data1/jide-local/MPO/DyCE/bin/DyCE"
    # spath = '/local/data/itg/fe/prod/data1/jide-local/P5/Out'

    ######## 1) Get df2go
    for t0t1 in allrange:
        for HLG in allHLG:
            t0 = t0t1[0]
            t1 = t0t1[1]
            slabel = '%d_%d_HLG%d' % (t0, t1, HLG)
            fname = os.path.join(dpath, 'df2go_62cols_ABD_HLGlarger_%s.csv.gz' % (slabel))
            print('- Loading: %s...' % (fname))
            df2go = pd.read_csv(fname)  #

            for strategy, sperc in zip(myStrategies, allperc):
                print('==================================================================================')
                print('=== Range: %d-%d HLG=%d Strategy:%s ===' % (t0, t1, HLG, strategy))
                print('==================================================================================')

                ## a) Filter by Strategy
                # sperc = 80 # 50% percentile
                df2gox = p5_helper_retrieveStrategy(df2go.copy(), strategy, sperc)

                ## b) Create input plan for engine
                print('(*) OBS1: Use the "bin" time as the starttime, not the parent-order start time (to simplify)')
                print('(*) OBS2: All bins included. Bin1 will be used to predict at Bin2 etc.')
                # print('(*) OBS2: No prediction for the bin1. So, simply exclude all first bin (DyCE does not generate)...')
                # print('(*) OBS3: Start time is shifted 5min back to generate prediction for the current time...')

                inputfile = pd.DataFrame({
                    'token': df2gox['#key'] + '_bin' + df2gox['bin'].astype(str),
                    # 'token':df2go['tradeDate'].astype(str)+'_'+df2go['ticker']+'_'+df2go['clientMoniker']+'_bin'+df2go['bin'].astype(str)+'_size'+df2go['ExShareForBin'].astype(str),
                    'ISO': 'USA',
                    'date': df2gox['tradeDate'],  #
                    'ticker': df2gox['ticker'],
                    # 'starttime':((df2gox['bin']-1)*300+34200-300).astype(str)+'s', # SHIFT START TIME BY 5MIN FOR PREDICTION!
                    # 'endtime':df2gox['lastFillTime'].astype(str)+'s',   # [ ] CHECK WITH KONSTANTIN WHICH END TIME TO USE...
                    'starttime': ((df2gox['bin'] - 1) * 300 + 34200).astype(str) + 's',  # NO SHIFT
                    'endtime': '57600s',  # SET TO THE LAST BIN...
                    'startshare': 0,
                    'endshare': 100  # Use a fixed amount
                    # 'endshare':df2gox['ExShareForBin'] # [ ] CHECK WITH KONSTANTIN WHICH SIZE TO USE... CORRECT? Does it matter?
                })
                print('- Total number of loaded clusters:', inputfile.shape[0])

                ## c) SPLIT BY WEEK AND SAVE
                # spath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/Plans'
                datalabel = slabel
                # myweeks = [[1,7],[8,15],[16,23],[24,31]]
                onlydates = inputfile.date.astype(str).apply(lambda s: s[-2:])
                for i, week in enumerate(myweeks):
                    plan = inputfile[onlydates.astype(int).between(week[0], week[1])]
                    fname = os.path.join(planpath,
                                         'input_ABD_HLGlarger_%s_%s_week%d.plan' % (datalabel, strategy, i + 1))
                    print('- Saving %s...' % fname)
                    plan.reset_index(drop=True).to_csv(fname, index=False, header=False)

                    ## d) DyCE: Run the input.plan
                #     # newfe-s18
                #     DyCEpath = "/local/data/itg/fe/prod/data1/jide-local/DyCE/ref/DyCE/bin/DyCE"
                #     ppath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5'

                #     # feres-s12
                #     #DyCEpath = "/local/data/itg/fe/prod/data1/jide-local/MPO/DyCE/bin/DyCE"
                #     #spath = '/local/data/itg/fe/prod/data1/jide-local/P5/Out'

                spath = os.path.join(ppath, 'Out')

                for i, week in enumerate(myweeks):
                    tic = time.time()
                    planfile = os.path.join(ppath, 'Plans',
                                            'input_ABD_HLGlarger_%s_%s_week%d.plan' % (datalabel, strategy, i + 1))
                    mylabel = 'ABD_HLGlarger_%s_%s_week%d' % (datalabel, strategy, i + 1)
                    cmd = f'{DyCEpath} -p {planfile} -S VWAP -onlyAnalytics -c -o {spath} -l {mylabel}'
                    try:
                        print('- Running DyCE: %s' % cmd)
                        subprocess.call(cmd, shell=True)
                        fileList = []
                        fileList.append(glob.glob(os.path.join(spath, 'intraday*%s.*' % mylabel)))
                        fileList.append(glob.glob(os.path.join(spath, 'basicData*%s.*' % mylabel)))
                        fileList.append(glob.glob(os.path.join(spath, 'costAndRisk*%s.*' % mylabel)))
                        fileList.append(glob.glob(os.path.join(spath, 'postProcess*%s.*' % mylabel)))
                        print('- Extra files removed...')
                        for f in fileList:
                            if f:
                                subprocess.call('rm %s' % f[0], shell=True)
                        print('- Time taken to compute DyCE %dmin.' % ((time.time() - tic) // 60))
                        outpath = spath
                        try:
                            ji_wrapper_save_df2plot_volaPrediction_fromDf2go(df2gox.copy(), outpath, df2plotpath,
                                                                             mylabel)
                        except:
                            print('** Warning ** Failed to save df2plot.. Check DyCE output...')
                    except:
                        print('** Warning ** Failed to run DyCE:', cmd)
                        print('**  - df2plot not saved for:', mylabel)
                        print('**  - Check why...')

            print('=========================== Done ===================================================')


def p5_helper_print_binStats_30min(dfx,label):
    print('************* Bin stats: %s **************'%label)
    print('* - Non-traded bins:%d'%(np.sum(dfx.FillNbrForBin ==0)))
    print('* - Traded bins:%d (%d%%)'%(np.sum(dfx.FillNbrForBin >0), 100*np.sum(dfx.FillNbrForBin >0)/dfx.shape[0]))
    nn = np.sum(dfx.FillNbrForBin >0)
    a = np.sum((dfx.firstTrade == 1) & (dfx.very1stTrade == 0) & (dfx.gapDuration.between(1, 2)))
    b = np.sum((dfx.firstTrade == 1) & (dfx.very1stTrade == 0) & (dfx.gapDuration.between(3, 13)))
    c = np.sum( (dfx.firstTrade == 1 & (dfx.very1stTrade == 0)) )
    print('* - 1st-tradedBins (exclusive):%d%% (short:%d%%, long:%d%%)' % (100*c/nn, 100*a/c, 100*b/c))
    print('* - Very 1st-traded bins:%d%%' % (100 * np.sum( (dfx.very1stTrade == 1) & (dfx.FillNbrForBin >0))/nn))
    print('* - Non 1st-traded bins:%d%%' % (100 * np.sum((dfx.firstTrade == 0) & (dfx.FillNbrForBin >0)) / nn))
    print('******************************************')

def p5_helper_print_binStats(dfx,label):
    print('************* Bin stats: %s **************'%label)
    print('* - Non-traded bins:%d'%(np.sum(dfx.FillNbrForBin ==0)))
    print('* - Traded bins:%d (%d%%)'%(np.sum(dfx.FillNbrForBin >0), 100*np.sum(dfx.FillNbrForBin >0)/dfx.shape[0]))
    nn = np.sum(dfx.FillNbrForBin >0)
    a = np.sum((dfx.firstTrade == 1) & (dfx.very1stTrade == 0) & (dfx.gapDuration.between(1, 3)))
    b = np.sum((dfx.firstTrade == 1) & (dfx.very1stTrade == 0) & (dfx.gapDuration.between(4, 77)))
    c = np.sum( (dfx.firstTrade == 1 & (dfx.very1stTrade == 0)) )
    print('* - 1st-tradedBins (exclusive):%d%% (short:%d%%, long:%d%%)' % (100*c/nn, 100*a/c, 100*b/c))
    print('* - Very 1st-traded bins:%d%%' % (100 * np.sum( (dfx.very1stTrade == 1) & (dfx.FillNbrForBin >0))/nn))
    print('* - Non 1st-traded bins:%d%%' % (100 * np.sum((dfx.firstTrade == 0) & (dfx.FillNbrForBin >0)) / nn))
    print('******************************************')


def p5_wrapper_generate_plots_Return_April26(df2plot,set2run,alllabels,lastBin=78,quantiles = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95], sizeVar = 'adjSizeShare',sizeLabel = 'adjSizeShare', condVarX = 'pred30min', condLabelX = 'predVolaSur30min', savepath = './', allsavelabel=None):
    """ Generate and save plots and tables for gapTrades

    - Previous:  def p5_wrapper_generate_plots_April16"""

    ## 2) Get masks
    df2plot = df2plot[df2plot.gapTrades == 1]
    allTradeLabels = []
    allmasks = []
    allTradeLabels.append('allGapTrades')
    allmasks.append((df2plot.gapTrades == 1))   # Will include everything
    # allTradeLabels.append('gapTrades(dist=1)')
    # allmasks.append((df2plot.gapTrades == 1) &  (df2plot.gapDistance == 1) )  # gap right after


    #quantiles = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
    if allsavelabel is None:
        allsavelabel = ['']*len(set2run) # in case it was not passed, it will create an empty string

    for rangeHLG, gLabel, savelabel in zip(set2run, alllabels, allsavelabel):
        print('\n========================================== Group: %s ==========================================' % gLabel)
        # IMPORTANT: SaveLabels should have the same order as allTradeLabels!
        allSaveLabels = []
        allSaveLabels.append('allGapTrades_%s' % savelabel)
        #allSaveLabels.append('gapTradesDist1_%s' % savelabel)

        for tmask, tradeLabel, sLabel in zip(allmasks, allTradeLabels, allSaveLabels):
            print('\n----------------------- Subgroup: %s ---' % (tradeLabel))

            # imask = (df2plot.HLG.isin(rangeHLG)) # & (df2plot.firstTrade!=1)
            imask = (df2plot.HLG.isin(rangeHLG)) & tmask
            print('- Mask size:', np.sum(imask))
            mylabel = '%s, %s' % (tradeLabel, gLabel)
            # To PDF
            pdffile = os.path.join(savepath,'plots_Return_x_Size_%s.pdf' % (sLabel))
            pdfobj = PdfPages(pdffile)

            condrange = 3
            print('----- Predicted volatility surprise -----')
            #condVar = 'pred30min'
            #condLabel = 'predVolaSur30min'
            #savelabel = '%s_%s_by_%s' % (tradeLabel,gLabel,condLabelX)
            savelabel = 'x_Size_by_%s_%s' % (condLabelX, sLabel)
            ji.helper_plot_Return_bySize_byGapDist_withCondition(df2plot[imask], sizeVar, sizeLabel, condVarX, condLabelX,
                                                         condrange, mylabel, lastBin, quantiles, savepath, savelabel,pdfobj)

            print('----- Realized volatility surprise -----')
            condVar = 'surVola'
            condLabel = 'realVolaSur'
            condrange = 3
            #savelabel = '%s_%s_by_%s' % (tradeLabel, gLabel, condLabel)
            savelabel = 'x_Size_by_%s_%s' % (condLabel, sLabel)
            ji.helper_plot_Return_bySize_byGapDist_withCondition(df2plot[imask], sizeVar, sizeLabel, condVar, condLabel,
                                                         condrange, mylabel, lastBin, quantiles, savepath, savelabel, pdfobj)

            print('- Figure saved in', pdffile)
            pdfobj.close()

def p5_wrapper_generate_plots_30min_April16(df2plot,set2run,alllabels,quantiles = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95], sizeVar = 'adjSizeShare',sizeLabel = 'adjSizeShare', condVarX = 'pred30min', condLabelX = 'predVolaSur30min', savepath = './', allsavelabel=None):
    """ Generate and save plots and tables
    - Previous:  def p5_wrapper_generate_plots_April3"""

    ## 2) Get masks
    df2plot = df2plot[df2plot.FillNbrForBin > 0]
    allTradeLabels = []
    allmasks = []
    allTradeLabels.append('All-trades')
    allmasks.append((df2plot.firstTrade == 1) | (df2plot.firstTrade != 1))  # Will include everything
    allTradeLabels.append('1st-trades (exclusive)')
    allmasks.append(df2plot.firstTrade == 1 & (df2plot.very1stTrade == 0))
    allTradeLabels.append('Very 1st-trades')
    allmasks.append(df2plot.very1stTrade == 1)
    allTradeLabels.append('Non 1st-trades')
    allmasks.append(df2plot.firstTrade == 0)
    allTradeLabels.append('1st-trades (dur:[1,2])')
    allmasks.append((df2plot.firstTrade == 1) & (df2plot.very1stTrade == 0) & (df2plot.gapDuration.between(1, 2)))
    allTradeLabels.append('1st-trades (dur:[3,13])')
    allmasks.append((df2plot.firstTrade == 1) & (df2plot.very1stTrade == 0) & (df2plot.gapDuration.between(3, 13)))

    ## 3) PLOT
    # dpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data'
    # df2plot1 = pd.read_csv(os.path.join(dpath,'df2plot_2017_HLGlarger_ABD_Q1Q2_HLG1.csv.gz'))
    #set2run = [[1]]
    #alllabels = ['HLG=1 (Q1Q2)']

    #quantiles = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
    if allsavelabel is None:
        allsavelabel = ['']*len(set2run) # in case it was not passed, it will create an empty string

    for rangeHLG, gLabel, savelabel in zip(set2run, alllabels, allsavelabel):
        print('\n========================================== Group: %s ==========================================' % gLabel)
        # IMPORTANT: SaveLabels should have the same order as allTradeLabels!
        allSaveLabels = []
        allSaveLabels.append('AllTradedBins_%s' % savelabel)
        allSaveLabels.append('1stTradedBinExclusive_%s' % savelabel)
        allSaveLabels.append('Very1stTradedBin_%s' % savelabel)
        allSaveLabels.append('Non1stTradedBin_%s' % savelabel)
        allSaveLabels.append('1stTradedBinDurShort_%s' % savelabel)
        allSaveLabels.append('1stTradedBinDurLong_%s' % savelabel)

        for tmask, tradeLabel, sLabel in zip(allmasks, allTradeLabels, allSaveLabels):
            print('\n----------------------- Subgroup: %s ---' % (tradeLabel))

            # imask = (df2plot.HLG.isin(rangeHLG)) # & (df2plot.firstTrade!=1)
            imask = (df2plot.HLG.isin(rangeHLG)) & tmask
            print('- Mask size:', np.sum(imask))
            mylabel = '%s, %s' % (tradeLabel, gLabel)
            # To PDF
            pdffile = os.path.join(savepath,'plots_CostReturn_x_Size_%s.pdf' % (sLabel))
            pdfobj = PdfPages(pdffile)

            # AdjShareSize
            # sizeVar = 'pred30min'
            # sizeVar = 'prate'
            # sizeLabel = 'Volatility Surprise (30min prediction)'

            #sizeVar = 'adjSizeShare'
            #sizeLabel = 'adjSizeShare'


            condrange = 3
            print('----- Predicted volatility surprise -----')
            #condVar = 'pred30min'
            #condLabel = 'predVolaSur30min'
            #savelabel = '%s_%s_by_%s' % (tradeLabel,gLabel,condLabelX)
            savelabel = 'x_Size_by_%s_%s' % (condLabelX, sLabel)
            ji.helper_plot_Cost_Return_byX_withCondition(df2plot[imask], sizeVar, sizeLabel, condVarX, condLabelX,
                                                         condrange, mylabel, quantiles, savepath, savelabel,pdfobj)

            print('----- Realized volatility surprise -----')
            condVar = 'surVola'
            condLabel = 'realVolaSur'
            condrange = 3
            #savelabel = '%s_%s_by_%s' % (tradeLabel, gLabel, condLabel)
            savelabel = 'x_Size_by_%s_%s' % (condLabel, sLabel)
            ji.helper_plot_Cost_Return_byX_withCondition(df2plot[imask], sizeVar, sizeLabel, condVar, condLabel,
                                                         condrange, mylabel, quantiles, savepath, savelabel, pdfobj)

            print('- Figure saved in', pdffile)
            pdfobj.close()

def p5_wrapper_generate_plots_April3(df2plot,set2run,alllabels,quantiles = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95], sizeVar = 'adjSizeShare',sizeLabel = 'adjSizeShare', condVarX = 'pred30min', condLabelX = 'predVolaSur30min', savepath = './', allsavelabel=None):

    ## 2) Get masks
    df2plot = df2plot[df2plot.FillNbrForBin > 0]
    allTradeLabels = []
    allmasks = []
    allTradeLabels.append('All-trades')
    allmasks.append((df2plot.firstTrade == 1) | (df2plot.firstTrade != 1))  # Will include everything
    allTradeLabels.append('1st-trades (exclusive)')
    allmasks.append(df2plot.firstTrade == 1 & (df2plot.very1stTrade == 0))
    allTradeLabels.append('Very 1st-trades')
    allmasks.append(df2plot.very1stTrade == 1)
    allTradeLabels.append('Non 1st-trades')
    allmasks.append(df2plot.firstTrade == 0)
    allTradeLabels.append('1st-trades (dur:[1,3])')
    allmasks.append((df2plot.firstTrade == 1) & (df2plot.very1stTrade == 0) & (df2plot.gapDuration.between(1, 3)))
    allTradeLabels.append('1st-trades (dur:[4,77])')
    allmasks.append((df2plot.firstTrade == 1) & (df2plot.very1stTrade == 0) & (df2plot.gapDuration.between(4, 77)))

    ## 3) PLOT
    # dpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data'
    # df2plot1 = pd.read_csv(os.path.join(dpath,'df2plot_2017_HLGlarger_ABD_Q1Q2_HLG1.csv.gz'))
    #set2run = [[1]]
    #alllabels = ['HLG=1 (Q1Q2)']

    #quantiles = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
    if allsavelabel is None:
        allsavelabel = ['']*len(set2run) # in case it was not passed, it will create an empty string

    for rangeHLG, gLabel, savelabel in zip(set2run, alllabels, allsavelabel):
        print('\n========================================== Group: %s ==========================================' % gLabel)
        # IMPORTANT: SaveLabels should have the same order as allTradeLabels!
        allSaveLabels = []
        allSaveLabels.append('AllTradedBins_%s' % savelabel)
        allSaveLabels.append('1stTradedBinExclusive_%s' % savelabel)
        allSaveLabels.append('Very1stTradedBin_%s' % savelabel)
        allSaveLabels.append('Non1stTradedBin_%s' % savelabel)
        allSaveLabels.append('1stTradedBinDurShort_%s' % savelabel)
        allSaveLabels.append('1stTradedBinDurLong_%s' % savelabel)

        for tmask, tradeLabel, sLabel in zip(allmasks, allTradeLabels, allSaveLabels):
            print('\n----------------------- Subgroup: %s ---' % (tradeLabel))

            # imask = (df2plot.HLG.isin(rangeHLG)) # & (df2plot.firstTrade!=1)
            imask = (df2plot.HLG.isin(rangeHLG)) & tmask
            print('- Mask size:', np.sum(imask))
            mylabel = '%s, %s' % (tradeLabel, gLabel)
            # To PDF
            pdffile = os.path.join(savepath,'plots_CostReturn_x_Size_%s.pdf' % (sLabel))
            pdfobj = PdfPages(pdffile)

            # AdjShareSize
            # sizeVar = 'pred30min'
            # sizeVar = 'prate'
            # sizeLabel = 'Volatility Surprise (30min prediction)'

            #sizeVar = 'adjSizeShare'
            #sizeLabel = 'adjSizeShare'


            condrange = 3
            print('----- Predicted volatility surprise -----')
            #condVar = 'pred30min'
            #condLabel = 'predVolaSur30min'
            #savelabel = '%s_%s_by_%s' % (tradeLabel,gLabel,condLabelX)
            savelabel = 'x_Size_by_%s_%s' % (condLabelX, sLabel)
            ji.helper_plot_Cost_Return_byX_withCondition(df2plot[imask], sizeVar, sizeLabel, condVarX, condLabelX,
                                                         condrange, mylabel, quantiles, savepath, savelabel,pdfobj)

            print('----- Realized volatility surprise -----')
            condVar = 'surVola'
            condLabel = 'realVolaSur'
            condrange = 3
            #savelabel = '%s_%s_by_%s' % (tradeLabel, gLabel, condLabel)
            savelabel = 'x_Size_by_%s_%s' % (condLabel, sLabel)
            ji.helper_plot_Cost_Return_byX_withCondition(df2plot[imask], sizeVar, sizeLabel, condVar, condLabel,
                                                         condrange, mylabel, quantiles, savepath, savelabel, pdfobj)

            print('- Figure saved in', pdffile)
            pdfobj.close()


def p5_wrapper_examine_CostxSize(df2go,costName,costLabel, plotHist=True, perc2exclude = 1, excludeZeroY = True, saveflag = None):
    """Wrapper to repeat a set of analyses."""
    # Create PDF if requested
    if saveflag is not None:
        pdfobj = matplotlib.backends.backend_pdf.PdfPages(
            os.path.join(saveflag.outDir, "fig_%s_x_Size_%s.pdf" %(costName,saveflag.label)))

    daytime = [22, 77]  # bin range to evaluate 1 (9:30).. 78(16:00)
    # daytime = [1, 78]  # bin range to evaluate 1 (9:30).. 78(16:00)

    ## 0) Make sure to exclude trades without client execution
    dfx = df2go[df2go.empCost != 0].copy()
    dfx = df2go[df2go.empCost != 0].copy()
    print('- Excluded bins without client execution: %d (%d%%)' % (
    np.sum(df2go.empCost == 0), 100 * np.sum(df2go.empCost == 0) / df2go.shape[0]))
    # Outlier
    #dfx = jifin.remove_outliers_df_down(dfx, percent=perc2exclude / 2, vars2rem=[costName])
    #dfx = jifin.remove_outliers_df_up(dfx, percent=perc2exclude / 2, vars2rem=[costName])
    dfx = jifin.remove_outliers_df_2tails(dfx, percent=perc2exclude, vars2rem=[costName])

    ## Commons parameters

    ## I) Realized Vola (HL)
    condName = 'realVola'
    cflag = {'nsplit':3, 'daytime':daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}   # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xprate(dfx, costName, costLabel, condName,cflag, pdfobj)

    ## II) Realized Vola Surprise (HL)
    condName = 'surVola'
    cflag = {'nsplit': 3, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}   # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xprate(dfx, costName, costLabel, condName, cflag, pdfobj)

    ## III) NormStyle
    condName = 'normStyle'
    #cflag = {'nsplit': 4, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}   # bin range to evaluate 1 (9:30).. 78(16:00)
    print('* Using fixed ranges for normStyle...')
    cflag = {'nsplit': [[-5,-0.07],[-0.07,0],[0,0.12],[0.12,4.1]], 'daytime': daytime, 'plotHist': plotHist,
             'excludeZeroY': excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)

    p5_helper2wrapper_xprate(dfx, costName, costLabel, condName, cflag, pdfobj)

    ## III.b) Time of the day
    condName = 'bin'
    cflag = {'nsplit': 4, 'daytime': [2, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}   # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xprate(dfx, costName, costLabel, condName, cflag, pdfobj)

    ## IV) binDuration
    condName = 'binDuration'
    # nsplit: is for the condVar
    condrange = [[1, 40], [40, 70], [70, 78]]
    cflag = {'nsplit': condrange, 'daytime': [2, 77], 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xprate(dfx, costName, costLabel, condName, cflag, pdfobj)

    ## V) Spread (time weighted)
    condName = 'spread'
    cflag = {'nsplit': 3, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xprate(dfx, costName, costLabel, condName, cflag, pdfobj)

    ## VI) Spread (time weighted)
    condName = 'TradeWeightedSpread'
    cflag = {'nsplit': 3, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}   # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xprate(dfx, costName, costLabel, condName, cflag, pdfobj)

    ## VII) FillNbr
    condName = 'FillNbrForBin'
    cflag = {'nsplit': 3, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}  # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xprate(dfx, costName, costLabel, condName, cflag, pdfobj)

    ## VIII) Strategy
    condName = 'algoStrategy'
    condrange = ['VWAP:100', 'Volume Participation:100', 'TWAP:100']
    cflag = {'nsplit': condrange, 'daytime': daytime, 'plotHist':plotHist, 'excludeZeroY':excludeZeroY}   # bin range to evaluate 1 (9:30).. 78(16:00)
    p5_helper2wrapper_xprate(dfx, costName, costLabel, condName, cflag, pdfobj)

    if saveflag is not None:
        pdfobj.close()

def p5_helper2wrapper_xprate(dfx,costName,costLabel,condName,dicflag, pdfobj = None):
    """nsplit: is for the condVar. It can be integer, a range or string values"""

    # Get flag vars
    flag = argparse.Namespace(**dicflag)
    daytime = flag.daytime
    nsplit = flag.nsplit
    excludeZeroY = flag.excludeZeroY

    allbins = dfx.bin
    # 0) Distribution
    plotHist = False
    if flag.plotHist:
        if type(nsplit)==int:
            plotHist = True
        elif type(nsplit[0]) != str:
            plotHist = True

    print('## Condition Variable:',condName)
    print(dfx[condName].describe())
    if plotHist:
        sns.distplot(dfx[condName],bins=15)
        plt.title('Distribution of %s' % condName)
        print('%s>0: %d bins(5min)' % (condName, np.sum(dfx[condName] > 0)))
        print('%s<0: %d bins(5min)' % (condName, np.sum(dfx[condName] < 0)))

    # 1) Define the value to examine
    Y = dfx[costName]
    size = dfx.prate # CHOOSE: normalized size
    size_buckets = [[0,0.025],[0.025,0.05],[0.05,0.1],[0.1,1]] # depends on the size metric
    print('- Note: Size buckets are pre-specified')
    # 2) CondVar
    condVar = dfx[condName] # CHOOSE
    #nsplit = 3 # number of categories for condVar
    #nsplit = [[1,40],[40,70],[70,78]]  # set manually
    allbins = dfx.bin
    daytime = [22,77] # bin range to evaluate 1 (9:30).. 78(16:00)
    #daytime = [1, 78]  # bin range to evaluate 1 (9:30).. 78(16:00)

    #daytime = [22,77] # bin range to evaluate 1 (9:30).. 78(16:00)
    #myflag = {'title':'EmpCost x Size','ylabel':'EmpCost(bps)', 'condlabel':condName,'xlabel':'prate'}
    myflag = {'title': '%s x Size' % costName, 'ylabel': '%s' % costLabel, 'condlabel': condName, 'xlabel': 'prate'}
    # 3) Plot
    p5_helper_plot_Y_byBuckets(Y,size,allbins,condVar,size_buckets,daytime,nsplit,myflag,excludeZeroY,pdfobj)


def p5_wrapper_examine_distribution(df2go,varName,condName,condrange, perc2exclude = 1):
    """ds1: variable of interest
       ds2: condition variable
       condrange: can be #split or a list of ranges"""
    #perc2exclude = 5 # percent each side
    # FILTER
    #dfx = df2go[df2go.empCost != 0].copy()
    #print('- Excluded bins without client execution: %d (%d%%)' % (np.sum(df2go.empCost == 0), 100 * np.sum(df2go.empCost == 0) / df2go.shape[0]))
    # Outlier
    dfx = df2go.copy()
    #dfx = jifin.remove_outliers_df_down(dfx, percent=perc2exclude/2, vars2rem=[varName])
    #dfx = jifin.remove_outliers_df_up(dfx, percent=perc2exclude / 2, vars2rem=[varName])
    dfx = jifin.remove_outliers_df_2tails(dfx, percent=perc2exclude, vars2rem=[varName])

    ## 1) Set dataframes
    ds1 = dfx[varName]
    ds2 = dfx[condName]
    ## 2) Create split if not given
    if type(condrange) == int:
        out, bins = pd.qcut(ds2, q=condrange, labels=False, retbins=True)
        myranges = [[bins[i], bins[i + 1]] for i in range(len(out.unique()))]
        ncond = condrange
    else:
        myranges = condrange
        ncond = len(condrange)
    ## 3) PLot
    allY = []
    rlabels = []
    fig = plt.figure(figsize=(20, 5))
    for i in range(1, ncond + 1):
        crange = myranges[i - 1]
        plt.subplot(1, ncond, i)
        # sns.distplot(ds1[ds2.between(crange[0],crange[1])]) # DOES NOT SHOW FREQUENCY...
        # ds1[ds2.between(crange[0],crange[1])].hist(bins=100) # not sure how to get the x and y...
        #print('crange[0]: ',crange[0])
        if type(crange[0]) != str:
            Y = ds1[ds2.between(crange[0], crange[1])].values
            print('- %s range#%d: %2.6f-%2.6f (nclusters=%d --> %2.2f)' %(condName, i, crange[0], crange[1], len(Y), len(Y)/ds1.shape[0]))
            rlabels.append('%d: %2.6f-%2.6f'%(i, crange[0], crange[1]))
        else:
            Y = ds1[ds2==crange].values
            print('- %s Value#%d: %s' %(condName, i, crange))
            rlabels.append('%d: %s' % (i, crange))

        y, x, _ = plt.hist(Y, bins=100)
        # print(dfx[condName].describe())
        # print('%s>0: %d bins(5min)' % (condName,np.sum(dfx[condName] > 0)))
        # print('%s<0: %d bins(5min)' % (condName,np.sum(dfx[condName] < 0)))
        if type(crange[0])!= str:
            plt.title('Hist of %s (Range#%d:%2.2f-%2.2f)' % (varName, i, crange[0], crange[1]), fontsize=12)
            #rlabels.append('Range#%d:%2.2f-%2.2f)'%(i, crange[0], crange[1]))
        else:
            plt.title('Hist of %s (Value#%d:%s)' % (varName, i, crange))
            #rlabels.append('Value#%d:%s)' % (i, crange))

        # plt.annotate('test',xy=(x.max()*0.3,0.95*y.max()))
        plt.text(x.max() * 0.1, 0.95 * y.max(), 'Mean:%2.4f' % Y.mean(), fontsize=16)
        plt.text(x.max() * 0.1, 0.85 * y.max(), 'Std:%2.2f' % Y.std(), fontsize=16)
        plt.text(x.max() * 0.1, 0.75 * y.max(), 'Count:%d' % len(Y), fontsize=16)
        plt.xlabel("%s\ncondition: %s" % (varName, condName), fontsize=16)
        plt.grid(True)
#        if type(crange[0]) != str:
#            print('- %s range#%d: %2.6f-%2.6f' %(condName, i, crange[0], crange[1]))
#            rlabels.append('%d: %2.6f-%2.6f'%(i, crange[0], crange[1]))
#        else:
#            print('- %s Value#%d: %s' %(condName, i, crange))
#            rlabels.append('%d: %s' % (i, crange))

        allY.append(Y)

    ## 4) Two-sample t-tests
    ipairs = list(itertools.combinations(range(len(allY)), 2))
    #print('- Two-sample t-test (%s) among different conditions (%s)'%(varName,condName))
    mytable = []
    for i, j in ipairs:
        v1 = allY[i]
        v2 = allY[j]
        t, p = stats.ttest_ind(v1, v2)
        #print('%s  x  %s --> t=%2.4f, p=%2.6f' % (rlabels[i], rlabels[j], t, p))
        mytable.append([rlabels[i].split(':')[0],rlabels[i].split(':')[1],
                        rlabels[j].split(':')[0], rlabels[j].split(':')[1],'%2.2f'%t,'%2.6f'%p])

    #F,p = stats.f_oneway(np.array(allY))
    #print('F-stat=%2.2f, p-value=%2.6f'%(F,p))
    return pd.DataFrame(mytable, columns=['Condition', '', 'Condition', '', 't-stat', 'p-value']), allY

"""
    ## 4) PLot Scatter
    fig = plt.figure(figsize=(20, 5))
    for i in range(1, ncond + 1):
        crange = myranges[i - 1]
        plt.subplot(1, ncond, i)
        #Y = ds1[ds2.between(crange[0], crange[1])].values

        x = ds2[ds2.between(crange[0], crange[1])].values
        y = ds1[ds2.between(crange[0], crange[1])].values
        print('- Range#%d, r=%2.4f, p=%2.6f (n=%d)' %(i,pearsonr(x, y)[0], pearsonr(x, y)[1], len(x)))
        plt.title('Range#%d'%(i))
        plt.plot(x, y, '.', alpha=0.08)
        plt.ylabel('%s' % varName)
        plt.xlabel('%s' % condName)
"""

def p5_wrapper_loadClusters_saveInputfiles(datalabel,dpath,splanpath,DyCEpath,p5path,nsplit=5):
    #datalabel = '2017_HLGlarger_ABD_Q1Q2_HLG0'

    # 1) LOAD PREVIOUS DATA AND ADD (TO SAVE TIME)
    # This dataset already computed
    # - volume-related values
    # - F+G
    tic = time.time()
    print('- Loading %s...'%datalabel)
    df2go = pd.read_csv(os.path.join(dpath,'df2go_62cols_%s.csv.gz'%datalabel)) #
    df2go = df2go.drop(columns=['Unnamed: 0'])
    print('Time taken to load the file %d seconds...'%(time.time()-tic))
    print(df2go.head())

    # 2) CREATE INPUT FILE
    ## Create input plan for engine
    print('(*) OBS1: Use the "bin" time as the starttime, not the parent-order start time (to simplify)')
    print('(*) OBS2: All bins included. Bin1 will be used to predict at Bin2 etc.')
    #print('(*) OBS2: No prediction for the bin1. So, simply exclude all first bin (DyCE does not generate)...')
    #print('(*) OBS3: Start time is shifted 5min back to generate prediction for the current time...')

    #df2gox = df2go[df2go.bin != 1]
    df2gox = df2go # now include all bins
    inputfile = pd.DataFrame({
            'token':df2gox['#key']+'_bin'+df2gox['bin'].astype(str),
            #'token':df2go['tradeDate'].astype(str)+'_'+df2go['ticker']+'_'+df2go['clientMoniker']+'_bin'+df2go['bin'].astype(str)+'_size'+df2go['ExShareForBin'].astype(str),
            'ISO':'USA',
            'date':df2gox['tradeDate'], #
            'ticker':df2gox['ticker'],
            #'starttime':((df2gox['bin']-1)*300+34200-300).astype(str)+'s', # SHIFT START TIME BY 5MIN FOR PREDICTION!
            #'endtime':df2gox['lastFillTime'].astype(str)+'s',   # [ ] CHECK WITH KONSTANTIN WHICH END TIME TO USE...
            'starttime':((df2gox['bin']-1)*300+34200).astype(str)+'s', # NO SHIFT
            'endtime':'57600s',   # SET TO THE LAST BIN...
            'startshare':0,
            'endshare':100 # Use a fixed amount
            #'endshare':df2gox['ExShareForBin'] # [ ] CHECK WITH KONSTANTIN WHICH SIZE TO USE... CORRECT? Does it matter?
        })
    print('- Total number of loaded clusters:',inputfile.shape[0])

    # 3) SPLIT AND SAVE PLANS
    ## SPLIT AND SAVE ------------> [ ] Copy to feres-s12 to run DyCE
    #nsplit = 5
    for i,plan in enumerate(np.array_split(inputfile,nsplit)):
        spath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/Plans'
        plan.reset_index(drop=True).to_csv(os.path.join(splanpath,'input_%s_part%d.plan'%(datalabel,i+1)),index=False,header=False)

    ## 3) DyCE: Run the input.plan

    # newfe-s18
    #DyCEpath = "/local/data/itg/fe/prod/data1/jide-local/DyCE/ref/DyCE/bin/DyCE"

    # feres-s12
    #DyCEpath = "/local/data/itg/fe/prod/data1/jide-local/MPO/DyCE/bin/DyCE"
    #spath = '/local/data/itg/fe/prod/data1/jide-local/P5/Out'

    outpath = os.path.join(p5path,'Out')
    for i in range(nsplit):
        planfile = os.path.join(p5path,'Plans','input_%s_part%d.plan'%(datalabel,i+1))
        mylabel = '%s_part%d'%(datalabel,i+1)
        cmd = f'{DyCEpath} -p {planfile} -S VWAP -onlyAnalytics -c -o {outpath} -l {mylabel} &'
        print(cmd)

def p5_helper_plot_Y_byBuckets(Y,sizex,allbins,condVar,size_buckets,daytime,nsplit,myflag, excludeZeroY=True, pdfobj=None):
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
    if excludeZeroY:
        print('* NOTE: Excluded Y equal to ZERO: %d (out %d)'%(np.sum(Y == 0),len(Y)))
        imask = (Y != 0)
        Y = Y[imask]
        size = sizex[imask]
        condVar = condVar[imask]
        allbins = allbins[imask]
    else:
        size = sizex

    # If no size_buckets is specified, only the number of buckets
    if type(size_buckets)==int:
        # Split condVar into bins (equally distributed)
        #out, bins = pd.qcut(size, q=size_buckets, labels=False, retbins=True)
        print('Size_bucket(before):',size_buckets)
        out, bins = pd.qcut(size, q=size_buckets, labels=False, retbins=True) # Exclude no Exec
        size_buckets = [[float('%2.6f'%bins[i]), float('%2.6f'%bins[i + 1])] for i in range(len(out.unique()))]
        print('* Note: size bucket created before filtering by daytime. Thus, will not be exactly equal after filtering...')

    flag = argparse.Namespace(**myflag)
    # Add daytime info to title (assume 78 bins of 5 min)
    # time.strftime("%H:%M:%S", time.gmtime(0))
    usStart = 34200 # 9:30 in seconds
    t0 = time.strftime("%H:%M", time.gmtime((daytime[0]-1)*5*60+usStart)) # 5min bins
    t1 = time.strftime("%H:%M", time.gmtime((daytime[1])*5*60+usStart))
    flag.title = flag.title+' (Time GMT:%s-%s)'%(t0,t1)

    if type(nsplit)==int:
        # Split condVar into bins (equally distributed)
        #out,bins = pd.qcut(condVar,q=nsplit,labels=False,retbins= True)
        #print('nsplit:', nsplit)
        #print('condVar(unique):', condVar.unique())
        print('* Note: condition bucket created before filtering by daytime. Thus, will not be exactly equal after filtering...')
        try:
            out, bins = pd.qcut(condVar, q=nsplit, labels=False, retbins=True) # exclude zeros of Y (no execution...)
        except:
            try:
                print("** Repeated bin edges with nsplit", nsplit, ". Try nsplit=3..")
                out, bins = pd.qcut(condVar, q=4, labels=False,
                                    retbins=True)  # exclude zeros of Y (no execution...)
            except:
                print("** Removed duplicated bin edges. Got repeated bin edges with nsplit",nsplit,"...")
                out, bins = pd.qcut(condVar, q=nsplit,duplicates = 'drop', labels=False, retbins=True)  # exclude zeros of Y (no execution...)
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
    #print('ncond+1:',ncond+1)
    #print('len(mybins):', len(mybins))
    #for i in range(1, ncond + 1):
    for i in range(1, len(mybins) + 1):
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
    ji.plot_bars_buckets_withTable(yall,ystd, size_buckets, binlabels,flag,others,pdfobj)

def ji_wrapper_create_compute_allFeatures_lab(dfalgo):
    """Wrapper to create and compute all features, including EmpCost and volatility etc.
    It will return a df2go."""
    tic0 = time.time()
    # 0) Create df2go
    # toInclude = ['itgID[4]','ticker[5]','sign[7]','tradeDate[17]','mddv[32]','histVola[34]']
    #toInclude = ['itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]', 'mddv[32]', 'histVola[34]', 'openPrice[35]']
    #toInclude = ['itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]', 'mddv[32]', 'histVola[34]', 'openPrice[35]','algoStrategy[48]']
    toInclude = ['clientMoniker[2]', 'itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]','firstFillBin[22]','lastFillBin[23]','relSize[28]','lg[30]','HLG[31]','mddv[32]', 'histVola[34]',
                 'openPrice[35]', 'algoStrategy[48]']

    df2go, df2go_before = ji_helper_create_df2go(dfalgo, toInclude)
    print('- Created df2go:', df2go.shape)

    # 1) Add list of dailyDOB
    tic = time.time()
    #mymeasures = ['midQuote', 'LowPrice', 'HighPrice', 'vwap', 'spread', 'TradeWeightedSpread', 'shareVolume']
    mymeasures = ['midQuote']
    for mymeasure in mymeasures:
        df2go = ji_helper_add_dailyDOB_in_df2go(df2go, df2go_before, mymeasure)
        print('... Added', mymeasure, 'in df2go:', df2go.shape)
    print('... DailyDOB loaded and added in %2.4f seconds.\n' % (time.time() - tic))

    # 5) Add the benchmark price, i.e. the previous bin midquote MQ(j-1)
    df2go = add_priceBM_in_df2go_fast(df2go)

    # 6) Compute the Empirical Cost
    # 6.1) Handle missing bmPrice (rare but happens when MQ price is missing for first bin of the ...)
    df2go = fill_missing_priceBM(df2go)

    return df2go, df2go_before

def ji_wrapper_create_compute_allFeatures_COMPACT(dfalgo, whichDB='RED', cleanMissing=True):
    """Wrapper to create and compute all features, including EmpCost and volatility etc.
    It will return a df2go."""
    tic0 = time.time()
    # 0) Create df2go
    # toInclude = ['itgID[4]','ticker[5]','sign[7]','tradeDate[17]','mddv[32]','histVola[34]']
    #toInclude = ['itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]', 'mddv[32]', 'histVola[34]', 'openPrice[35]']
    #toInclude = ['itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]', 'mddv[32]', 'histVola[34]', 'openPrice[35]','algoStrategy[48]']

    if whichDB == 'RED':
        toInclude = ['#key[1]', 'itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]','firstFillTime[19]','lastFillTime[20]','arriveBin[21]','firstFillBin[22]','lastFillBin[23]','relSize[28]','lg[30]','HLG[31]','mddv[32]', 'histVola[34]',
                 'openPrice[35]','closeTm1[38]' ,'fracMO[47]','algoStrategy[48]','algoType[49]']
        df2go, df2go_before = ji_helper_create_df2go(dfalgo, toInclude)
        print('- Created df2go:', df2go.shape)
    elif whichDB=='ABD':
        toInclude = ['#key[1]', 'itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]','FirstFillTime[19]','endTime[20]','arriveBin[21]', 'FirstFillBin[22]',
                 'endBin[23]', 'relSize[28]', 'lg[30]', 'HLG[31]', 'mddv[32]', 'histVola[34]', 'openPrice[35]',
                 'closeTm1[38]', 'algoStrategy[48]','algoType[49]','SpreadCapture[65]']
        df2go, df2go_before = ji_helper_create_df2go_ABD(dfalgo, toInclude)
        print('- Created df2go:', df2go.shape)
    else:
        error('whichDB not recognized...')

    # 1) Add list of dailyDOB
    tic = time.time()
    mymeasures = ['midQuote', 'LowPrice', 'HighPrice', 'vwap', 'spread', 'shareVolume']
    for mymeasure in mymeasures:
        df2go = ji_helper_add_dailyDOB_in_df2go(df2go, df2go_before, mymeasure)
        print('... Added', mymeasure, 'in df2go:', df2go.shape)
    print('... DailyDOB loaded and added in %2.4f seconds.\n' % (time.time() - tic))

    # 2) Add historical Volatility (weights)
    tic = time.time()
    df2go = ji_helper_add_histVola_in_df2go(df2go, df2go_before)
    print('... Historical volatility loaded in %2.4f seconds.' % (time.time() - tic))
    # 2.5) Add historical Volume
    tic = time.time()
    df2go = ji_helper_add_histVolu_in_df2go(df2go, df2go_before)
    print('... Historical volume loaded in %2.4f seconds.' % (time.time() - tic))

    # 3) Estimate realized volatility
    tic = time.time()
    print('=== Computing realized volatility...')
    #df2go['realVola'] = np.vectorize(jifin.compute_vola_maxmin)(df2go['HighPrice'], df2go['LowPrice'],
    df2go['realVola'] = np.vectorize(jifin.compute_vola_maxmin_silent)(df2go['HighPrice'], df2go['LowPrice'],
                                                              df2go['midQuote'], df2go['midQuote'])
    if df2go['realVola'].isnull().values.any():
        print('(*) Note: Unchanged price set to 0.01 and division by zero to NaN...')
    print('... MaxMin Volatility computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))

    # 4) Compute surprise of volatility
    tic = time.time()
    df2go['surVola'] = np.vectorize(jifin.my_divide)(df2go['realVola'], df2go['histVola5min'])
    print('... Volatility surprise computed in %2.4f seconds (using vectorize).' % (time.time() - tic))

    ## DEBUGGING
    print('- Percent of zero realVola: %2.2f' % (np.sum(df2go.realVola == 0) / df2go.shape[0]))
    if (np.sum(df2go.realVola == 0) / df2go.shape[0])>0.9:
        df2go.head(2000).to_csv('0_df2go_PROBLEM.csv')
        error('ERROR!')

    # 5) Add the benchmark price, i.e. the previous bin midquote MQ(j-1)
    df2go = add_priceBM_in_df2go_fast(df2go)

    # 6) Compute the Empirical Cost
    # 6.1) Handle missing bmPrice (rare but happens when MQ price is missing for first bin of the ...)
    df2go = fill_missing_priceBM(df2go,silent=True)
    # 6.1.1) Fill null after all...
    if np.sum(df2go.priceBM.isnull()) > 0:
        print('- Filling %d null priceBM with zeros...' % (np.sum(df2go.priceBM.isnull())))
        df2go.loc[df2go.priceBM.isnull(), 'priceBM'] = 0
    # 6.2) Compute
    tic = time.time()
    # obs: vectorize more efficient than using df.apply()
    df2go['empCost'] = np.vectorize(jifin.compute_empCost_silent)(df2go['ExPriceForBin'], df2go['priceBM'],
                                                         df2go['sign'])
    nnull = np.sum(df2go.empCost.isnull())
    if nnull > 0:
        print('- Number of empCost filled with NaN: %d...' % nnull)
    print('... Empirical Cost computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))
    #df2go.to_csv('df2go_42cols_Step6.csv')

    # 6.3) Compute F ang G components
    tic = time.time()
    # obs: vectorize more efficient than using df.apply()
    try:
        df2go['Fcost'] = np.vectorize(jifin.compute_F)(df2go['midQuote'], df2go['priceBM'], df2go['vwap'],
                                                               df2go['sign'])
        df2go['Gcost'] = np.vectorize(jifin.compute_G)(df2go['midQuote'], df2go['priceBM'], df2go['vwap'],
                                                       df2go['sign'])
        print('... F and G costs computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))
    except:
        print('- PROBLEM: computing F/Gcost. Check 0_df2go_step7.csv...')
        df2go.to_csv('0_df2go_step7.csv.gz',compression='gzip')
        error('- ERROR ...')

    # 7) Compute participation rate
    tic = time.time()
    df2go['prate'] = np.vectorize(jifin.compute_prate_silent)(df2go['ExShareForBin'], df2go['shareVolume'])
    # Check
    nn = np.sum(df2go.prate==1)
    if nn > 0:
        print('- Number of prate set to 1 when (exShare > mktVol): %d...' % nn)
    print('... Computed participation rate in %2.4f seconds (using vectorizer).' % (time.time() - tic))

    # 8) Compute Style
    print('=== Computing Style...')
    tic = time.time()
    df2go['style'] = np.vectorize(jifin.compute_style_silent)(df2go['ExPriceForBin'], df2go['vwap']
                                                     , df2go['priceBM'], df2go['prate'], df2go['sign'])
    # Check
    ncheck = np.sum(df2go.style==-123456)
    if ncheck > 0:
        print('- (*) Note: Number of prate=1 where Style was filled with zeros: %d...' %ncheck)
        df2go.loc[df2go.style==-123456, 'style'] = 0.0
    ncheck = np.sum(df2go.style == -1234567)
    if ncheck > 0:
        print('- (*) Note: Number of bmPrice=0 where Style was filled with zeros: %d...' % ncheck)
        df2go.loc[df2go.style == -1234567, 'style'] = 0.0
    print('... Style computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))

    # 9) Compute Normalized Style
    tic = time.time()
    df2go['normStyle'] = np.vectorize(jifin.compute_style_normalized_silent)(df2go['ExPriceForBin'], df2go['vwap']
                                                                      , df2go['HighPrice'], df2go['LowPrice'],
                                                                      df2go['prate'], df2go['sign'])
    # Check
    ncheck = np.sum(df2go.normStyle == -123456)
    if ncheck > 0:
        print('- (*) Note: Number of prate=1 where Style was filled with zeros: %d...' % ncheck)
        df2go.loc[df2go.normStyle == -123456, 'style'] = 0.0
    print('... Style (normalized) computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))

    # # 10) Add quadratic volatility from the 15sec MQ
    # tic = time.time()
    # df2go = ji_helper_add_quadVola_stdVola_15Sec_in_df2go(df2go, df2go_before)
    # print('... Quad and Std volatility computed in %2.4f seconds (takes time to load...).' % (time.time() - tic))
    #
    # # 11) Compute and add the cumulative and last 30 min volatilities (HL and quadVola)
    # tic = time.time()
    # df2go = add_last30_cum_vola(df2go)
    # print('... Cumulative and last30min volatilities (HL,quadVola) computed in %2.4f seconds.' % (time.time() - tic))
    #
    # # 12) Add cumulative Return from the 15sec MQ
    # tic = time.time()
    # df2go = ji_helper_add_cumReturn15sec_in_df2go(df2go, df2go_before)
    # print('... Cumulative return computed in %2.4f seconds (takes time to load...).' % (time.time() - tic))

    # 13) Add Return of the 5min bin
    tic = time.time()
    df2go = ji_helper_add_Return_in_df2go(df2go, silent=True)
    print('... Return and signedReturn computed in %2.4f seconds (takes time to load...).' % (time.time() - tic))

    # 14) Add volume-related
    # # Add MM normalization costs
    # df2go = ji_wrapper_add_MaxMinNormCost(df2go)
    print('=== Adding dollarVolume, surVolume and FGCost...')
    df2go['dollarVolume'] = df2go.shareVolume * df2go.vwap / 1000000  # in million
    df2go['surVolume'] = df2go['dollarVolume'] / df2go.histVolu5min
    df2go['FGcost'] = df2go.Fcost + df2go.Gcost

    # 15) Add First Trade indicator
    print('=== Adding 1stTrade indicator...')
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

    # 16) Add additional feats
    print('=== Adding adjusted values due to gap after trade for: historicals, returns, costs...')
    df2go = ji_get_adjusted_milanBM_historicals(df2go)
    df2go = ji_compute_adjusted_Returns_surprises(df2go)
    df2go = ji_compute_normalized_CostReturn(df2go)

    ## DEBUGGING
    print('- Number of zero realVola: %2.2f' % (np.sum(df2go.realVola == 0) / df2go.shape[0]))
    if (np.sum(df2go.realVola == 0) / df2go.shape[0])>0.9:
        df2go.head(2000).to_csv('0_df2go_PROBLEM.csv')
        error('ERROR!')


    print('\n**(TOTAL) df2go created and computed in %2.4f seconds.' % (time.time() - tic0))
    print('******************************************************\n')

    return df2go

def ji_wrapper_create_compute_allFeatures_COMPACT_debug(dfalgo, whichDB='RED', cleanMissing=True):
    """Wrapper to create and compute all features, including EmpCost and volatility etc.
    It will return a df2go."""
    tic0 = time.time()
    # 0) Create df2go
    # toInclude = ['itgID[4]','ticker[5]','sign[7]','tradeDate[17]','mddv[32]','histVola[34]']
    #toInclude = ['itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]', 'mddv[32]', 'histVola[34]', 'openPrice[35]']
    #toInclude = ['itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]', 'mddv[32]', 'histVola[34]', 'openPrice[35]','algoStrategy[48]']

    if whichDB == 'RED':
        toInclude = ['#key[1]', 'itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]','firstFillTime[19]','lastFillTime[20]','arriveBin[21]','firstFillBin[22]','lastFillBin[23]','relSize[28]','lg[30]','HLG[31]','mddv[32]', 'histVola[34]',
                 'openPrice[35]','closeTm1[38]' ,'fracMO[47]','algoStrategy[48]','algoType[49]']
        df2go, df2go_before = ji_helper_create_df2go(dfalgo, toInclude)
        print('- Created df2go:', df2go.shape)
    elif whichDB=='ABD':
        toInclude = ['#key[1]', 'itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]','FirstFillTime[19]','endTime[20]','arriveBin[21]', 'FirstFillBin[22]',
                 'endBin[23]', 'relSize[28]', 'lg[30]', 'HLG[31]', 'mddv[32]', 'histVola[34]', 'openPrice[35]',
                 'closeTm1[38]', 'algoStrategy[48]','algoType[49]','SpreadCapture[65]']
        df2go, df2go_before = ji_helper_create_df2go_ABD(dfalgo, toInclude)
        print('- Created df2go:', df2go.shape)
    else:
        error('whichDB not recognized...')

    # 1) Add list of dailyDOB
    tic = time.time()
    #mymeasures = ['midQuote', 'LowPrice', 'HighPrice', 'vwap', 'spread', 'shareVolume']
    mymeasures = ['midQuote']
    for mymeasure in mymeasures:
        df2go = ji_helper_add_dailyDOB_in_df2go_debug(df2go, df2go_before, mymeasure)
        print('... Added', mymeasure, 'in df2go:', df2go.shape)
    print('... DailyDOB loaded and added in %2.4f seconds.\n' % (time.time() - tic))

    # # 2) Add historical Volatility (weights)
    # tic = time.time()
    # df2go = ji_helper_add_histVola_in_df2go(df2go, df2go_before)
    # print('... Historical volatility loaded in %2.4f seconds.' % (time.time() - tic))
    # # 2.5) Add historical Volume
    # tic = time.time()
    # df2go = ji_helper_add_histVolu_in_df2go(df2go, df2go_before)
    # print('... Historical volume loaded in %2.4f seconds.' % (time.time() - tic))
    #
    # # 3) Estimate realized volatility
    # tic = time.time()
    # df2go['realVola'] = np.vectorize(jifin.compute_vola_maxmin)(df2go['HighPrice'], df2go['LowPrice'],
    #                                                           df2go['midQuote'], df2go['midQuote'])
    # print('... MaxMin Volatility computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))
    #
    # # 4) Compute surprise of volatility
    # tic = time.time()
    # df2go['surVola'] = np.vectorize(jifin.my_divide)(df2go['realVola'], df2go['histVola5min'])
    # print('... Volatility surprise computed in %2.4f seconds (using vectorize).' % (time.time() - tic))
    #
    # ## DEBUGGING
    # print('- Number of zero realVola: %2.2f' % (np.sum(df2go.realVola == 0) / df2go.shape[0]))
    # if (np.sum(df2go.realVola == 0) / df2go.shape[0])>0.9:
    #     df2go.head(2000).to_csv('0_df2go_PROBLEM.csv')
    #     error('ERROR!')
    #
    # # 5) Add the benchmark price, i.e. the previous bin midquote MQ(j-1)
    # df2go = add_priceBM_in_df2go_fast(df2go)
    #
    # # 6) Compute the Empirical Cost
    # # 6.1) Handle missing bmPrice (rare but happens when MQ price is missing for first bin of the ...)
    # df2go = fill_missing_priceBM(df2go)
    # # 6.1.1) Fill null after all...
    # if np.sum(df2go.priceBM.isnull()) > 0:
    #     print('- Filling %d null priceBM with zeros...' % (np.sum(df2go.priceBM.isnull())))
    #     df2go.loc[df2go.priceBM.isnull(), 'priceBM'] = 0
    # # 6.2) Compute
    # tic = time.time()
    # # obs: vectorize more efficient than using df.apply()
    # df2go['empCost'] = np.vectorize(jifin.compute_empCost)(df2go['ExPriceForBin'], df2go['priceBM'],
    #                                                      df2go['sign'])
    # print('... Empirical Cost computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))
    # #df2go.to_csv('df2go_42cols_Step6.csv')
    #
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
    #
    # # 7) Compute participation rate
    # tic = time.time()
    # df2go['prate'] = np.vectorize(jifin.compute_prate)(df2go['ExShareForBin'], df2go['shareVolume'])
    # print('... Computed participation rate in %2.4f seconds (using vectorizer).' % (time.time() - tic))
    #
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
    #
    # # 13) Add Return of the 5min bin
    # tic = time.time()
    # df2go = ji_helper_add_Return_in_df2go(df2go)
    # print('... Return and signedReturn computed in %2.4f seconds (takes time to load...).' % (time.time() - tic))
    #
    # # 14) Add volume-related
    # # Add MM normalization costs
    # df2go = ji_wrapper_add_MaxMinNormCost(df2go)
    # df2go['dollarVolume'] = df2go.shareVolume * df2go.vwap / 1000000  # in million
    # df2go['surVolume'] = df2go['dollarVolume'] / df2go.histVolu5min
    # df2go['FGcost'] = df2go.Fcost + df2go.Gcost
    #
    # # 15) Add First Trade indicator
    # execP = df2go['ExPriceForBin'].values
    # firstTrade = []
    # firstTrade.append(0)  # initialize first bin
    # for i in range(len(execP) - 1):
    #     if (execP[i] == 0) & (execP[i + 1] != 0):
    #         # print('execP[%d]=%2.4f, val=%2.4f'%(i,execP[i],execP[i+1]))
    #         firstTrade.append(1)
    #     else:
    #         firstTrade.append(0)
    # df2go['firstTrade'] = np.array(firstTrade)
    # # Correct the first bin: if there is trade, it is the first trade of the day
    # df2go.loc[(df2go.bin == 1) & (df2go.ExPriceForBin != 0), 'firstTrade'] = 1
    # print('- First trade indicator added.')
    #
    # # 16) Add additional feats
    # df2go = ji_get_adjusted_milanBM_historicals(df2go)
    # df2go = ji_compute_adjusted_Returns_surprises(df2go)
    # df2go = ji_compute_normalized_CostReturn(df2go)
    #
    # ## DEBUGGING
    # print('- Number of zero realVola: %2.2f' % (np.sum(df2go.realVola == 0) / df2go.shape[0]))
    # if (np.sum(df2go.realVola == 0) / df2go.shape[0])>0.9:
    #     df2go.head(2000).to_csv('0_df2go_PROBLEM.csv')
    #     error('ERROR!')


    print('\n**(TOTAL) df2go created and computed in %2.4f seconds.' % (time.time() - tic0))
    print('******************************************************\n')

    return df2go

def ji_p5_add_nonTradeBinTypes(dfx,lastBin=78):
    # 1) Create: nonTraded
    dfx['nonTraded'] = np.zeros(dfx.shape[0])
    imask = (dfx.FillNbrForBin == 0)
    dfx.loc[imask, 'nonTraded'] = 1
    # 2) auxiliary
    dfx['shiftplus'] = dfx.nonTraded.shift(1)
    dfx['shiftminus'] = dfx.nonTraded.shift(-1)
    # 3) Create: afterTrades and beforeTrades
    dfx['afterTrades'] = np.zeros(dfx.shape[0])
    imask = (dfx.nonTraded == 1) & (dfx.shiftplus == 0) & (dfx.bin != 1)
    dfx.loc[imask, 'afterTrades'] = 1
    dfx['beforeTrades'] = np.zeros(dfx.shape[0])
    imask = (dfx.nonTraded == 1) & (dfx.shiftminus == 0) & (dfx.bin != lastBin)
    dfx.loc[imask, 'beforeTrades'] = 1
    # # 4) Create: gapTrades - nonTraded bins that happens within the parent order horizon
    # dfx['gapTrades'] = dfx['nonTraded'].values

    # 4) Create: lastTraded
    dfx['lastTraded'] = np.zeros(dfx.shape[0])
    imask = (dfx.nonTraded == 0) & (dfx.shiftminus == 1)
    dfx.loc[imask, 'lastTraded'] = 1
    # 5) Clean
    dfx.drop(columns=['shiftplus', 'shiftminus'], inplace=True)

    return dfx

def ji_wrapper_create_compute_allFeatures(dfalgo, whichDB='RED', cleanMissing=True):
    """Wrapper to create and compute all features, including EmpCost and volatility etc.
    It will return a df2go."""
    tic0 = time.time()
    # 0) Create df2go
    # toInclude = ['itgID[4]','ticker[5]','sign[7]','tradeDate[17]','mddv[32]','histVola[34]']
    #toInclude = ['itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]', 'mddv[32]', 'histVola[34]', 'openPrice[35]']
    #toInclude = ['itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]', 'mddv[32]', 'histVola[34]', 'openPrice[35]','algoStrategy[48]']

    if whichDB == 'RED':
        toInclude = ['#key[1]', 'itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]','firstFillTime[19]','lastFillTime[20]','arriveBin[21]','firstFillBin[22]','lastFillBin[23]','relSize[28]','lg[30]','HLG[31]','mddv[32]', 'histVola[34]',
                 'openPrice[35]','closeTm1[38]' ,'algoStrategy[48]']
        df2go, df2go_before = ji_helper_create_df2go(dfalgo, toInclude)
        print('- Created df2go:', df2go.shape)
    elif whichDB=='ABD':
        toInclude = ['#key[1]', 'itgID[4]', 'ticker[5]', 'sign[7]', 'tradeDate[17]','firstFillTime[19]','lastFillTime[20]','arriveBin[21]', 'FirstFillBin[22]',
                 'endBin[23]', 'relSize[28]', 'lg[30]', 'HLG[31]', 'mddv[32]', 'histVola[34]', 'openPrice[35]',
                 'closeTm1[38]', 'algoStrategy[48]']
        df2go, df2go_before = ji_helper_create_df2go_ABD(dfalgo, toInclude)
        print('- Created df2go:', df2go.shape)
    else:
        error('whichDB not recognized...')

    # 1) Add list of dailyDOB
    tic = time.time()
    mymeasures = ['midQuote', 'LowPrice', 'HighPrice', 'vwap', 'spread', 'TradeWeightedSpread', 'shareVolume']
    for mymeasure in mymeasures:
        df2go = ji_helper_add_dailyDOB_in_df2go(df2go, df2go_before, mymeasure)
        print('... Added', mymeasure, 'in df2go:', df2go.shape)
    print('... DailyDOB loaded and added in %2.4f seconds.\n' % (time.time() - tic))

    # 2) Add historical Volatility (weights)
    tic = time.time()
    df2go = ji_helper_add_histVola_in_df2go(df2go, df2go_before)
    print('... Historical volatility loaded in %2.4f seconds.' % (time.time() - tic))
    # 2.5) Add historical Volume
    tic = time.time()
    df2go = ji_helper_add_histVolu_in_df2go(df2go, df2go_before)
    print('... Historical volume loaded in %2.4f seconds.' % (time.time() - tic))

    # 3) Estimate realized volatility
    tic = time.time()
    df2go['realVola'] = np.vectorize(jifin.compute_vola_maxmin)(df2go['HighPrice'], df2go['LowPrice'],
                                                              df2go['midQuote'], df2go['midQuote'])
    print('... MaxMin Volatility computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))

    # 4) Compute surprise of volatility
    tic = time.time()
    df2go['surVola'] = np.vectorize(jifin.my_divide)(df2go['realVola'], df2go['histVola5min'])
    print('... Volatility surprise computed in %2.4f seconds (using vectorize).' % (time.time() - tic))

    # 5) Add the benchmark price, i.e. the previous bin midquote MQ(j-1)
    df2go = add_priceBM_in_df2go_fast(df2go)

    # 6) Compute the Empirical Cost
    # 6.1) Handle missing bmPrice (rare but happens when MQ price is missing for first bin of the ...)
    df2go = fill_missing_priceBM(df2go)
    # 6.1.5) Fill null after all...
    if np.sum(df2go.priceBM.isnull())>0:
        print('- Filling %d null priceBM with zeros...'%(np.sum(df2go.priceBM.isnull())))
        df2go.loc[df2go.priceBM.isnull(),'priceBM'] = 0
    # 6.2) Compute
    tic = time.time()
    # obs: vectorize more efficient than using df.apply()
    df2go['empCost'] = np.vectorize(jifin.compute_empCost)(df2go['ExPriceForBin'], df2go['priceBM'],
                                                         df2go['sign'])
    print('... Empirical Cost computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))
    #df2go.to_csv('df2go_42cols_Step6.csv')

    # 6.3) Compute F ang G components
    tic = time.time()
    # obs: vectorize more efficient than using df.apply()
    df2go['Fcost'] = np.vectorize(jifin.compute_F)(df2go['midQuote'], df2go['priceBM'], df2go['vwap'],
                                                           df2go['sign'])
    df2go['Gcost'] = np.vectorize(jifin.compute_G)(df2go['midQuote'], df2go['priceBM'], df2go['vwap'],
                                                   df2go['sign'])
    print('... F and G costs computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))
    #df2go.to_csv('0_df2go_step7.csv')

    # 7) Compute participation rate
    tic = time.time()
    df2go['prate'] = np.vectorize(jifin.compute_prate)(df2go['ExShareForBin'], df2go['shareVolume'])
    print('... Computed participation rate in %2.4f seconds (using vectorizer).' % (time.time() - tic))

    # 8) Compute Style
    tic = time.time()
    df2go['style'] = np.vectorize(jifin.compute_style)(df2go['ExPriceForBin'], df2go['vwap']
                                                     , df2go['priceBM'], df2go['prate'], df2go['sign'])
    print('... Style computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))

    # 9) Compute Normalized Style
    tic = time.time()
    df2go['normStyle'] = np.vectorize(jifin.compute_style_normalized)(df2go['ExPriceForBin'], df2go['vwap']
                                                                      , df2go['HighPrice'], df2go['LowPrice'],
                                                                      df2go['prate'], df2go['sign'])
    print('... Style (normalized) computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))

    # 10) Add quadratic volatility from the 15sec MQ
    tic = time.time()
    df2go = ji_helper_add_quadVola_stdVola_15Sec_in_df2go(df2go, df2go_before)
    print('... Quad and Std volatility computed in %2.4f seconds (takes time to load...).' % (time.time() - tic))

    # 11) Compute and add the cumulative and last 30 min volatilities (HL and quadVola)
    tic = time.time()
    df2go = add_last30_cum_vola(df2go)
    print('... Cumulative and last30min volatilities (HL,quadVola) computed in %2.4f seconds.' % (time.time() - tic))

    # 12) Add cumulative Return from the 15sec MQ
    tic = time.time()
    df2go = ji_helper_add_cumReturn15sec_in_df2go(df2go, df2go_before)
    print('... Cumulative return computed in %2.4f seconds (takes time to load...).' % (time.time() - tic))

    # 13) Add Return of the 5min bin
    tic = time.time()
    df2go = ji_helper_add_Return_in_df2go(df2go)
    print('... Return and signedReturn computed in %2.4f seconds (takes time to load...).' % (time.time() - tic))

    # 14) Add volume-related
    # Add MM normalization costs
    df2go = ji_wrapper_add_MaxMinNormCost(df2go)
    df2go['dollarVolume'] = df2go.shareVolume * df2go.vwap / 1000000  # in million
    df2go['surVolume'] = df2go['dollarVolume'] / df2go.histVolu5min
    df2go['FGcost'] = df2go.Fcost + df2go.Gcost

    # 15) Add First Trade indicator
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

    # 16) Add additional feats
    df2go = ji_get_adjusted_milanBM_historicals(df2go)
    df2go = ji_compute_adjusted_Returns_surprises(df2go)
    df2go = ji_compute_normalized_CostReturn(df2go)

    print('\n**(TOTAL) df2go created and computed in %2.4f seconds.' % (time.time() - tic0))
    print('******************************************************\n')

    return df2go

def ji_clean_missingData(df2go,excludeNonExecuted=True):
    # 15) Cleaning clusters with missing data
    print('----- Removing rows with missing data: histVolu and others ------')
    print('- Number of rows before:',df2go.shape[0])
    # Missing historical volume
    mask2exclude = df2go.histVolu5min.isnull()
    print('- Number of trades without histVolume: %d. Excluding rows...' % np.sum(mask2exclude))
    df2go = df2go[~mask2exclude]

    # Exclude non-traded bins
    if excludeNonExecuted:
        # Excluded non-executed bins (to save space)
        print('- Excluding bins without client execution: %d (%d%%)...' % (
        np.sum(df2go.ExPriceForBin == 0), 100 * np.sum(df2go.ExPriceForBin == 0) / df2go.shape[0]))
        df2go = df2go[df2go.ExPriceForBin != 0].reset_index(drop=True) #
        try:
            aux = df2go['return']
            N = np.sum(aux==0)
            print('- Return==0 after removing no-exec bins: --> %d out of %d (%2.2f)' % (N, df2go.shape[0], N / df2go.shape[0]))
        except:
            print('- No Return information available...')

    # Missing empCost
    mask2exclude = df2go.empCost.isnull()
    print('- Number of null empCost: %d. Excluding rows...' % np.sum(mask2exclude))
    df2go = df2go[~mask2exclude].reset_index(drop=True)

    if 'HighPrice' in df2go.columns:
        print('Missing HighPrice:', df2go[df2go.HighPrice == 0].shape[0])
        print('Missing LowPrice&HighPrice:', df2go[(df2go.LowPrice == 0) | (df2go.HighPrice == 0)].shape[0])
        imask = (df2go.LowPrice == 0) | (df2go.HighPrice == 0)
        df2go = df2go[~imask]
        print('- Quick fix: deleted %d rows with  missing HighLowPrices...'%np.sum(imask))

    print('- Number of rows after:', df2go.shape[0])
    print('----- (Done cleaning) ------')

    return df2go


def get_prediction(row,bin2predict,col2use):
    # bin2predict: 1 is for 5min, 2 is for 10min, 6 is for 30min, etc
    if row[col2use]:
        #return row[col2use].split(';')[row.binIndex+(bin2predict-1)]
        return row[col2use].split(';')[row.binIndex-1]
    else:
        return None

def ji_wrapper_redo_adjustment_1stTrade_ABD2017(datalabel,extralabel=''):
    """Given the datalabel used to save the clusterdata it will load, fix adjustment and save with the same name"""

    ## Paths
    outpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/Out'
    dpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data'
    tic0 = time.time()

    ## 1) LOAD PREVIOUS DATA AND UPDATE
    print('- Loading df2go_62cols_%s.csv.gz' % datalabel)
    fname = os.path.join(dpath, 'df2go_62cols_%s.csv.gz' % datalabel)
    df2go = pd.read_csv(fname)  # ~1GB...
    if 'Unnamed: 0' in df2go.columns:
        df2go = df2go.drop(columns=['Unnamed: 0'])
    df2go = ji_readjust_1sTrade(df2go)
    print('- Saving: %s...' % (fname))
    df2go.to_csv(fname, compression='gzip', index=False)  #
    print('- Total computation time: %d minutes...' % ((time.time() - tic0) // 60))

def ji_wrapper_save_df2plot2017_volaPrediction(datalabel,extralabel=''):
    """Given the datalabel used to save the clusterdata as well as the input files split into parts,
    it will load the engine predictions and merge on the df2go, and save as df2plot
    - 2019/04/12: add back mddv, histVolaW and histVoluW for the 30min computation"""

    ## Paths
    outpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/Out'
    dpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data'
    df2plotpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data_df2plot'  # Where df2plot will be saved
    #df2plotpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data_df2plot_TMP'  # Where df2plot will be saved

    tic0 = time.time()

    ## 1) LOAD PREVIOUS DATA AND ADD (TO SAVE TIME)
    print('- Loading df2go...')
    tic = time.time()
    # df2go = pd.read_csv('df2go_42cols_2017_LG9up.csv')
    # df2go = pd.read_csv('df2go_41cols_2017_LG5upBIG.csv') # (3.3GB size...)
    # df2go = pd.read_csv(os.path.join(dpath,'df2go_52cols_2017_HLGlarger.csv')) # 2.9GB...
    # df2go = pd.read_csv(os.path.join(dpath,'df2go_53cols_2017_HLGlarger.csv.gz')) # 2.5GB...
    # df2go = pd.read_csv(os.path.join(dpath,'df2go_54cols_2017_HLGlarger_ABD.csv.gz')) # 2.5GB...
    df2go = pd.read_csv(os.path.join(dpath, 'df2go_62cols_%s.csv.gz' % datalabel))  # ~1GB...
    try:
        df2go = df2go.drop(columns=['Unnamed: 0'])
    except:
        print('- Index column already removed.')
    # Add bin number in eky to match with engine output
    df2go.rename(columns={'#key': 'token'}, inplace=True)
    df2go['token'] = df2go['token'] + '_bin' + df2go.bin.astype(str)
    print('- Time taken to load df2go: %d seconds...' % (time.time() - tic))
    df2go.head()


    ## 2) Load and concatenate parts (has to be done since split is not exactly on the day 78 bins...)
    tic = time.time()
    dfall = pd.DataFrame()
    previousMissing = False
    for i in range(10):
        try:
            fname = os.path.join(outpath, 'marketCondition_%s_part%d.csv' % (datalabel, i + 1))
            if os.path.isfile(fname):
                df1 = pd.read_csv(fname)
            else:
                fname = os.path.join(outpath, 'marketCondition_%s_part%d.csv.gz' % (datalabel, i + 1))
                df1 = pd.read_csv(fname)
            print('- Loading marketCondition part#%d:' % (i + 1), df1.shape)
            if previousMissing:
                df1 = df1.iloc[df1.index[df1.binIndex == 0][
                                   0]:]  # Since previous is missing, exclude the discontinuous bin. Start from the first bin==0
                print('-- Since previous part is missing, exclude the first rows without predecessors...')
            dfall = dfall.append(df1[['token', 'binIndex', 'marketConditionPrediction']])
            previousMissing = False
        except:
            print('- (missing not loaded: marketCondition_%s_part%d.csv...)' % (datalabel, i + 1))
            previousMissing = True  # Indicator for the next part
    print('- Time taken to load and concatenate engine output: %d seconds...' % (time.time() - tic))
    dfall.head()

    ## 3.1) Create prediction column
    print('- Creating 5min prediction columns...')
    tic = time.time()
    t2predict = 1  # 1 for 5min, 6 for 30 min, 12 for 1hour  predictions
    predlabel = 'pred%smin' % (t2predict * 5)
    # Initialize
    dfall[predlabel] = [None] * dfall.shape[0]
    # Select the bins in which prediction is available
    imask = dfall.binIndex > (
        t2predict)  # if t2predict=6 (30min), select after binIndex=7 (inclusive) since will look at bin=7-6=1
    # Shift by t2predict (IMPORTANT: all bins should be ordered)
    dfall.loc[imask, predlabel] = dfall.shift(periods=t2predict).loc[imask, 'marketConditionPrediction']
    # Updated selected
    dfall[predlabel] = dfall.apply(get_prediction, bin2predict=t2predict, col2use=predlabel, axis=1).astype(float)
    print('- Time taken to create the prediction column (start in 30min): %d seconds...' % (time.time() - tic))

    ## 3.2) Create prediction column
    print('- Creating 30min prediction columns...')
    tic = time.time()
    t2predict = 6  # 1 for 5min, 6 for 30 min, 12 for 1hour  predictions
    predlabel = 'pred%smin' % (t2predict * 5)
    # Initialize
    dfall[predlabel] = [None] * dfall.shape[0]
    # Select the bins in which prediction is available
    imask = dfall.binIndex > (
        t2predict)  # if t2predict=6 (30min), select after binIndex=7 (inclusive) since will look at bin=7-6=1
    # Shift by t2predict (IMPORTANT: all bins should be ordered)
    dfall.loc[imask, predlabel] = dfall.shift(periods=t2predict).loc[imask, 'marketConditionPrediction']
    # Updated selected
    dfall[predlabel] = dfall.apply(get_prediction, bin2predict=t2predict, col2use=predlabel, axis=1).astype(float)
    print('- Time taken to create the prediction column (start in 30min): %d seconds...' % (time.time() - tic))

    ## 4) Merge

    # cols2include_full = ['token', 'tradeDate', 'arriveBin', 'relSize', 'lg', 'HLG', 'algoStrategy', 'algoType',
    #                 'SpreadCapture',
    #                 'binDuration', 'bin', 'ExPriceForBin', 'ExShareForBin', 'FillNbrForBin',
    #                 'midQuote', 'LowPrice', 'HighPrice', 'vwap', 'spread', 'shareVolume',
    #                 'histVola5min', 'histVolu5min', 'realVola',
    #                 'surVola', 'priceBM', 'empCost', 'Fcost', 'Gcost', 'prate',
    #                 'normStyle', 'signedReturn',
    #                 'dollarVolume', 'surVolume', 'firstTrade', 'adjBM',
    #                 'adjHistVola5min', 'adjHistVolu5min', 'adjRealVola', 'adjSizeShare',
    #                 'gapDuration', 'very1stTrade', 'adjSignedReturn', 'adjSurVola',
    #                 'adjSurVolume', 'normAdjSignedReturn', 'normSignedReturn',
    #                 'normEmpCost']
    #
    cols2include = ['token','tradeDate','arriveBin','relSize','HLG','mddv','histVola','closeTm1','algoType','SpreadCapture',
           'binDuration','bin','ExPriceForBin', 'ExShareForBin', 'FillNbrForBin',
           'midQuote', 'LowPrice', 'HighPrice', 'vwap', 'spread', 'shareVolume', 'histVolaW', 'histVoluW',
           'histVola5min','histVolu5min','realVola',
           'surVola', 'priceBM', 'empCost', 'prate',
           'normStyle', 'signedReturn',
            'surVolume', 'firstTrade', 'adjBM',
           'adjHistVola5min', 'adjHistVolu5min', 'adjRealVola', 'adjSizeShare','sizeShare', # sizeShare added in 2019/04/10
           'gapDuration', 'very1stTrade', 'adjSignedReturn', 'adjSurVola',
           'adjSurVolume', 'normAdjSignedReturn', 'normSignedReturn',
           'normEmpCost']


    df2go = df2go[cols2include].merge(dfall[['token','pred5min','pred30min']],on='token')
    del dfall

    # Back fill for the initial bins
    df2go[['pred5min','pred30min']] = df2go[['pred5min','pred30min']].fillna(method='bfill')

    ## 5) Cleaning
    cleanMissing = False
    if cleanMissing:
        df2go = ji_clean_missingData(df2go, excludeNonExecuted=False)

    ## 6) Update prediction of the firstTrade==1 after a gap (very1stTrade==0)

    # - a) Fix duration (when there is trade in Bin1): (firstTrade==1) & (very1stTrade==0) & (gapDuration==0)
    cond = (df2go.gapDuration == 0) & (df2go.firstTrade == 1) & (df2go.very1stTrade == 0)
    if np.sum(cond) > 0:
        print('** Fixing! ** 1stTrade after gap with duration ZERO: %d cases...' % np.sum(cond))
        df2go = ji_p5_correct_duration(df2go, cond)

    # - b) Corrected prediction of 1st trade after long gap
    df2go = ji_p5_add_corrected_prediction_1stTrade(df2go)

    # Checking consistency...
    ji_p5_check_df2plot(df2go)

    ## 7) Saving
    print('- Saving df2plot. Be patient...')
    tic = time.time()
    #spath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data'
    df2go.to_csv(os.path.join(df2plotpath, 'df2plot_%s%s.csv.gz' %(datalabel,extralabel)), compression='gzip', index=False)
    print('- df2plot saved:', os.path.join(df2plotpath, 'df2plot_%s%s.csv.gz'%(datalabel,extralabel)))
    print('---')
    print('- TOTAL COMPUTATION TIME: %d minutes...'%((time.time() - tic0)//60))
    print('------------------- DONE ----------------------------')


def ji_wrapper_save_df2plot_volaPrediction(datalabel,extralabel=''):
    """Given the datalabel used to save the clusterdata as well as the input files split into parts,
    it will load the engine predictions and merge on the df2go, and save as df2plot"""

    ## Paths
    outpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/Out'
    dpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data'
    df2plotpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data_df2plot'  # Where df2plot will be saved
    #df2plotpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data_df2plot_TMP'  # Where df2plot will be saved

    tic0 = time.time()

    ## 1) LOAD PREVIOUS DATA AND ADD (TO SAVE TIME)
    print('- Loading df2go...')
    tic = time.time()
    # df2go = pd.read_csv('df2go_42cols_2017_LG9up.csv')
    # df2go = pd.read_csv('df2go_41cols_2017_LG5upBIG.csv') # (3.3GB size...)
    # df2go = pd.read_csv(os.path.join(dpath,'df2go_52cols_2017_HLGlarger.csv')) # 2.9GB...
    # df2go = pd.read_csv(os.path.join(dpath,'df2go_53cols_2017_HLGlarger.csv.gz')) # 2.5GB...
    # df2go = pd.read_csv(os.path.join(dpath,'df2go_54cols_2017_HLGlarger_ABD.csv.gz')) # 2.5GB...
    df2go = pd.read_csv(os.path.join(dpath, 'df2go_62cols_%s.csv.gz' % datalabel))  # ~1GB...
    try:
        df2go = df2go.drop(columns=['Unnamed: 0'])
    except:
        print('- Index column already removed.')
    # Add bin number in eky to match with engine output
    df2go.rename(columns={'#key': 'token'}, inplace=True)
    df2go['token'] = df2go['token'] + '_bin' + df2go.bin.astype(str)
    print('- Time taken to load df2go: %d seconds...' % (time.time() - tic))
    df2go.head()


    ## 2) Load and concatenate parts (has to be done since split is not exactly on the day 78 bins...)
    tic = time.time()
    dfall = pd.DataFrame()
    previousMissing = False
    for i in range(10):
        try:
            fname = os.path.join(outpath, 'marketCondition_%s_part%d.csv' % (datalabel, i + 1))
            if os.path.isfile(fname):
                df1 = pd.read_csv(fname)
            else:
                fname = os.path.join(outpath, 'marketCondition_%s_part%d.csv.gz' % (datalabel, i + 1))
                df1 = pd.read_csv(fname)
            print('- Loading marketCondition part#%d:' % (i + 1), df1.shape)
            if previousMissing:
                df1 = df1.iloc[df1.index[df1.binIndex == 0][
                                   0]:]  # Since previous is missing, exclude the discontinuous bin. Start from the first bin==0
                print('-- Since previous part is missing, exclude the first rows without predecessors...')
            dfall = dfall.append(df1[['token', 'binIndex', 'marketConditionPrediction']])
            previousMissing = False
        except:
            print('- (missing not loaded: marketCondition_%s_part%d.csv...)' % (datalabel, i + 1))
            previousMissing = True  # Indicator for the next part
    print('- Time taken to load and concatenate engine output: %d seconds...' % (time.time() - tic))
    dfall.head()

    ## 3.1) Create prediction column
    print('- Creating 5min prediction columns...')
    tic = time.time()
    t2predict = 1  # 1 for 5min, 6 for 30 min, 12 for 1hour  predictions
    predlabel = 'pred%smin' % (t2predict * 5)
    # Initialize
    dfall[predlabel] = [None] * dfall.shape[0]
    # Select the bins in which prediction is available
    imask = dfall.binIndex > (
        t2predict)  # if t2predict=6 (30min), select after binIndex=7 (inclusive) since will look at bin=7-6=1
    # Shift by t2predict (IMPORTANT: all bins should be ordered)
    dfall.loc[imask, predlabel] = dfall.shift(periods=t2predict).loc[imask, 'marketConditionPrediction']
    # Updated selected
    dfall[predlabel] = dfall.apply(get_prediction, bin2predict=t2predict, col2use=predlabel, axis=1).astype(float)
    print('- Time taken to create the prediction column (start in 30min): %d seconds...' % (time.time() - tic))

    ## 3.2) Create prediction column
    print('- Creating 30min prediction columns...')
    tic = time.time()
    t2predict = 6  # 1 for 5min, 6 for 30 min, 12 for 1hour  predictions
    predlabel = 'pred%smin' % (t2predict * 5)
    # Initialize
    dfall[predlabel] = [None] * dfall.shape[0]
    # Select the bins in which prediction is available
    imask = dfall.binIndex > (
        t2predict)  # if t2predict=6 (30min), select after binIndex=7 (inclusive) since will look at bin=7-6=1
    # Shift by t2predict (IMPORTANT: all bins should be ordered)
    dfall.loc[imask, predlabel] = dfall.shift(periods=t2predict).loc[imask, 'marketConditionPrediction']
    # Updated selected
    dfall[predlabel] = dfall.apply(get_prediction, bin2predict=t2predict, col2use=predlabel, axis=1).astype(float)
    print('- Time taken to create the prediction column (start in 30min): %d seconds...' % (time.time() - tic))

    ## 4) Merge

    # cols2include_full = ['token', 'tradeDate', 'arriveBin', 'relSize', 'lg', 'HLG', 'algoStrategy', 'algoType',
    #                 'SpreadCapture',
    #                 'binDuration', 'bin', 'ExPriceForBin', 'ExShareForBin', 'FillNbrForBin',
    #                 'midQuote', 'LowPrice', 'HighPrice', 'vwap', 'spread', 'shareVolume',
    #                 'histVola5min', 'histVolu5min', 'realVola',
    #                 'surVola', 'priceBM', 'empCost', 'Fcost', 'Gcost', 'prate',
    #                 'normStyle', 'signedReturn',
    #                 'dollarVolume', 'surVolume', 'firstTrade', 'adjBM',
    #                 'adjHistVola5min', 'adjHistVolu5min', 'adjRealVola', 'adjSizeShare',
    #                 'gapDuration', 'very1stTrade', 'adjSignedReturn', 'adjSurVola',
    #                 'adjSurVolume', 'normAdjSignedReturn', 'normSignedReturn',
    #                 'normEmpCost']
    #
    cols2include = ['token','tradeDate','arriveBin','relSize','HLG','algoType','SpreadCapture',
           'binDuration','bin','ExPriceForBin', 'ExShareForBin', 'FillNbrForBin',
           'midQuote', 'LowPrice', 'HighPrice', 'vwap', 'spread', 'shareVolume',
           'histVola5min','histVolu5min','realVola',
           'surVola', 'priceBM', 'empCost', 'prate',
           'normStyle', 'signedReturn',
            'surVolume', 'firstTrade', 'adjBM',
           'adjHistVola5min', 'adjHistVolu5min', 'adjRealVola', 'adjSizeShare','sizeShare', # sizeShare added in 2019/04/10
           'gapDuration', 'very1stTrade', 'adjSignedReturn', 'adjSurVola',
           'adjSurVolume', 'normAdjSignedReturn', 'normSignedReturn',
           'normEmpCost']


    df2go = df2go[cols2include].merge(dfall[['token','pred5min','pred30min']],on='token')
    del dfall

    # Back fill for the initial bins
    df2go[['pred5min','pred30min']] = df2go[['pred5min','pred30min']].fillna(method='bfill')

    ## 5) Cleaning
    cleanMissing = False
    if cleanMissing:
        df2go = ji_clean_missingData(df2go, excludeNonExecuted=False)

    ## 6) Update prediction of the firstTrade==1 after a gap (very1stTrade==0)

    # - a) Fix duration (when there is trade in Bin1): (firstTrade==1) & (very1stTrade==0) & (gapDuration==0)
    cond = (df2go.gapDuration == 0) & (df2go.firstTrade == 1) & (df2go.very1stTrade == 0)
    if np.sum(cond) > 0:
        print('** Fixing! ** 1stTrade after gap with duration ZERO: %d cases...' % np.sum(cond))
        df2go = ji_p5_correct_duration(df2go, cond)

    # - b) Corrected prediction of 1st trade after long gap
    df2go = ji_p5_add_corrected_prediction_1stTrade(df2go)

    # Checking consistency...
    ji_p5_check_df2plot(df2go)

    ## 7) Saving
    print('- Saving df2plot. Be patient...')
    tic = time.time()
    #spath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data'
    df2go.to_csv(os.path.join(df2plotpath, 'df2plot_%s%s.csv.gz' %(datalabel,extralabel)), compression='gzip', index=False)
    print('- df2plot saved:', os.path.join(df2plotpath, 'df2plot_%s%s.csv.gz'%(datalabel,extralabel)))
    print('---')
    print('- TOTAL COMPUTATION TIME: %d minutes...'%((time.time() - tic0)//60))
    print('------------------- DONE ----------------------------')

def ji_wrapper_save_df2plot_volaPrediction_v2(datalabel,extralabel=''):
    """Given the datalabel used to save the clusterdata as well as the input files split into parts,
    it will load the engine predictions and merge on the df2go, and save as df2plot"""

    ## Paths
    outpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/Out'
    dpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data'
    tic0 = time.time()

    ## 1) LOAD PREVIOUS DATA AND ADD (TO SAVE TIME)
    print('- Loading df2go...')
    tic = time.time()
    # df2go = pd.read_csv('df2go_42cols_2017_LG9up.csv')
    # df2go = pd.read_csv('df2go_41cols_2017_LG5upBIG.csv') # (3.3GB size...)
    # df2go = pd.read_csv(os.path.join(dpath,'df2go_52cols_2017_HLGlarger.csv')) # 2.9GB...
    # df2go = pd.read_csv(os.path.join(dpath,'df2go_53cols_2017_HLGlarger.csv.gz')) # 2.5GB...
    # df2go = pd.read_csv(os.path.join(dpath,'df2go_54cols_2017_HLGlarger_ABD.csv.gz')) # 2.5GB...
    df2go = pd.read_csv(os.path.join(dpath, 'df2go_62cols_%s.csv.gz' % datalabel))  # ~1GB...
    df2go = df2go.drop(columns=['Unnamed: 0'])
    # Add bin number in eky to match with engine output
    df2go.rename(columns={'#key': 'token'}, inplace=True)
    df2go['token'] = df2go['token'] + '_bin' + df2go.bin.astype(str)
    print('- Time taken to load df2go: %d seconds...' % (time.time() - tic))
    df2go.head()


    ## 2) Load and concatenate parts (has to be done since split is not exactly on the day 78 bins...)
    tic = time.time()
    dfall = pd.DataFrame()
    previousMissing = False
    for i in range(10):
        try:
            df1 = pd.read_csv(os.path.join(outpath, 'marketCondition_%s_part%d.csv' % (datalabel, i + 1)))
            print('- Loading marketCondition part#%d:' % (i + 1), df1.shape)
            if previousMissing:
                df1 = df1.iloc[df1.index[df1.binIndex == 0][
                                   0]:]  # Since previous is missing, exclude the discontinuous bin. Start from the first bin==0
                print('-- Since previous part is missing, exclude the first rows without predecessors...')
            dfall = dfall.append(df1[['token', 'binIndex', 'marketConditionPrediction']])
            previousMissing = False
        except:
            print('- (missing not loaded: marketCondition_%s_part%d.csv...)' % (datalabel, i + 1))
            previousMissing = True  # Indicator for the next part
    print('- Time taken to load and concatenate engine output: %d seconds...' % (time.time() - tic))
    dfall.head()

    ## 3.1) Create prediction column
    print('- Creating 5min prediction columns...')
    tic = time.time()
    t2predict = 1  # 1 for 5min, 6 for 30 min, 12 for 1hour  predictions
    predlabel = 'pred%smin' % (t2predict * 5)
    # Initialize
    dfall[predlabel] = [None] * dfall.shape[0]
    # Select the bins in which prediction is available
    imask = dfall.binIndex > (
        t2predict)  # if t2predict=6 (30min), select after binIndex=7 (inclusive) since will look at bin=7-6=1
    # Shift by t2predict (IMPORTANT: all bins should be ordered)
    dfall.loc[imask, predlabel] = dfall.shift(periods=t2predict).loc[imask, 'marketConditionPrediction']
    # Updated selected
    dfall[predlabel] = dfall.apply(get_prediction, bin2predict=t2predict, col2use=predlabel, axis=1).astype(float)
    print('- Time taken to create the prediction column (start in 30min): %d seconds...' % (time.time() - tic))

    ## 3.2) Create prediction column
    print('- Creating 30min prediction columns...')
    tic = time.time()
    t2predict = 6  # 1 for 5min, 6 for 30 min, 12 for 1hour  predictions
    predlabel = 'pred%smin' % (t2predict * 5)
    # Initialize
    dfall[predlabel] = [None] * dfall.shape[0]
    # Select the bins in which prediction is available
    imask = dfall.binIndex > (
        t2predict)  # if t2predict=6 (30min), select after binIndex=7 (inclusive) since will look at bin=7-6=1
    # Shift by t2predict (IMPORTANT: all bins should be ordered)
    dfall.loc[imask, predlabel] = dfall.shift(periods=t2predict).loc[imask, 'marketConditionPrediction']
    # Updated selected
    dfall[predlabel] = dfall.apply(get_prediction, bin2predict=t2predict, col2use=predlabel, axis=1).astype(float)
    print('- Time taken to create the prediction column (start in 30min): %d seconds...' % (time.time() - tic))

    ## 4) Merge

    cols2include = ['token','tradeDate','arriveBin','relSize','HLG','algoType','SpreadCapture',
           'binDuration','bin','ExPriceForBin', 'ExShareForBin', 'FillNbrForBin',
           'midQuote', 'LowPrice', 'HighPrice', 'vwap', 'spread', 'shareVolume',
           'histVola5min','histVolu5min','realVola',
           'surVola', 'priceBM', 'empCost', 'prate',
           'normStyle', 'signedReturn',
            'surVolume', 'firstTrade', 'adjBM',
           'adjHistVola5min', 'adjHistVolu5min', 'adjRealVola', 'adjSizeShare', 'sizeShare',
           'gapDuration', 'very1stTrade', 'adjSignedReturn', 'adjSurVola',
           'adjSurVolume', 'normAdjSignedReturn', 'normSignedReturn',
           'normEmpCost']

    df2go = df2go[cols2include].merge(dfall[['token','pred5min','pred30min']],on='token')
    del dfall

    # Back fill for the initial bins
    df2go[['pred5min','pred30min']] = df2go[['pred5min','pred30min']].fillna(method='bfill')

    ## 5) Cleaning
    cleanMissing = False
    if cleanMissing:
        df2go = ji_clean_missingData(df2go, excludeNonExecuted=False)

    ## 6) Saving
    print('- Saving df2plot. Be patient...')
    tic = time.time()
    spath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data'
    df2go.to_csv(os.path.join(spath, 'df2plot_%s%s.csv.gz' %(datalabel,extralabel)), compression='gzip', index=False)
    print('- df2plot saved:', os.path.join(spath, 'df2plot_%s%s.csv.gz'%(datalabel,extralabel)))
    print('---')
    print('- TOTAL COMPUTATION TIME: %d minutes...'%((time.time() - tic0)//60))
    print('------------------- DONE ----------------------------')


def ji_wrapper_save_df2plot_volaPrediction_fromDf2go(df2go,outpath,spath,datalabel,extralabel=''):
    """Given the datalabel used to save the clusterdata as well as the input files split into parts,
    it will load the engine predictions and merge on the df2go, and save as df2plot
    - outpath: location of DyCE output
    - spath: where df2plot will be saved"""

    print('------------------- COMPUTING df2plot for %s --------------'%datalabel)
    ## Paths
    #outpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/Out'
    #dpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data'
    tic0 = time.time()

    ## 1) LOAD PREVIOUS DATA AND ADD (TO SAVE TIME)
    #print('- Loading df2go...')
    tic = time.time()
    #df2go = pd.read_csv(os.path.join(dpath, 'df2go_62cols_%s.csv.gz' % datalabel))  # ~1GB...
    #df2go = df2go.drop(columns=['Unnamed: 0'])

    # Add bin number in key to match with engine output
    df2go.rename(columns={'#key': 'token'}, inplace=True)
    df2go['token'] = df2go['token'] + '_bin' + df2go.bin.astype(str)
    print('- Time taken to load df2go: %d seconds...' % (time.time() - tic))
    df2go.head()


    ## 2) Load and concatenate parts (has to be done since split is not exactly on the day 78 bins...)
    tic = time.time()
    dfall = pd.DataFrame()

    try:
        fname = os.path.join(outpath, 'marketCondition_%s.csv' % (datalabel))
        if os.path.isfile(fname):
            df1 = pd.read_csv(fname)
        else:
            df1 = pd.read_csv(os.path.join(outpath, 'marketCondition_%s.csv.gz' % (datalabel)))
        print('- Loading marketCondition %s:' % datalabel, df1.shape)
        dfall = dfall.append(df1[['token', 'binIndex', 'marketConditionPrediction']])
    except:
        print('- (missing not loaded: marketCondition_%s.csv...)' % (datalabel))

    print('- Time taken to load and concatenate engine output: %d seconds...' % (time.time() - tic))
    dfall.head()

    ## 3.1) Create prediction column
    print('- Creating 5min prediction columns...')
    tic = time.time()
    t2predict = 1  # 1 for 5min, 6 for 30 min, 12 for 1hour  predictions
    predlabel = 'pred%smin' % (t2predict * 5)
    # Initialize
    dfall[predlabel] = [None] * dfall.shape[0]
    # Select the bins in which prediction is available
    imask = dfall.binIndex > (t2predict)  # if t2predict=6 (30min), select after binIndex=7 (inclusive) since will look at bin=7-6=1
    # Shift by t2predict (IMPORTANT: all bins should be ordered)
    dfall.loc[imask, predlabel] = dfall.shift(periods=t2predict).loc[imask, 'marketConditionPrediction']
    # Updated selected
    dfall[predlabel] = dfall.apply(get_prediction, bin2predict=t2predict, col2use=predlabel, axis=1).astype(float)
    print('- Time taken to create the prediction column (start in 30min): %d seconds...' % (time.time() - tic))

    ## 3.2) Create prediction column
    print('- Creating 30min prediction columns...')
    tic = time.time()
    t2predict = 6  # 1 for 5min, 6 for 30 min, 12 for 1hour  predictions
    predlabel = 'pred%smin' % (t2predict * 5)
    # Initialize
    dfall[predlabel] = [None] * dfall.shape[0]
    # Select the bins in which prediction is available
    imask = dfall.binIndex > (t2predict)  # if t2predict=6 (30min), select after binIndex=7 (inclusive) since will look at bin=7-6=1
    # Shift by t2predict (IMPORTANT: all bins should be ordered)
    dfall.loc[imask, predlabel] = dfall.shift(periods=t2predict).loc[imask, 'marketConditionPrediction']
    # Updated selected
    dfall[predlabel] = dfall.apply(get_prediction, bin2predict=t2predict, col2use=predlabel, axis=1).astype(float)
    print('- Time taken to create the prediction column (start in 30min): %d seconds...' % (time.time() - tic))

    ## 4) Merge

    cols2include = ['token','tradeDate','arriveBin','relSize','HLG','mddv','histVola','closeTm1','algoType','SpreadCapture',
           'binDuration','bin','ExPriceForBin', 'ExShareForBin', 'FillNbrForBin',
           'midQuote', 'LowPrice', 'HighPrice', 'vwap', 'spread', 'shareVolume','histVolaW','histVoluW',
           'histVola5min','histVolu5min','realVola',
           'surVola', 'priceBM', 'empCost', 'prate',
           'normStyle', 'signedReturn',
            'surVolume', 'adjBM',
           'adjHistVola5min', 'adjHistVolu5min', 'adjRealVola', 'adjSizeShare', 'sizeShare',
           'firstTrade', 'gapDuration', 'very1stTrade', 'adjSignedReturn', 'adjSurVola',
           'adjSurVolume', 'normAdjSignedReturn', 'normSignedReturn',
           'normEmpCost']

    df2go = df2go[cols2include].merge(dfall[['token','pred5min','pred30min']],on='token')

    # Back fill for the initial bins
    df2go[['pred5min','pred30min']] = df2go[['pred5min','pred30min']].fillna(method='bfill')

    ## 5) Cleaning
    cleanMissing = False
    if cleanMissing:
        df2go = ji_clean_missingData(df2go, excludeNonExecuted=False)

    ## 6) Update prediction of the firstTrade==1 after a gap (very1stTrade==0)

    # - a) Fix duration (when there is trade in Bin1): (firstTrade==1) & (very1stTrade==0) & (gapDuration==0)
    cond = (df2go.gapDuration == 0) & (df2go.firstTrade == 1) & (df2go.very1stTrade == 0)
    if np.sum(cond) > 0:
        print('** Fixing! ** 1stTrade after gap with duration ZERO: %d cases...' % np.sum(cond))
        df2go = ji_p5_correct_duration(df2go, cond)

    # - b) Corrected prediction of 1st trade after long gap
    df2go = ji_p5_add_corrected_prediction_1stTrade(df2go)

    # Checking consistency...
    ji_p5_check_df2plot(df2go)

    ## 7) Saving
    print('- Saving df2plot...')
    tic = time.time()
    #spath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data'
    df2go.to_csv(os.path.join(spath, 'df2plot_%s%s.csv.gz' %(datalabel,extralabel)), compression='gzip', index=False)
    print('- df2plot saved:', os.path.join(spath, 'df2plot_%s%s.csv.gz'%(datalabel,extralabel)))
    print('---')
    print('- Time taken to save df2plot: %d seconds...'%((time.time() - tic0)))
    print('------------------- df2plot computed successfully -----------------------')

def ji_p5_add_corrected_prediction_1stTrade_30min(df2go):
    #cond = (df2go.firstTrade==1) & (df2go.very1stTrade==0) & (df2go.gapDuration>=6)
    cond = (df2go.firstTrade == 1) & (df2go.very1stTrade == 0) & (df2go.gapDuration >= 1)
    print('- Generating corrected prediction for the 1stTrades after gap>=30min: %d cases...'%np.sum(cond))

#    pred5min = df2go.pred5min.copy().values
    pred30min = df2go.pred30min.copy().values

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
 #               pred5min[i] = df2go.pred5min.iloc[j+6] # last.activeBin+1
                pred30min[i] = df2go.pred30min.iloc[j+1]
                break
            else:
                if cbin[j] == 1:
                    # very1stTrade[i] = 1
                    print('** Warning!! ** Reach Bin1 without active bin. Check...')
                    break  # reached the 2nd bin, no more update
                # next iteration
                j += -1
    #print('- Correction done: update with the prediction done at the last active bin (gap>30min).')
#    df2go['adjPred5min'] = pred5min
    df2go['adjPred30min'] = pred30min
    return df2go

def ji_p5_add_corrected_prediction_1stTrade(df2go):
    cond = (df2go.firstTrade==1) & (df2go.very1stTrade==0) & (df2go.gapDuration>=6)
    print('- Generating corrected prediction for the 1stTrades after gap>30min: %d cases...'%np.sum(cond))

    pred5min = df2go.pred5min.copy().values
    pred30min = df2go.pred30min.copy().values

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
                pred5min[i] = df2go.pred5min.iloc[j+6] # last.activeBin+1
                pred30min[i] = df2go.pred30min.iloc[j+6]
                break
            else:
                if cbin[j] == 1:
                    # very1stTrade[i] = 1
                    print('** Warning!! ** Reach Bin1 without active bin. Check...')
                    break  # reached the 2nd bin, no more update
                # next iteration
                j += -1
    print('- Correction done: update with the prediction done at the last active bin (gap>30min).')
    df2go['adjPred5min'] = pred5min
    df2go['adjPred30min'] = pred30min
    return df2go

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

def ji_p5_check_df2plot(df2plot, label = 'df2plot'):
    ## Check Case 1: werid high adjRealVola
    cond = (df2plot.adjRealVola > 1) & (df2plot.firstTrade != 1)
    if np.sum(cond) > 0:
        print('** Warning! ** High adjRealVola detected (non-1stTrade bin): %d cases...' % np.sum(cond))
    else:
        print('- High adjRealVola detected (non-1stTrade bin): %d cases.' % np.sum(cond))
    cond = (df2plot.adjRealVola > 1) & (df2plot.firstTrade == 1)
    if np.sum(cond) > 0:
        print('** Warning! ** High adjRealVola detected (1stTrade bin): %d cases...' % np.sum(cond))
    else:
        print('- High adjRealVola detected (1stTrade bin): %d cases.' % np.sum(cond))
    cond = (df2plot.gapDuration == 0) & (df2plot.firstTrade == 1) & (df2plot.very1stTrade == 0)
    if np.sum(cond) > 0:
        print('** Warning! ** 1stTrade after gap with duration ZERO: %d cases...' % np.sum(cond))
    else:
        print('- 1stTrade after gap with duration ZERO: %d cases.' % np.sum(cond))
    xx = (np.sum(df2plot.realVola == 0) / df2plot.shape[0])
    if xx > 0.8:
        print('** Warning! ** Number of zero realVola too high! %2.2f...' % xx)
    print('- Consistency check done: %s'%label)

def ji_readjust_1sTrade(df2go):
    """MUST step to fix 1stTraded bin
    - Create an adjusted prediction for the 1sTraded bin, where prediction is done in the last traded bin, not the current
    - Fix the special case where there are double occurrence of 1stTraded and very1stTraded in Bin1
    - Missing very1stTrade: happened when it was the Bin1, which was never checked..."""

    print('- Updating and fixing the 1stTraded bins')

    print('- Use the last traded MQ, not the previous one. Sum-up historical vola/volu weights')
    newBM = df2go.priceBM.copy().values
    newHistVola5min = df2go.histVola5min.copy().values
    newHistVolu5min = df2go.histVolu5min.copy().values
    newRealVola = df2go.realVola.copy().values
    durationFromLast = np.zeros(df2go.shape[0])
    very1stTrade = np.zeros(df2go.shape[0])

    # auxiliary
    bin = df2go.bin.values
    MQ = df2go.midQuote.values
    maxP = df2go.HighPrice.values
    minP = df2go.LowPrice.values

    shareTraded = df2go.ExShareForBin.values
    #close = df2go.closeTm1.values # USE THIS AS THIS INFO IS INCLUDED...
    print('- Using open price instead of close price...')
    close = df2go.openPrice.values # placeholder

    nfill = df2go.FillNbrForBin.values
    # historicals
    histVola = df2go.histVola.values
    mddv = df2go.mddv.values
    histVolaW = df2go.histVolaW.values
    histVoluW = df2go.histVoluW.values

    # Compute sizeShare (milan)
    sizeShare0 = shareTraded/(mddv*1000000*histVoluW/close) # Do not consider gap (convert MDDV to $)
    sizeShare = shareTraded / (mddv * 1000000 * histVoluW / close)  # Do not consider gap (convert MDDV to $)

    iupdate = df2go.index[(df2go.firstTrade == 1) & (df2go.bin != 1)]

    # # Since bin==1 are left out, set the very1stTrade indicator
    # very1stTrade[(df2go.firstTrade == 1) & (df2go.bin == 1)] = 1

    print('- Number of priceBM to update (after gap):', len(iupdate))
    for i in iupdate:
        # print('i:',i)
        # Initialize
        sumVolaW = histVolaW[i]*histVolaW[i]
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
                newBM[i] = MQ[j] # last active bin price
                newHistVola5min[i] = histVola[i]*np.sqrt(sumVolaW) # histVola * sqrt(sum(weights^2)) : accumulated weight
                newHistVolu5min[i] = mddv[i]*sumVoluW # MDDV * sum(shares): accumulated weights
                # compute HL volatility
                MQ0 = MQ[j+1] #last.activeBin+1
                #newRealVola[i] = jifin.compute_vola_maxmin(np.max(allmaxP),np.min(allminP),MQ0,MQ1)
                newRealVola[i] = jifin.compute_vola_maxmin_silent(np.max(allmaxP), np.min(allminP), MQ0, MQ1)
                if newRealVola[i] is None:
                    print('(*) Note: Unchanged price set to 0.01 and division by zero to NaN...')
                # Size (milan) adjusted
                sizeShare[i] = shareTraded[i] / (mddv[i] *1000000 * sumVoluW / close[i]) # convert MDDV (mln) to $1
                # duration
                durationFromLast[i] = i-j-1
                break
            else:
                # sum-up weights
                sumVolaW += histVolaW[j]*histVolaW[j]
                sumVoluW += histVoluW[j]
                # keep all max/min
                if (maxP[j]!=0) & (minP[j]!=0): # fix the high volatility ~2 when minP=0 (2019/04/08)
                    allmaxP.append(maxP[j])
                    allminP.append(minP[j])
                # next iteration
                j += -1

    ## Update the very1stTrade that happened at Bin1
    iupdate = df2go.index[(df2go.firstTrade == 1) & (df2go.bin == 1)]
    for i in iupdate:
        very1stTrade[i] = 1
        very1stTrade[i+1:i+78] = 0 # erase the other ones in the day since there should be only one on the day (IMPORTANT: Assume bins are in order)
    print('- Update very1stTrade that happened at Bin1...')

    # Update dataframe
    df2go['adjBM'] = newBM
    df2go['adjHistVola5min'] = newHistVola5min
    df2go['adjHistVolu5min'] = newHistVolu5min
    df2go['adjRealVola'] = newRealVola
    df2go['sizeShare'] = sizeShare0
    df2go['adjSizeShare'] = sizeShare
    df2go['gapDuration'] = durationFromLast
    df2go['very1stTrade'] = very1stTrade

    return df2go

def ji_p5_get_gapTrades_gapDistance(df2go, lastBin=78):
    """
    gapTrades: nonTraded bins that happen during the parent order execution
    gapDistance: distance between the gapBin and the previous lastTraded bin
    gapLastTradedSize: sizeShare of the lastTraded bin
    """
    print('- Finding the gapTrades and gapDuration by iterating through bins...')
    gapTrades = df2go.nonTraded.copy().values
    lastTraded = df2go.lastTraded.copy().values
    sizeShare = df2go.sizeShare.copy().values

    distFromLastTraded = np.zeros(df2go.shape[0])
    sizeLastTraded = np.zeros(df2go.shape[0])

    # auxiliary
    bin = df2go.bin.values
    nfill = df2go.FillNbrForBin.values # indicate execution

    ## 1) Reset all nonTraded that happen after the last trade
    iupdate = df2go.index[(bin == lastBin)]
    print('- Step 1: Number of bins to reset backward ', len(iupdate))
    for i in iupdate:
        #print('i:',i)

        # Search back until end of the day (Todo: optimize)
        #j = i - 1  # one back
        j = i
        while True:
            #print(' j:',j)
            if gapTrades[j]==0: # stop condition
                break
            else:
                gapTrades[j] = 0
                # next iteration
                j += -1
    ## 2) Reset all nonTraded that happen before the first trade
    iupdate = df2go.index[(bin == 1)]
    print('- Step 2: Number of bins to reset forward ', len(iupdate))
    for i in iupdate:
        # Search forward until start of the day (Todo: optimize)
        j = i
        while True:
            if gapTrades[j]==0: # stop condition
                break
            else:
                gapTrades[j] = 0
                # next iteration
                j += 1

    ## 3) Get distance from gapTrades and the lastTraded bin
    iupdate = df2go.index[(gapTrades == 1)]
    print('- Step 3: Number of gapTrades ', len(iupdate))
    for i in iupdate:
        # Search backward until lastTraded (Todo: optimize)
        j = i-1
        dist = 1
        while True:
            if lastTraded[j]==1: # stop condition
                distFromLastTraded[i] = dist
                break
            else:
                distFromLastTraded[i] = dist
                # next iteration
                dist += 1
                j -= 1

    ## 4) Fill size of gapTrades with the size of the lastTraded bin
    iupdate = df2go.index[(lastTraded == 1)]
    print('- Step 4: Number of lastTraded to move forward', len(iupdate))
    for i in iupdate:
        # Search forward while reach non gapTrade (Todo: optimize ?)
        j = i+1
        while True:
            if gapTrades[j]==0: # stop condition
                break
            else:
                sizeLastTraded[j] = sizeShare[i]
                # next iteration
                j += 1

    # Update dataframe
    df2go['gapTrades'] = gapTrades
    df2go['gapDistance'] = distFromLastTraded
    df2go['sizeLastTraded'] = sizeLastTraded

    return df2go

def ji_get_adjusted_milanBM_historicals(df2go): # updated version 2019/04/08
    # Add the milanBM - Working (check 0_df2go_check_milanBM_head2000.xlsx)
    # Checking: 0_df2go_check_milanBM_adjusted_head2000.xlsx
    # Historicals: volatility and volume: sum up weights from the last active bin

    print('- Use the last traded MQ, not the previous one. Sum-up historical vola/volu weights')
    newBM = df2go.priceBM.copy().values
    newHistVola5min = df2go.histVola5min.copy().values
    newHistVolu5min = df2go.histVolu5min.copy().values
    newRealVola = df2go.realVola.copy().values
    durationFromLast = np.zeros(df2go.shape[0])
    very1stTrade = np.zeros(df2go.shape[0])

    # auxiliary
    bin = df2go.bin.values
    MQ = df2go.midQuote.values
    maxP = df2go.HighPrice.values
    minP = df2go.LowPrice.values

    shareTraded = df2go.ExShareForBin.values
    # close = df2go.closeTm1.values # USE THIS AS THIS INFO IS INCLUDED...
    print('- Using open price instead of close price...')
    close = df2go.openPrice.values  # placeholder

    nfill = df2go.FillNbrForBin.values
    # historicals
    histVola = df2go.histVola.values
    mddv = df2go.mddv.values
    histVolaW = df2go.histVolaW.values
    histVoluW = df2go.histVoluW.values

    # Compute sizeShare (milan)
    sizeShare0 = shareTraded / (mddv * 1000000 * histVoluW / close)  # Do not consider gap (convert MDDV to $)
    sizeShare = shareTraded / (mddv * 1000000 * histVoluW / close)  # Do not consider gap (convert MDDV to $)

    iupdate = df2go.index[(df2go.firstTrade == 1) & (df2go.bin != 1)]

    # # Since bin==1 are left out, set the very1stTrade indicator
    # very1stTrade[(df2go.firstTrade == 1) & (df2go.bin == 1)] = 1

    print('- Number of priceBM to update (after gap):', len(iupdate))
    nwarning = 0
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
                newHistVola5min[i] = histVola[i] * np.sqrt(
                    sumVolaW)  # histVola * sqrt(sum(weights^2)) : accumulated weight
                newHistVolu5min[i] = mddv[i] * sumVoluW  # MDDV * sum(shares): accumulated weights
                # compute HL volatility
                MQ0 = MQ[j + 1]  # last.activeBin+1
                #newRealVola[i] = jifin.compute_vola_maxmin(np.max(allmaxP), np.min(allminP), MQ0, MQ1)
                newRealVola[i] = jifin.compute_vola_maxmin_silent(np.max(allmaxP), np.min(allminP), MQ0, MQ1)
                if not newRealVola[i]:
                    nwarning += 1

                # Size (milan) adjusted
                sizeShare[i] = shareTraded[i] / (mddv[i] * 1000000 * sumVoluW / close[i])  # convert MDDV (mln) to $1
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

    if nwarning>0:
        print('(*) Note: Number of special vola case: %d. Unchanged price set to 0.01 or division by zero to NaN...'%(nwarning))

    ## Update the very1stTrade that happened at Bin1
    iupdate = df2go.index[(df2go.firstTrade == 1) & (df2go.bin == 1)]
    for i in iupdate:
        very1stTrade[i] = 1
        very1stTrade[
        i + 1:i + 78] = 0  # erase the other ones in the day since there should be only one on the day (IMPORTANT: Assume bins are in order)
    print('- Update very1stTrade that happened at Bin1...')

    # Update dataframe
    df2go['adjBM'] = newBM
    df2go['adjHistVola5min'] = newHistVola5min
    df2go['adjHistVolu5min'] = newHistVolu5min
    df2go['adjRealVola'] = newRealVola
    df2go['sizeShare'] = sizeShare0
    df2go['adjSizeShare'] = sizeShare
    df2go['gapDuration'] = durationFromLast
    df2go['very1stTrade'] = very1stTrade

    return df2go

def ji_get_adjusted_milanBM_historicals_previous(df2go): # version with BUG (April 5)
    # Add the milanBM - Working (check 0_df2go_check_milanBM_head2000.xlsx)
    # Checking: 0_df2go_check_milanBM_adjusted_head2000.xlsx
    # Historicals: volatility and volume: sum up weights from the last active bin
    print('- Use the last traded MQ, not the previous one. Sum-up hostorical vola/volu weights')
    newBM = df2go.priceBM.copy().values
    newHistVola5min = df2go.histVola5min.copy().values
    newHistVolu5min = df2go.histVolu5min.copy().values
    newRealVola = df2go.realVola.copy().values
    durationFromLast = np.zeros(df2go.shape[0])
    very1stTrade = np.zeros(df2go.shape[0])

    # auxiliary
    bin = df2go.bin.values
    MQ = df2go.midQuote.values
    maxP = df2go.HighPrice.values
    minP = df2go.LowPrice.values

    shareTraded = df2go.ExShareForBin.values
    #close = df2go.closeTm1.values # USE THIS AS THIS INFO IS INCLUDED...
    print('- Using open price instead of close price...')
    close = df2go.openPrice.values # placeholder

    nfill = df2go.FillNbrForBin.values
    # historicals
    histVola = df2go.histVola.values
    mddv = df2go.mddv.values
    histVolaW = df2go.histVolaW.values
    histVoluW = df2go.histVoluW.values

    # Compute sizeShare (milan)
    sizeShare0 = shareTraded/(mddv*1000000*histVoluW/close) # Do not consider gap (convert MDDV to $)
    sizeShare = shareTraded / (mddv * 1000000 * histVoluW / close)  # Do not consider gap (convert MDDV to $)

    iupdate = df2go.index[(df2go.firstTrade == 1) & (df2go.bin != 1)]
    # Since bin==1 are left out, set the very1stTrade indicator
    very1stTrade[(df2go.firstTrade == 1) & (df2go.bin == 1)] = 1

    print('- Number of priceBM to update:', len(iupdate))
    for i in iupdate:
        # print('i:',i)
        # Initialize
        sumVolaW = histVolaW[i]*histVolaW[i]
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
                newBM[i] = MQ[j] # last active bin price
                newHistVola5min[i] = histVola[i]*np.sqrt(sumVolaW) # histVola * sqrt(sum(weights^2)) : accumulated weight
                newHistVolu5min[i] = mddv[i]*sumVoluW # MDDV * sum(shares): accumulated weights
                # compute HL volatility
                MQ0 = MQ[j+1] #last.activeBin+1
                newRealVola[i] = jifin.compute_vola_maxmin(np.max(allmaxP),np.min(allminP),MQ0,MQ1)
                # Size (milan) adjusted
                sizeShare[i] = shareTraded[i] / (mddv[i] *1000000 * sumVoluW / close[i]) # convert MDDV (mln) to $1
                # duration
                durationFromLast[i] = i-j-1
                break
            else:
                # sum-up weights
                sumVolaW += histVolaW[j]*histVolaW[j]
                sumVoluW += histVoluW[j]
                # keep all max/min
                allmaxP.append(maxP[j])
                allminP.append(minP[j])
                # next iteration
                j += -1
    # Update dataframe
    df2go['adjBM'] = newBM
    df2go['adjHistVola5min'] = newHistVola5min
    df2go['adjHistVolu5min'] = newHistVolu5min
    df2go['adjRealVola'] = newRealVola
    df2go['sizeShare'] = sizeShare0
    df2go['adjSizeShare'] = sizeShare
    df2go['gapDuration'] = durationFromLast
    df2go['very1stTrade'] = very1stTrade
    return df2go

def ji_compute_adjusted_Returns_surprises(df2go):
    print('=== Get adjusted bmPrice and historicals and compute:')
    print('- adjNormSignedReturn')
    print('- adjSurVola')
    print('- adjSurVolume')
    df2go['adjSignedReturn'] = np.vectorize(jifin.compute_signedReturn5min_silent)(df2go['midQuote'], df2go['adjBM'],df2go['sign'])
    df2go['adjSurVola'] = np.vectorize(jifin.my_divide)(df2go['adjRealVola'], df2go['adjHistVola5min'])
    df2go['adjSurVolume'] = np.vectorize(jifin.my_divide)(df2go['dollarVolume'], df2go['adjHistVolu5min'])
    df2go['normAdjSignedReturn'] = np.vectorize(jifin.my_divide)(df2go['adjSignedReturn'], df2go['adjHistVola5min']) / 10000 # remove bps
    # Checking
    nn = np.sum(df2go['adjSignedReturn'].isnull())
    if nn > 0:
        print('(*) Note: Number of missing bmPrice where adjSignedReturn was set to None: %d...' % nn)
        print('(*) Note: Number of missing bmPrice where adjSignedReturn was set to None: %d...' % nn)

    return df2go

def ji_compute_normalized_CostReturn(df2go):
    # Add regular normalized EmpCost and Return normalized by histVola
    df2go['normSignedReturn'] = np.vectorize(jifin.my_divide)(df2go['signedReturn'], df2go['histVola5min']) / 10000 # remove bps
    # Add EmpCost normalized by histVola as well
    df2go['normEmpCost'] = np.vectorize(jifin.my_divide)(df2go['empCost'], df2go['histVola5min']) / 10000  # remove bps

    return df2go

def ji_wrapper_add_MaxMinNormCost(df2go):
    tic = time.time()
    df2go['empCostMM'] = np.vectorize(jifin.compute_empCostMaxMin)(df2go['ExPriceForBin'], df2go['priceBM'],
                                                                   df2go['HighPrice'], df2go['LowPrice'], df2go['sign'])
    df2go['FcostMM'] = np.vectorize(jifin.compute_FMaxMin)(df2go['midQuote'], df2go['priceBM'], df2go['vwap'],
                                                           df2go['HighPrice'], df2go['LowPrice'], df2go['sign'])
    df2go['GcostMM'] = np.vectorize(jifin.compute_GMaxMin)(df2go['midQuote'], df2go['priceBM'], df2go['vwap'],
                                                           df2go['HighPrice'], df2go['LowPrice'], df2go['sign'])
    print('... MaxMin normalized costs (empCostMM,FcostMM,GcostMM) computed in %2.4f seconds (using vectorizer).' % (time.time() - tic))

    return df2go

def ji_helper_create_df2go_ABD(dfalgo,toInclude):
    """Create a df2go with selected columns """
    df2go = dfalgo[toInclude].reset_index(drop=True)
    df2go = df2go.rename(columns=lambda x: x[:x.find('[')])  # remove the last letters
    # Add the duration and drop-off first and last fill bin info to save space
    df2go['binDuration'] = df2go['endBin'] - df2go['FirstFillBin'] + 1
    df2go = df2go.drop(columns=['FirstFillBin', 'endBin'])

    #df2go['binDuration'] = df2go['lastFillBin'] - df2go['firstFillBin'] + 1
    #df2go = df2go.drop(columns=['firstFillBin', 'lastFillBin'])
    # Make a copy with keys before stacking
    df2go_before = df2go.copy()
    # execution share (vol)
    dfexshare4bin = pd.DataFrame(get_agg2bin_float(dfalgo['aggExSharesForBin[54]']), columns=range(1, 79)).astype(int)
    # execution price
    dfexprince4bin = pd.DataFrame(get_agg2bin_float(dfalgo['aggExPricesForBin[55]']), columns=range(1, 79))
    # fill number
    dffill4bin = pd.DataFrame(get_agg2bin_float(dfalgo['aggFillNbrForBin[57]']), columns=range(1, 79)).astype(int)

    # a) Stack Ex-price bins to columns
    df2go = pd.concat([df2go, dfexprince4bin], axis=1)  # Join exe-price (concat column to stack)
    #df2go = df2go.set_index(['itgID', 'ticker', 'sign','tradeDate', 'mddv', 'histVola','openPrice'])
    #indexcols = [x[:x.find('[')] for x in toInclude]
    indexcols = list(df2go_before.columns)
    print('Using following as index:',indexcols)
    df2go = df2go.set_index(indexcols)
    #print(df2go_before.columns.values)
    #df2go = df2go.set_index(df2go_before.columns.values)
    df2go.columns.name = 'bin'
    df2go = df2go.stack()  # become a series
    df2go.name = 'ExPriceForBin'
    df2go = df2go.reset_index()
    # b) Stack Ex-share bins and then concat
    dfx = dfexshare4bin.stack()
    dfx.name = 'ExShareForBin'
    dfx = dfx.reset_index(drop=True)
    df2go = pd.concat([df2go, dfx], axis=1)  # Join exe-price (concat column to stack)
    # c) Stack Fill Nbr and then concat
    dfx = dffill4bin.stack()
    dfx.name = 'FillNbrForBin'
    dfx = dfx.reset_index(drop=True)
    df2go = pd.concat([df2go, dfx], axis=1)  # Join fill number (concat column to stack)

    return df2go,df2go_before

def ji_helper_create_df2go(dfalgo,toInclude):
    """Create a df2go with selected columns """
    df2go = dfalgo[toInclude].reset_index(drop=True)
    df2go = df2go.rename(columns=lambda x: x[:x.find('[')])  # remove the last letters
    # Add the duration and drop-off first and last fill bin info to save space
    df2go['binDuration'] = df2go['lastFillBin'] - df2go['firstFillBin'] + 1
    df2go = df2go.drop(columns=['firstFillBin', 'lastFillBin'])

    #df2go['binDuration'] = df2go['lastFillBin'] - df2go['firstFillBin'] + 1
    #df2go = df2go.drop(columns=['firstFillBin', 'lastFillBin'])
    # Make a copy with keys before stacking
    df2go_before = df2go.copy()
    # execution share (vol)
    dfexshare4bin = pd.DataFrame(get_agg2bin_float(dfalgo['aggExSharesForBin[54]']), columns=range(1, 79)).astype(int)
    # execution price
    dfexprince4bin = pd.DataFrame(get_agg2bin_float(dfalgo['aggExPricesForBin[55]']), columns=range(1, 79))
    # fill number
    dffill4bin = pd.DataFrame(get_agg2bin_float(dfalgo['aggFillNbrForBin[57]']), columns=range(1, 79)).astype(int)

    # a) Stack Ex-price bins to columns
    df2go = pd.concat([df2go, dfexprince4bin], axis=1)  # Join exe-price (concat column to stack)
    #df2go = df2go.set_index(['itgID', 'ticker', 'sign','tradeDate', 'mddv', 'histVola','openPrice'])
    #indexcols = [x[:x.find('[')] for x in toInclude]
    indexcols = list(df2go_before.columns)
    print('Using following as index:',indexcols)
    df2go = df2go.set_index(indexcols)
    #print(df2go_before.columns.values)
    #df2go = df2go.set_index(df2go_before.columns.values)
    df2go.columns.name = 'bin'
    df2go = df2go.stack()  # become a series
    df2go.name = 'ExPriceForBin'
    df2go = df2go.reset_index()
    # b) Stack Ex-share bins and then concat
    dfx = dfexshare4bin.stack()
    dfx.name = 'ExShareForBin'
    dfx = dfx.reset_index(drop=True)
    df2go = pd.concat([df2go, dfx], axis=1)  # Join exe-price (concat column to stack)
    # c) Stack Fill Nbr and then concat
    dfx = dffill4bin.stack()
    dfx.name = 'FillNbrForBin'
    dfx = dfx.reset_index(drop=True)
    df2go = pd.concat([df2go, dfx], axis=1)  # Join fill number (concat column to stack)

    return df2go,df2go_before

def ji_helper_add_dailyDOB_in_df2go(df2go,df2go_before,mymeasure,iso='USA'):
    print('=== Adding %s...'%(mymeasure))
    # 1) Get index from the cluster data
    toInclude = ['itgID', 'tradeDate']
    dfx = df2go_before[toInclude].reset_index(drop=True)
    # dfx[dfx.tradeDate==20170103].drop(columns=['tradeDate']).head()
    # 2) Add rows (DailyDOB) given {iso,itgID,tradeDate}
    dfall = pd.DataFrame()
    missingValue = False
    xdates = []
    for cDate in dfx.tradeDate.unique():
        #print('- cDate:', cDate)
    #for cDate in [20171229]:
        # get daily DOB
        dfdob = ji_read_DailyDOB_5min(iso, mymeasure, datetime.datetime.strptime(str(cDate), '%Y%m%d'))
        dfdob = dfdob.rename(columns={'ITGid': 'itgID'}).drop(
            columns=['Ticker', 'Long ID', 'Primary Exchange', 'bin1', 'bin2', 'bin3', 'bin82', 'bin83', 'bin84',
                     'bin85'])
        # merge with DOB for ids of interest
        dfaux = dfx[dfx.tradeDate == int(cDate)].reset_index()
        #dfmerged = pd.merge(dfaux, dfdob, on=['itgID']).drop(columns=['index']) # Does not handle missing values... DO NOT USE
        dfmerged = dfaux.merge(dfaux.merge(dfdob, how='left', on='itgID', sort=False)).drop(columns=['index'])  # 2019/04/22: handle missing values and keep original order

    #print(cDate, ': merged df1:', dfdob.shape, ' with df2:', dfaux.shape, '--> df3:', dfmerged.shape)
        #print(cDate, dfdob.shape, '+', dfaux.shape, '=', dfmerged.shape,end='')
        #print('%d:%d+'%(cDate,dfmerged.shape[1]), end='')

        # Check Special Days, ususually have zeros...
        if dfmerged.isnull().values.any():
            #print('\n(*) Missing DOB value! Likely SpecialDay or some problem %d...' % (cDate))
            #dfaux.to_csv('0_dfaux_tmp.csv')
            #dfmerged.to_csv('0_dfmerged_tmp.csv')
            missingValue = True
            xdates.append(cDate)

        print('%d+' %(dfmerged.shape[0]), end='')
        # dfmerged.head()
        dfall = dfall.append(dfmerged)
    print('')
    if missingValue:
        print('(*) Missing dailyDOB:%s observed on:'%mymeasure,xdates)
    # 3) Add the {measure} to df2go
    # FILL ZEROS (for Special Days)
    dfall = dfall.fillna(0)
    #   Stack {measure} bins and then concat
    dfx = dfall.drop(columns=['itgID', 'tradeDate']).stack()
    dfx.name = mymeasure
    dfx = dfx.reset_index(drop=True)
    #print('')
    return pd.concat([df2go, dfx], axis=1)  # Join (concat column to stack)

def ji_helper_add_dailyDOB_in_df2go_debug(df2go,df2go_before,mymeasure,iso='USA'):
    print('- Adding %s...'%(mymeasure))
    # 1) Get index from the cluster data
    toInclude = ['itgID', 'tradeDate']
    dfx = df2go_before[toInclude].reset_index(drop=True)
    # dfx[dfx.tradeDate==20170103].drop(columns=['tradeDate']).head()
    # 2) Add rows (DailyDOB) given {iso,itgID,tradeDate}
    dfall = pd.DataFrame()
    for cDate in dfx.tradeDate.unique():
        print('- Adding %d...'%cDate)
    #for cDate in [20171229]:
        # get daily DOB
        dfdob = ji_read_DailyDOB_5min(iso, mymeasure, datetime.datetime.strptime(str(cDate), '%Y%m%d'))
        dfdob = dfdob.rename(columns={'ITGid': 'itgID'}).drop(
            columns=['Ticker', 'Long ID', 'Primary Exchange', 'bin1', 'bin2', 'bin3', 'bin82', 'bin83', 'bin84',
                     'bin85'])
        # Fill in tickets with missing prices...
        #dfdob = dfdob.fillna(0)

        # merge with DOB for ids of interest
        dfaux = dfx[dfx.tradeDate == int(cDate)].reset_index()

        dfdob.to_csv('0_dfdob_20181026_GPN_Buy_ITMAA.csv')
        dfaux.to_csv('0_dfaux_20181026_GPN_Buy_ITMAA.csv')

        #dfmerged = pd.merge(dfaux, dfdob, on=['itgID']).drop(columns=['index'])
        dfmerged = dfaux.merge(dfaux.merge(dfdob, how='left', on='itgID', sort=False)).drop(columns=['index'])  # 2019/04/22: handle missing values and keep original order

        #print(cDate, ': merged df1:', dfdob.shape, ' with df2:', dfaux.shape, '--> df3:', dfmerged.shape)
        #print(cDate, dfdob.shape, '+', dfaux.shape, '=', dfmerged.shape,end='')
        #print('%d:%d+'%(cDate,dfmerged.shape[1]), end='')

        dfmerged.to_csv('0_dfmerged_20181026_GPN_Buy_ITMAA.csv')

        # Check Special Days, ususually have zeros...
        if np.sum((dfmerged == 0).astype(int).sum(axis=1)) != 0:
            print('\n(*) Check! Likely SpecialDay or some problem %d...' % (cDate))

        print('%d+' %(dfmerged.shape[0]), end='')
        # dfmerged.head()
        dfall = dfall.append(dfmerged)
    # 3) Add the {measure} to df2go
    # FILL ZEROS (for Special Days)
    dfall = dfall.fillna(0)
    dfall.to_csv('0_dfall_20181026_GPN_Buy_ITMAA.csv')

    #   Stack {measure} bins and then concat
    dfx = dfall.drop(columns=['itgID', 'tradeDate']).stack()
    dfx.name = mymeasure
    dfx = dfx.reset_index(drop=True)

    dfx.to_csv('0_dfx_20181026_GPN_Buy_ITMAA.csv')
    df2go.to_csv('0_df2go_20181026_GPN_Buy_ITMAA_previous.csv')

    print('')
    return pd.concat([df2go, dfx], axis=1)  # Join (concat column to stack)

def ji_helper_add_intradayStats15Sec_in_df2go(df2go,df2go_before,mymeasure='quadVola',iso='USA'):
    print('- Compute and add %s...'%(mymeasure))
    # 1) Get index from the cluster data
    toInclude = ['itgID', 'tradeDate']
    dfx = df2go_before[toInclude].reset_index(drop=True)
    # dfx[dfx.tradeDate==20170103].drop(columns=['tradeDate']).head()
    # 2) Add rows (DailyDOB) given {iso,itgID,tradeDate}
    dfall = pd.DataFrame()
    pretradeHours = ['bin%d'%i for i in range(1,42+1)]
    postradeHours = ['bin%d' % i for i in range(1603,1662+1)]

    for cDate in dfx.tradeDate.unique():
    #for cDate in [20171229]:
        # get daily DOB
        #dfdob = ji_read_DailyDOB_5min(iso, mymeasure, datetime.datetime.strptime(str(cDate), '%Y%m%d'))
        dfdob = ji_read_intradayStats15Sec('USA', 'midQuote', datetime.datetime.strptime(str(cDate), '%Y%m%d'))
        dfdob = dfdob.rename(columns={'ITGid': 'itgID'}).drop(
            columns=['Ticker', 'Long ID', 'Primary Exchange']+pretradeHours+postradeHours)
        # merge with DOB for ids of interest
        dfaux = dfx[dfx.tradeDate == int(cDate)].reset_index()
        dfmerged = pd.merge(dfaux, dfdob, on=['itgID']).drop(columns=['index'])
        #print(cDate, ': merged df1:', dfdob.shape, ' with df2:', dfaux.shape, '--> df3:', dfmerged.shape)
        #print(cDate, dfdob.shape, '+', dfaux.shape, '=', dfmerged.shape,end='')
        #print('%d:%d+'%(cDate,dfmerged.shape[1]), end='')

        # Check Special Days, ususually have zeros...
        if np.sum((dfmerged == 0).astype(int).sum(axis=1)) != 0:
            print('** Warning! Likely SpecialDay or some problem %d. Check!! (IntradayStats15sec) **' % (cDate))

        # Create custom aggregate metric (from 15sec to 5min)
        dfmerged = ji_helper_compute_vola_quad(dfmerged) # convert to 5min bins volatility

        print('%d+' %(dfmerged.shape[0]), end='')
        # dfmerged.head()
        dfall = dfall.append(dfmerged)
    # 3) Add the {measure} to df2go
    # FILL ZEROS (for Special Days)
    dfall = dfall.fillna(0)
    #   Stack {measure} bins and then concat
    dfx = dfall.drop(columns=['itgID', 'tradeDate']).stack()
    dfx.name = mymeasure
    dfx = dfx.reset_index(drop=True)
    print('')
    return pd.concat([df2go, dfx], axis=1)  # Join (concat column to stack)

def ji_helper_add_Return_in_df2go(df2go,iso='USA', silent=False):
    print('=== Computing return (MQ assumed to be constant within 5min bin)...')

    if silent:
        df2go['return'] = np.vectorize(jifin.compute_return5min_silent)(df2go['midQuote'], df2go['priceBM'])
        df2go['signedReturn'] = np.vectorize(jifin.compute_signedReturn5min_silent)(df2go['midQuote'], df2go['priceBM'],df2go['sign'])
    else:
        df2go['return'] = np.vectorize(jifin.compute_return5min)(df2go['midQuote'], df2go['priceBM'])
        df2go['signedReturn'] = np.vectorize(jifin.compute_signedReturn5min)(df2go['midQuote'], df2go['priceBM'], df2go['sign'])

    # Checking
    nn = np.sum(df2go['return'].isnull())
    if nn > 0:
        print('(*) Note: Number of missing bmPrice where Return was set to None: %d...' % nn)
    nn = np.sum(df2go['signedReturn'].isnull())
    if nn > 0:
        print('(*) Note: Number of missing bmPrice where signedReturn was set to None: %d...' % nn)

    return df2go


def ji_helper_add_cumReturn15sec_in_df2go(df2go,df2go_before,iso='USA'):
    print('- Computing cumulative return (based-on 15sec)...')
    # 1) Get index from the cluster data
    toInclude = ['itgID', 'tradeDate']
    dfx = df2go_before[toInclude].reset_index(drop=True)
    # dfx[dfx.tradeDate==20170103].drop(columns=['tradeDate']).head()

    # 2) Add rows (DailyDOB) given {iso,itgID,tradeDate}
    dfall = pd.DataFrame()
    dfall_merged = pd.DataFrame()
    pretradeHours = ['bin%d'%i for i in range(1,42+1)]
    postradeHours = ['bin%d' % i for i in range(1603,1662+1)]

    for cDate in dfx.tradeDate.unique():
    #for cDate in [20171229]:
        # get daily DOB
        #dfdob = ji_read_DailyDOB_5min(iso, mymeasure, datetime.datetime.strptime(str(cDate), '%Y%m%d'))
        dfdob = ji_read_intradayStats15Sec('USA', 'midQuote', datetime.datetime.strptime(str(cDate), '%Y%m%d'))
        dfdob = dfdob.rename(columns={'ITGid': 'itgID'}).drop(
            columns=['Ticker', 'Long ID', 'Primary Exchange']+pretradeHours+postradeHours)
        # merge with DOB for ids of interest
        dfaux = dfx[dfx.tradeDate == int(cDate)].reset_index()
        dfmerged = pd.merge(dfaux, dfdob, on=['itgID']).drop(columns=['index'])
        #print(cDate, ': merged df1:', dfdob.shape, ' with df2:', dfaux.shape, '--> df3:', dfmerged.shape)
        #print(cDate, dfdob.shape, '+', dfaux.shape, '=', dfmerged.shape,end='')
        #print('%d:%d+'%(cDate,dfmerged.shape[1]), end='')

        # Check Special Days, ususually have zeros...
        if np.sum((dfmerged == 0).astype(int).sum(axis=1)) != 0:
            print('** Warning! Likely SpecialDay or some problem %d. Check!! (IntradayStats15sec) **' % (cDate))

        # Create custom aggregate metric (from 15sec to 5min)
        dfmerged_return = ji_helper_compute_return15sec(dfmerged)  # compute cumulative return
#        dfmerged_quad = ji_helper_compute_vola_quad(dfmerged) # convert to 5min bins volatility

        print('%d+' %(dfmerged.shape[0]), end='')
        dfall = dfall.append(dfmerged_return)
        #dfall_merged = dfall_merged.append(dfmerged)

    print('')

    # 3) Add the {measure} to df2go
    # FILL ZEROS (for Special Days)
    dfall = dfall.fillna(0)

#    return dfall,dfall_merged

    #   Stack {measure} bins and then concat
    dfx = dfall.drop(columns=['itgID', 'tradeDate']).stack()
    dfx.name = 'cumReturn15sec'
    dfx = dfx.reset_index(drop=True)

    return pd.concat([df2go, dfx], axis=1)  # Join (concat column to stack)


def ji_helper_add_quadVola_stdVola_15Sec_in_df2go(df2go,df2go_before,iso='USA'):
    print('- Computing and adding Quadratic and Std-based Volatility...')
    # 1) Get index from the cluster data
    toInclude = ['itgID', 'tradeDate']
    dfx = df2go_before[toInclude].reset_index(drop=True)
    # dfx[dfx.tradeDate==20170103].drop(columns=['tradeDate']).head()
    # 2) Add rows (DailyDOB) given {iso,itgID,tradeDate}
    dfall = pd.DataFrame()
    dfall_std = pd.DataFrame()
    pretradeHours = ['bin%d'%i for i in range(1,42+1)]
    postradeHours = ['bin%d' % i for i in range(1603,1662+1)]

    for cDate in dfx.tradeDate.unique():
    #for cDate in [20171229]:
        # get daily DOB
        #dfdob = ji_read_DailyDOB_5min(iso, mymeasure, datetime.datetime.strptime(str(cDate), '%Y%m%d'))
        dfdob = ji_read_intradayStats15Sec('USA', 'midQuote', datetime.datetime.strptime(str(cDate), '%Y%m%d'))
        dfdob = dfdob.rename(columns={'ITGid': 'itgID'}).drop(
            columns=['Ticker', 'Long ID', 'Primary Exchange']+pretradeHours+postradeHours)
        # merge with DOB for ids of interest
        dfaux = dfx[dfx.tradeDate == int(cDate)].reset_index()
        dfmerged = pd.merge(dfaux, dfdob, on=['itgID']).drop(columns=['index'])
        #print(cDate, ': merged df1:', dfdob.shape, ' with df2:', dfaux.shape, '--> df3:', dfmerged.shape)
        #print(cDate, dfdob.shape, '+', dfaux.shape, '=', dfmerged.shape,end='')
        #print('%d:%d+'%(cDate,dfmerged.shape[1]), end='')

        # Check Special Days, ususually have zeros...
        if np.sum((dfmerged == 0).astype(int).sum(axis=1)) != 0:
            print('** Warning! Likely SpecialDay or some problem %d. Check!! (IntradayStats15sec) **' % (cDate))

        # Create custom aggregate metric (from 15sec to 5min)
        dfmerged_quad = ji_helper_compute_vola_quad(dfmerged) # convert to 5min bins volatility
        dfmerged_std = ji_helper_compute_vola_std(dfmerged)  # convert to 5min bins volatility

        print('%d+' %(dfmerged.shape[0]), end='')
        # dfmerged.head()
        dfall = dfall.append(dfmerged_quad)
        dfall_std = dfall_std.append(dfmerged_std)

    print('')

    # 3) Add the {measure} to df2go
    # FILL ZEROS (for Special Days)
    dfall = dfall.fillna(0)
    #   Stack {measure} bins and then concat
    dfx = dfall.drop(columns=['itgID', 'tradeDate']).stack()
    dfx.name = 'quadVola'
    dfx = dfx.reset_index(drop=True)

    # 4) Add the stdVola to df2go
    # FILL ZEROS (for Special Days)
    dfall_std = dfall_std.fillna(0)
    #   Stack {measure} bins and then concat
    dfx_std = dfall_std.drop(columns=['itgID', 'tradeDate']).stack()
    dfx_std.name = 'stdVola'
    dfx_std = dfx_std.reset_index(drop=True)


    return pd.concat([df2go, dfx, dfx_std], axis=1)  # Join (concat column to stack)

def ji_helper_add_histVolu_in_df2go(df2go,df2go_before,iso='USA'):
    print('=== Adding historical volume (5min bins)...')
    # 1) Get index from the cluster data
    #toInclude = ['itgID', 'tradeDate', 'histVola']
    #toInclude = ['itgID', 'tradeDate','mddv']
    toInclude = ['ticker', 'itgID', 'tradeDate', 'mddv']
    dfx = df2go_before[toInclude].reset_index(drop=True)
    # dfx[dfx.tradeDate==20170103].drop(columns=['tradeDate']).head()
    # 1.2) Get the start of month
    startmonth = [df2go_before.tradeDate.values[0]]
    for i in df2go_before.tradeDate.values:
        if str(i)[-3] != str(startmonth[-1])[-3]:
            startmonth.append(i)
    # 2) Add histVola5min rows given {iso,itgID,tradeDate}
    dfall = pd.DataFrame()
    for cDate in startmonth:
        # get monthly histVolu
        #dfvola = ji_read_voluWeights_5min(iso,datetime.datetime.strptime(str(cDate), '%Y%m%d')).stack().reset_index(drop=True)
        dfvola = ji_read_voluWeights_5min(iso, datetime.datetime.strptime(str(cDate), '%Y%m%d'))
        dfvola = dfvola.drop(columns=['LongID', 'coef_group', 'group', 'MOO', 'MOC', 'bin0', 'bin1', 'bin2', 'bin81', 'bin82', 'bin83','bin84'])
        dfvola = dfvola.rename(columns={'Ticker': 'ticker'})

        # merge with DOB for ids of interest
        # dfaux = dfx[dfx.tradeDate == int(cDate)].reset_index()
        dfaux = dfx[dfx.tradeDate.values // 100 == cDate // 100].reset_index(
            drop=True)  # select the ones within the month
        # dfmerged = pd.merge(dfaux, dfdob, on=['itgID']).drop(columns=['index'])
        #dfmerged = pd.concat([dfaux, pd.DataFrame(np.tile(dfvola.values, (dfaux.shape[0], 1)))], axis=1)
        dfmerged = pd.merge(dfaux, dfvola, on=['ticker']).drop(columns=['ticker'])

        print('%d+' % (dfmerged.shape[0]), end='')
        # dfmerged.head()
        dfall = dfall.append(dfmerged)
    print('=%d clusters'%(dfall.shape[0]))
         # 3) Add the {measure} to df2go
    #   Stack {measure} bins and then concat
    dfxx = dfall.drop(columns=['itgID', 'tradeDate','mddv']).stack()
    dfxx.name = 'histVoluW'
    dfxx = dfxx.reset_index(drop=True)
    #print('')
    df2go = pd.concat([df2go, dfxx], axis=1)  # Join (concat column to stack)
    # 4) Compute the actual histVolu: w's x vola
    #if 'histVola5min' in df2go:
    if 'histVolu5min' in df2go:
        print('... histVolu5min already exists. Do nothing...')
    else:
        df2go['histVolu5min'] = df2go.mddv * df2go.histVoluW
        print('... histVolu5min added.')
    #df2go.to_csv('df2go_step3.csv')
    return df2go

def ji_helper_add_histVola_in_df2go(df2go,df2go_before,iso='USA'):
    print('=== Adding historical vola (5min bins)...')
    # 1) Get index from the cluster data
    toInclude = ['itgID', 'tradeDate', 'histVola']
    dfx = df2go_before[toInclude].reset_index(drop=True)
    # dfx[dfx.tradeDate==20170103].drop(columns=['tradeDate']).head()
    # 1.2) Get the start of month
    startmonth = [df2go_before.tradeDate.values[0]]
    for i in df2go_before.tradeDate.values:
        if str(i)[-3] != str(startmonth[-1])[-3]:
            startmonth.append(i)
    # 2) Add histVola5min rows given {iso,itgID,tradeDate}
    dfall = pd.DataFrame()
    for cDate in startmonth:
        # get monthly histVola
        dfvola = ji_read_volaWeights_5min(iso,
                                              datetime.datetime.strptime(str(cDate), '%Y%m%d')).stack().reset_index(
            drop=True)
        # merge with DOB for ids of interest
        # dfaux = dfx[dfx.tradeDate == int(cDate)].reset_index()
        dfaux = dfx[dfx.tradeDate.values // 100 == cDate // 100].reset_index(
            drop=True)  # select the ones within the month
        # dfmerged = pd.merge(dfaux, dfdob, on=['itgID']).drop(columns=['index'])
        dfmerged = pd.concat([dfaux, pd.DataFrame(np.tile(dfvola.values, (dfaux.shape[0], 1)))], axis=1)
        print('%d+' % (dfmerged.shape[0]), end='')
        # dfmerged.head()
        dfall = dfall.append(dfmerged)
    print('=%d clusters'%(dfall.shape[0]))
         # 3) Add the {measure} to df2go
    #   Stack {measure} bins and then concat
    dfxx = dfall.drop(columns=['itgID', 'tradeDate', 'histVola']).stack()
    dfxx.name = 'histVolaW'
    dfxx = dfxx.reset_index(drop=True)
    #print('')
    df2go = pd.concat([df2go, dfxx], axis=1)  # Join (concat column to stack)
    #df2go_step2 = df2go.copy()
    # df2go_step2.to_csv('df2go_step2.csv')
    # 4) Compute the actual histVola: w's x vola
    if 'histVola5min' in df2go:
        print('... histVola5min already exists. Do nothing...')
    else:
        df2go['histVola5min'] = df2go.histVola * df2go.histVolaW
        print('... histVola5min added.')
    #df2go.to_csv('df2go_step3.csv')
    return df2go

def ji_helper_compute_vola_std(dfmerged):
    """It computes the std-based volatility for all clusters.
    It assumes that they are 15sec and convert to 5min bins.
    I.e., calculates the volatility of 5min bin based on 15sec data."""

    n = 20 # bin size 20x15sec --> 5min
    # Create basic structure
    dfqv = dfmerged[['itgID','tradeDate']].copy()

    # First bin
    M = dfmerged.iloc[:,2:2+n].values
    #qv = np.array([compute_vola_quadratic(M,n)])
    sv = np.array([jifin.compute_vola_quadratic(M, n)])
    cbins = range(2, 78 + 1)  # 5min bin number
    for cbin in cbins:
        M = dfmerged.iloc[:,2+n*(cbin-1):2+n*cbin].values
        #aux = itg.compute_vola_quadratic(M,n)
        #qv = np.concatenate((qv,[compute_vola_quadratic(M,n)]),axis=0)
        sv = np.concatenate((sv, [jifin.compute_vola_std(M, n)]), axis=0)

    return pd.concat([dfqv,pd.DataFrame(sv.T)],axis=1)

def ji_helper_compute_return15sec(dfmerged):
    """It computes the returnfor all clusters.
    It assumes that they are 15sec and convert to 5min bins.
    I.e., calculates the cumulative return of 5min bin based on 15sec data."""

    n = 20 # bin size 20x15sec --> 5min
    # Create basic structure
    dfx = dfmerged[['itgID','tradeDate']].copy()

    # First bin
    M = dfmerged.iloc[:,2:2+n].values
    #qv = np.array([jifin.compute_vola_quadratic(M,n)])
    ret = np.array([jifin.compute_return15sec(M, n)])
    cbins = range(2, 78 + 1)  # 5min bin number
    for cbin in cbins:
        M = dfmerged.iloc[:,2+n*(cbin-1):2+n*cbin].values
        ret = np.concatenate((ret,[jifin.compute_return15sec(M,n)]),axis=0)

    return pd.concat([dfx,pd.DataFrame(ret.T)],axis=1) # in bps

def ji_helper_compute_vola_quad(dfmerged):
    """It computes the quadratic volatility for all clusters.
    It assumes that they are 15sec and convert to 5min bins.
    I.e., calculates the volatility of 5min bin based on 15sec data."""

    n = 20 # bin size 20x15sec --> 5min
    # Create basic structure
    dfqv = dfmerged[['itgID','tradeDate']].copy()

    # First bin
    M = dfmerged.iloc[:,2:2+n].values
    qv = np.array([jifin.compute_vola_quadratic(M,n)])
    cbins = range(2, 78 + 1)  # 5min bin number
    for cbin in cbins:
        M = dfmerged.iloc[:,2+n*(cbin-1):2+n*cbin].values
        #aux = itg.compute_vola_quadratic(M,n)
        qv = np.concatenate((qv,[jifin.compute_vola_quadratic(M,n)]),axis=0)

    return pd.concat([dfqv,pd.DataFrame(qv.T)],axis=1)

def add_priceBM_in_df2go_fast(df2gox):
    """Assume as BM price the previous bin MQ.
    For first bin of the day, use the openPrice"""
    #df2go['priceBM'] = df2go.openPrice
    print('=== Adding priceBM to df. Note: it assumes that bin1 is always following bin78')
    df2go = df2gox.copy()
    # Copy MQ --> BM
    df2go['priceBM'] = df2go.midQuote.shift(1)
    # Correct the bin==1
    df2go.loc[df2go.bin==1,'priceBM'] = df2go.loc[df2go.bin==1,'openPrice']
    return df2go

def add_last30_cum_vola(df2gox):
    """Add cumulative and last 30min volatilities.
    Not efficient... TODO better...
    """
    df2go = df2gox.copy()

    # 1) Iterate by date
    cont = 1
    print('- Adding cum/last30 volatilities (by date and moniker)...')
    # Use np for quicker iteration
#    MQ = dfx.midQuote.values
#    LowP = dfx.LowPrice.values
#    HighP = dfx.HighPrice.values
#    quadVola = dfx.quadVola.values

    last30HLvola = np.zeros(df2go.shape[0])
    cumHLvola = np.zeros(df2go.shape[0])
    last30quadVola = np.zeros(df2go.shape[0])
    cumquadVola = np.zeros(df2go.shape[0])

    for cDate in df2go.tradeDate.unique():
        print('%d-'%cont,end='')
        # 2) Iterate by Moniker
        for id in df2go[df2go.tradeDate==cDate].clientMoniker.unique():

            imask = (df2go.clientMoniker == id) & (df2go.tradeDate == cDate)

            dfx = df2go.loc[imask, ['LowPrice', 'HighPrice', 'midQuote', 'quadVola']].reset_index(drop=True).copy()

            # Square qVola back to further sum
            dfx['quadVola2'] = dfx.quadVola ** 2
            # Create cumulative and last
            dfx['cumminLowPrice'] = dfx.LowPrice.cummin()
            dfx['cummaxHighPrice'] = dfx.HighPrice.cummax()
            dfx['min30LowPrice'] = dfx.LowPrice.rolling(6).min()
            dfx['max30HighPrice'] = dfx.HighPrice.rolling(6).max()
            dfx['first30MQ'] = dfx.midQuote.rolling(6).apply(lambda x: x[0])
            dfx['cumquadVola'] = dfx.quadVola2.cumsum().apply(np.sqrt)
            dfx['last30quadVola'] = dfx.quadVola2.rolling(6).sum().apply(np.sqrt)

            # Fill in last30min with cumulative when NaN
            dfx.loc[:5, 'min30LowPrice'] = dfx.loc[:5, 'cumminLowPrice']
            dfx.loc[:5, 'max30HighPrice'] = dfx.loc[:5, 'cummaxHighPrice']
            dfx.loc[:5, 'last30quadVola'] = dfx.loc[:5, 'cumquadVola']
            # Fill initial first30MQ with very first one
            dfx.loc[:5, 'first30MQ'] = dfx.midQuote.iloc[0]
            # Compute the last 30min and Cumu volatility
            dfx['last30HLvola'] = np.vectorize(jifin.compute_vola_maxmin)(dfx.max30HighPrice, dfx.min30LowPrice, dfx.first30MQ,
                                                                          dfx.midQuote)
            dfx['cumHLvola'] = np.vectorize(jifin.compute_vola_maxmin)(dfx.cummaxHighPrice, dfx.cumminLowPrice,
                                                                       dfx.midQuote.iloc[0], dfx.midQuote)

            last30HLvola[imask] = dfx['last30HLvola'].values
            cumHLvola[imask] = dfx['cumHLvola'].values
            last30quadVola[imask] = dfx['last30quadVola'].values
            cumquadVola[imask] = dfx['cumquadVola'].values
        cont += 1
    df2go['last30HLVola'] = last30HLvola
    df2go['cumHLVola'] = cumHLvola
    df2go['last30quadVola'] = last30quadVola
    df2go['cumquadVola'] = cumquadVola

    return df2go

def add_priceBM(df2gox):
    """Assume as BM price the previous bin MQ.
    For first bin of the day, use the openPrice"""
    #df2go['priceBM'] = df2go.openPrice
    df2go = df2gox.copy()
    # 1) Iterate by date
    cont = 1
    print('- Adding BM price (by date and itgID)...')
    MQ = df2go.midQuote.values
    openPrice = df2go.openPrice.values
    priceBM = np.zeros(len(openPrice))
    for cDate in df2go.tradeDate.unique():
        print('%d-'%cont,end='')
        # 2) Iterate by itgID
        for id in df2go[df2go.tradeDate==cDate].itgID.unique():
            #print('Adding BM price for:%d %s'%(cDate,id))

            #print('Completed clusters (total %d): %d \r' %(cont), end='')
            # Innefficient... use np array
            #df2go[(df2go.tradeDate==cDate)&(df2go.itgID==id)].iloc[1:].priceBM \
            #    = df2go[(df2go.tradeDate==cDate)&(df2go.itgID==id)].iloc[:-2].midQuote
            a = openPrice[(df2go.tradeDate==cDate)&(df2go.itgID==id)]
            b = MQ[(df2go.tradeDate == cDate) & (df2go.itgID == id)]
            a[1:] = b[:-1]
            priceBM[(df2go.tradeDate==cDate)&(df2go.itgID==id)] = a
                #= df2go[(df2go.tradeDate==cDate)&(df2go.itgID==id)].iloc[:-2].midQuote
        cont+=1
    df2go['priceBM'] = priceBM
    return df2go

def fill_missing_priceBM(df2gox, silent=False):
    """Fill in missing price with the previous bin value.
    It assumes that it is a single instance and not the first bin..."""
    df2go = df2gox.copy()
    imissing = df2go[df2go['priceBM'] == 0].index
    #df2go.iloc[imissing].priceBM = df2go.iloc[imissing-1].priceBM
    df2go.loc[imissing,'priceBM'] = df2go.loc[imissing-1,'priceBM']
    print('- Number of missing priceBM filled with previous value: %d...'%len(imissing))
    mlist = []
    for cont,i in enumerate(imissing):
        if silent:
            mlist.append('[%d] %s %d bin:%d, '%(cont,df2go.iloc[i].itgID,df2go.iloc[i].tradeDate,df2go.iloc[i].bin))
        else:
            print('Fill in %s %d bin:%d'%(df2go.iloc[i].itgID,df2go.iloc[i].tradeDate,df2go.iloc[i].bin))
        if df2go.iloc[i].bin==1:
            print('** WARNING! Wrong replacement for bin=1. Check data!! **')
    if not silent:
        print(mlist)
    return df2go


def create_df2pred(df2go,targetName='empCost',mylabel='next5min',shift5min=1):
    """Add a target column to be predicted.
    It will shift back {targetname} by {shift5min}
    - Will add NaN when there is nothing to predict. That is, will add NaN at the end of the transaction.
    - Will remove when empCost to predict is zero (nothing to predict)
    - Dropoff rows without histVolume. Just for simplicity... (ADD LATER...)
    """
    print('######################################################')
    print('## Creating dataframe to be used in prediction #######')
    print('######################################################')
    df2pred = df2go.copy()

    keys = ['tradeDate', 'clientMoniker', 'ticker']
    cols2in = keys + [targetName]
    dfx = df2pred[cols2in]
    dfx = dfx.set_index(keys)
    dfx = dfx.groupby(keys).shift(-1 * shift5min).reset_index()
    dfx = dfx.rename(columns={'%s' % targetName: 'target'})
    # df2go['%s_%s'%(targetName,mylabel)] = pd.concat([df2go,dfx],axis=1)[targetName].values
    newName = '%s_%s' % (targetName, mylabel)
    df2pred[newName] = pd.concat([df2pred, dfx], axis=1).target.values
    # pd.concat([df2go,dfx],axis=1).target
    # pd.merge(df2go,dfx,on=keys,how='inner')
    # df2go.head(1000).to_csv('df2go_DEBUG.csv')
    # df2go.head()

    ## 1) Dropoff Target and leave just the shifted (NO NEED. CAN USE...)
    #df2pred.drop(columns=[targetName])

    ## 2) Excluded non-executed bins (to save space)
    print('- Excluding bins without EmpCost to predict (i.e. there is not trade in the next bin): %d (%d%%)' % (
    np.sum(df2pred[newName] == 0), 100 * np.sum(df2pred[newName] == 0) / df2pred.shape[0]))
    df2pred = df2pred[df2pred[newName] != 0]

    ## 2.5) Excluded NaN bins (the last bin of the day)
    print('- Excluding bins with EmpCost==NaN (i.e. the last bin of the day): %d (%d%%)' % (
    np.sum(df2pred[newName].isnull()), 100 * np.sum(df2pred[newName].isnull()) / df2pred.shape[0]))
    df2pred = df2pred[df2pred[newName].notna()]

    ## 3)
    mask2exclude = df2pred.histVolu5min.isnull()
    print('- Excluding trades without histVolume:%d...' % np.sum(mask2exclude))
    df2pred = df2pred[~mask2exclude].reset_index(drop=True)

    return df2pred

###################################################
## DATABASE
###################################################

class scADB:  # This is connecting to the SQL engine (located in the Windows server)

    def __init__(self):
        self.configs = {
            'server': 'ADM-DBA01-SC',
            'port': '1433',
            'database': 'Analytics',
            'username': 'feprod',
            'password': 'feprod'
        }
        self.connectCmd = 'DRIVER={{ODBC Driver 13 for SQL Server}};SERVER={server};PORT={port};DATABASE={database};UID={username};PWD={password}'.format(
            **self.configs)
        self.connection = pyodbc.connect(self.connectCmd)


    def __del__(self):
        self.connection.close()


    # def getEngineOutput(self,ID,strat,schema = '[ITG\dxia]'): # PREVIOUS
    def get_quadVola15sec(self,tickers,mydate,iso='USA'):
        '''Read Quadratic Volatility for USA'''
        print('- Reading quadratic volatility for USA. For others, ISSUE_ISO has to be updated...')
        query_details = {
            'ID': iso,
            'TICKERSTRING': "','".join(tickers),
            'MYDATE': mydate,
            #'SCHEMA': schema,
            #'STRATEGY': strat
        }
        query = '''
            SELECT
                        --- DATEFORMAT(coeqc.tradingDate,'YYYYMMDD') AS tradeDate,
                        symb.FIGI_TICKER AS ticker,
                        coeqc.tradingDate AS tradeDate,
                        coeqc.binNumber AS bin,
                        coeqc.analyticValue AS quadVola
                        
            FROM vw_CoEqC AS coeqc
            JOIN Country AS cntry ON
                        cntry.ISO3 = '{ID}'
            JOIN SymbolMaster AS symb ON
                        coeqc.DerivativeIssueID = symb.DerivativeIssueID AND
                        coeqc.DerivativeIssueMICID = symb.DerivativeIssueMICID AND
                        symb.FIGI_EXCHANGE_CODE = cntry.BloombergCompositeCode
            JOIN ADB_enum AS analytic ON
                        analytic.Name = 'ANALYTIC_TYPE' AND
                        coeqc.analyticTypeID = analytic.Enum
            WHERE
                        --- coeqc.tradingDate = '2018-07-17' AND
                        coeqc.tradingDate = '{MYDATE}' AND
                        --- coeqc.tradingDate BETWEEN '20171227' AND '20171228' AND
                        symb.ISSUE_ISO = 'US' AND
                        --- symb.FIGI_TICKER = 'ITG' AND
                        symb.FIGI_TICKER in ('{TICKERSTRING}') AND
                        analytic.Type = 'VOLATILITY' AND
                                 binTypeID = 68 AND
                                 coeqc.tradingDate BETWEEN symb.CreationDate AND symb.TerminationDate
            ORDER BY tradeDate
        '''.format(**query_details)
        return pd.read_sql(query, self.connection)


###################################################
## VISUALIZE
###################################################
# Good ref for groupby and pivot-table: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
def get_agg2bin_int(dseries):
    """Given a dataSeries of aggBin, convert it to array of integers"""
    #xx = []
    #for z in dsrow.values:
    #    xx.append([int(x) for x in z.split(":")])
    #return xx
    return [[int(x) for x in z.split(":")] for z in dseries.values]

def get_agg2bin_float(dseries):
    """Given a dataSeries of aggBin, convert it to array of floats"""
    return [[float(x) for x in z.split(":")] for z in dseries.values]

def plot_algoDB_exprice(dfx,dicfilter,aggmethod):
    nn = argparse.Namespace(**d)

###################################################
## READING
###################################################
def ji_load_ABD_clusters_all(dpath,iso,allyy,allqq,colsToRead,d={"nrows":None}):
    nn = argparse.Namespace(**d)
    # Convert to tables indexing
    colsToReadx = [i - 1 for i in colsToRead]
    dfall = []
    for yy in allyy:
        for qq in allqq:
            if nn.nrows is not None:
                dfx = pd.read_csv(Path.joinpath(dpath, f'clustersABD_ALL_{iso}_{yy}{qq}.dat.gz'), compression='gzip',
                                  sep=',', usecols=colsToReadx, na_filter=False, nrows=int(nn.nrows))
            else:
                dfx = pd.read_csv(Path.joinpath(dpath, f'clustersABD_ALL_{iso}_{yy}{qq}.dat.gz'), compression='gzip',
                                  sep=',', usecols=colsToReadx, na_filter=False)

            # Filter by algoType
            # allalgo = [s.split(";") for s in dfx['algoType[49]']]

            # iselect = []
            # for strat in allalgo:
            #     isgood = False
            #     for ss in strat:
            #         if ss.split(":")[0] == 'SCHEDULED':
            #             if int(ss.split(":")[1]) >= nn.minpercent:
            #                 isgood = True
            #     iselect.append(isgood)

            # dfall.append(dfx[iselect])
            # print('Data loaded %s %d%s: Retrieved SCHEDULED(>%d%%) clusters: %d out of %d.' % (
            #            iso, yy, qq, nn.minpercent, sum(iselect), dfx.shape[0]))
            dfall.append(dfx)
    return pd.concat(dfall, axis=0)

def ji_load_algoDB_clusters_notRED(dpath,iso,allyy,allqq,colsToRead,d={"minpercent":100, "nrows":None}):
    nn = argparse.Namespace(**d)
    # Convert to tables indexing
    colsToReadx = [i - 1 for i in colsToRead]
    dfall = []
    for yy in allyy:
        for qq in allqq:
            if nn.nrows is not None:
                dfx = pd.read_csv(Path.joinpath(dpath, f'clustersABD_ALL_{iso}_{yy}{qq}.dat.gz'), compression='gzip',
                                  sep=',', usecols=colsToReadx, na_filter=False, nrows=int(nn.nrows))
            else:
                dfx = pd.read_csv(Path.joinpath(dpath, f'clustersABD_ALL_{iso}_{yy}{qq}.dat.gz'), compression='gzip',
                                  sep=',', usecols=colsToReadx, na_filter=False)
                # Filter by algoType
            allalgo = [s.split(";") for s in dfx['algoType[49]']]

            iselect = []
            for strat in allalgo:
                isgood = False
                for ss in strat:
                    if ss.split(":")[0] == 'SCHEDULED':
                        if int(ss.split(":")[1]) >= nn.minpercent:
                            isgood = True
                iselect.append(isgood)

            # Twoliner (NOT WORKING...)
            # iselectx = [[int(ss.split(":")[1])>=minpercent if ss else False for ss in strat if ss.split(":")[0]=='SCHEDULED'] for strat in allalgo]
            # ix = [True if i else False for i in iselectx]
            # dfx[ix]

            dfall.append(dfx[iselect])
            print('Data loaded %s %d%s: Retrieved SCHEDULED(>%d%%) clusters: %d out of %d.' % (
        iso, yy, qq, nn.minpercent, sum(iselect), dfx.shape[0]))

    return pd.concat(dfall, axis=0)

def ji_load_algoDB_clusters_ALL(dpath,iso,allyy,allqq,colsToRead,d={"nrows":None}):
    nn = argparse.Namespace(**d)
    # Convert to tables indexing
    colsToReadx = [i - 1 for i in colsToRead]
    dfall = []
    for yy in allyy:
        for qq in allqq:
            if nn.nrows is not None:
                dfx = pd.read_csv(Path.joinpath(dpath, f'clustersRED_ALGO_ALL_{iso}_{yy}{qq}.dat.gz'), compression='gzip',
                                  sep=',', usecols=colsToReadx, na_filter=False, nrows=int(nn.nrows))
            else:
                dfx = pd.read_csv(Path.joinpath(dpath, f'clustersRED_ALGO_ALL_{iso}_{yy}{qq}.dat.gz'), compression='gzip',
                                  sep=',', usecols=colsToReadx, na_filter=False)
                # Filter by algoType
            allalgo = [s.split(";") for s in dfx['algoType[49]']]

            # iselect = []
            # for strat in allalgo:
            #     isgood = False
            #     for ss in strat:
            #         if ss.split(":")[0] == 'SCHEDULED':
            #             if int(ss.split(":")[1]) >= nn.minpercent:
            #                 isgood = True
            #     iselect.append(isgood)
            #
            # dfall.append(dfx[iselect])
            # print('Data loaded %s %d%s: Retrieved SCHEDULED(>%d%%) clusters: %d out of %d.' % (iso, yy, qq, nn.minpercent, sum(iselect), dfx.shape[0]))
            dfall.append(dfx)
            print('Data loaded %s %d%s: Retrieved ALL strategy clusters: %d.' % (iso, yy, qq, dfx.shape[0]))

    return pd.concat(dfall, axis=0)

def ji_load_algoDB_clusters(dpath,iso,allyy,allqq,colsToRead,d={"minpercent":100, "nrows":None}):
    nn = argparse.Namespace(**d)
    # Convert to tables indexing
    colsToReadx = [i - 1 for i in colsToRead]
    dfall = []
    for yy in allyy:
        for qq in allqq:
            if nn.nrows is not None:
                dfx = pd.read_csv(Path.joinpath(dpath, f'clustersRED_ALGO_ALL_{iso}_{yy}{qq}.dat.gz'), compression='gzip',
                                  sep=',', usecols=colsToReadx, na_filter=False, nrows=int(nn.nrows))
            else:
                dfx = pd.read_csv(Path.joinpath(dpath, f'clustersRED_ALGO_ALL_{iso}_{yy}{qq}.dat.gz'), compression='gzip',
                                  sep=',', usecols=colsToReadx, na_filter=False)
                # Filter by algoType
            allalgo = [s.split(";") for s in dfx['algoType[49]']]

            iselect = []
            for strat in allalgo:
                isgood = False
                for ss in strat:
                    if ss.split(":")[0] == 'SCHEDULED':
                        if int(ss.split(":")[1]) >= nn.minpercent:
                            isgood = True
                iselect.append(isgood)

            # Twoliner (NOT WORKING...)
            # iselectx = [[int(ss.split(":")[1])>=minpercent if ss else False for ss in strat if ss.split(":")[0]=='SCHEDULED'] for strat in allalgo]
            # ix = [True if i else False for i in iselectx]
            # dfx[ix]

            dfall.append(dfx[iselect])
            print('Data loaded %s %d%s: Retrieved SCHEDULED(>%d%%) clusters: %d out of %d.' % (
        iso, yy, qq, nn.minpercent, sum(iselect), dfx.shape[0]))

    return pd.concat(dfall, axis=0)

def ji_read_voluWeights_5min(iso, date):
    """
    Read the monthly Intraday Volume file for the specified date and ISO. If the file doesn't exists, look back 12 months.
    """
    file_date = date
    months_back = 0
    rtpath = Path("/export/prod/data1")
    # rtpath = Path("$RT") # did not work...
    while True:
        try:
            month_str = file_date.strftime("%Y%m")
            # file = Path(f"/export/prod/data1/HistoricalFiles/IntradayVolume/1.0/{iso}/monthly/vwapFE_5min_{iso}_1.0_{month_str}.dat.gz")
            # file = Path(f"/export/prod/data1/HistoricalFiles/IntradayVolatility/1.0/{iso}/volatilityFE_5min_{iso}_1.0_{month_str}.dat.gz")
            file = Path.joinpath(rtpath,
                                 #f"HistoricalFiles/IntradayVolatility/1.0/{iso}/volatilityFE.FOMC_MINUTES_5min_{iso}_1.0_{month_str}.dat.gz")
                                 #f"HistoricalFiles/IntradayVolatility/1.0/{iso}/volatilityFE.FOMC_5min_{iso}_1.0_{month_str}.dat.gz")
                                 #f"HistoricalFiles/IntradayVolatility/1.0/{iso}/volatilityFE_5min_{iso}_1.0_{month_str}.dat.gz")
                                 f"HistoricalFiles/IntradayVolume/1.0/{iso}/monthly/vwapFE_5min_{iso}_1.0_{month_str}.dat.gz")

            log.info(f"Reading {file}")
            # df = pd.read_csv(file, compression='gzip', sep=',')
            #df = pd.read_csv(file, compression='gzip', header=7, sep=',')
            df = pd.read_csv(file, compression='gzip', header=9, sep=',', quotechar='"')
            df.rename(columns={'#<Column Name>Long ID': 'LongID'}, inplace=True)
            df['Ticker'] = df['Ticker'].astype(str)
        except IOError:
            if months_back > 12:
                log.fatal("Can't find file for the previous 12 months")
                raise
            else:
                file_date = file_date.replace(day=1) - datetime.timedelta(days=1)
                months_back += 1
                log.warn(f"Can't find file for {file_date}. Looking back 1 month")
                continue
        except:
            log.fatal('Unexpected error')
            raise

        # print('Loaded vola weights:',file)
        break

    return df

def ji_read_volaWeights_5min(iso, date):
    """
    Read the monthly Intraday Volatility file for the specified date and ISO. If the file doesn't exists, look back 12 months.
    """
    file_date = date
    months_back = 0
    rtpath = Path("/export/prod/data1")
    # rtpath = Path("$RT") # did not work...
    while True:
        try:
            month_str = file_date.strftime("%Y%m")
            # file = Path(f"/export/prod/data1/HistoricalFiles/IntradayVolume/1.0/{iso}/monthly/vwapFE_5min_{iso}_1.0_{month_str}.dat.gz")
            # file = Path(f"/export/prod/data1/HistoricalFiles/IntradayVolatility/1.0/{iso}/volatilityFE_5min_{iso}_1.0_{month_str}.dat.gz")
            file = Path.joinpath(rtpath,
                                 #f"HistoricalFiles/IntradayVolatility/1.0/{iso}/volatilityFE.FOMC_MINUTES_5min_{iso}_1.0_{month_str}.dat.gz")
                                 #f"HistoricalFiles/IntradayVolatility/1.0/{iso}/volatilityFE.FOMC_5min_{iso}_1.0_{month_str}.dat.gz")
                                 f"HistoricalFiles/IntradayVolatility/1.0/{iso}/volatilityFE_5min_{iso}_1.0_{month_str}.dat.gz")
            log.info(f"Reading {file}")
            # df = pd.read_csv(file, compression='gzip', sep=',')
            df = pd.read_csv(file, compression='gzip', header=6, sep=',')
            # df.rename(columns={'#<Column Name>Long ID': 'LongID'}, inplace=True)
            # df['Ticker'] = df['Ticker'].astype(str)
        except IOError:
            if months_back > 12:
                log.fatal("Can't find file for the previous 12 months")
                raise
            else:
                file_date = file_date.replace(day=1) - datetime.timedelta(days=1)
                months_back += 1
                log.warn(f"Can't find file for {file_date}. Looking back 1 month")
                continue
        except:
            log.fatal('Unexpected error')
            raise

        # print('Loaded vola weights:',file)
        break

    return df

def ji_read_intradayStats15Sec(iso, metric, date):
    """
    Read the intraday MQ in 15sec bins. If the file doesn't exists, look back 10 days.
    """
    file_date = date
    days_back = 0
    #rtpath = Path("/export/prod/data1")
    global path_tickDataUSD15sec
    #rtpath = Path("/home/jide/Data/tickData_USA_IntradayStats15Sec")
    rtpath = Path(path_tickDataUSD15sec)
    # rtpath = Path("$RT") # did not work...
    while True:
        try:
            Ymd_str = file_date.strftime("%Y%m%d")
            m_str = file_date.strftime("%m")
            d_str = file_date.strftime("%d")
            file = Path.joinpath(rtpath,
                    f"{metric}/{file_date.year}/{m_str}/{d_str}/{metric}_{Ymd_str}.dat.gz")
            log.info(f"Reading {file}")
            # df = pd.read_csv(file, compression='gzip', sep=',')
            df = pd.read_csv(file, compression='gzip', header=7, sep=',', quotechar='"')
            df.rename(columns={'#<Column Name>"ITGid"': 'ITGid'}, inplace=True)
            df['Ticker'] = df['Ticker'].astype(str)
        except IOError:
            if days_back > 10:
                log.fatal("Can't find file for the previous 10 days")
                raise
            else:
                file_date = file_date - datetime.timedelta(days=1)
                days_back += 1
                log.warn(f"Can't find file for {file_date}. Looking back 1 day")
                continue
        except:
            log.fatal('Unexpected error')
            raise

        # print('Loaded vola weights:',file)
        break

    return df

def ji_read_DailyDOB_5min(iso, metric, date):
    """
    Read the monthly Intraday {metric} file for the specified date and ISO. If the file doesn't exists, look back 12 months.
    date: should be datetime type
    mydatetime = datetime.datetime(2017,1,3)

    """
    file_date = date
    days_back = 0
    rtpath = Path("/export/prod/data1")
    # rtpath = Path("$RT") # did not work...
    while True:
        try:
            Ymd_str = file_date.strftime("%Y%m%d")
            m_str = file_date.strftime("%m")
            d_str = file_date.strftime("%d")
            file = Path.joinpath(rtpath,
                                 f"FETAQ/tickData/{iso}/DailyDOB/{metric}/{file_date.year}/{m_str}/{d_str}/{metric}_{Ymd_str}.dat.gz")
            log.info(f"Reading {file}")
            # df = pd.read_csv(file, compression='gzip', sep=',')
            df = pd.read_csv(file, compression='gzip', header=7, sep=',', quotechar='"')
            df.rename(columns={'#<Column Name>"ITGid"': 'ITGid'}, inplace=True)
            df['Ticker'] = df['Ticker'].astype(str)
        except IOError:
            if days_back > 10:
                log.fatal("Can't find file for the previous 10 days")
                raise
            else:
                file_date = file_date - datetime.timedelta(days=1)
                days_back += 1
                log.warn(f"Can't find file for {file_date}. Looking back 1 day")
                continue
        except:
            log.fatal('Unexpected error')
            raise

        # print('Loaded vola weights:',file)
        break

    return df

def ji_read_intraday_MarketConditions_5min(iso,mytype,metric, date):
    """
    Read the intraday {metric} file (from Kris) for the specified date and ISO. If the file doesn't exists, look back 12 months.
    Types:
        HistMC = Historical Market Conditions
        SurMC = Unnormalized Surprises
        NormMC = Normalized Surprises
    Metric:
        Uncond30MinSpread
        Uncond30MinVolatility
        Uncond30MinVolume
        Uncond30MinVolumeBF
        UncondCumSpread
        UncondCumVolatility
        UncondCumVolume
        UncondCumVolumeBF
        UncondResidSpread
        UncondResidVolatility
        UncondResidVolume
        UncondResidVolumeBF
    """
    file_date = date
    days_back = 0
    #rtpath = Path("/export/prod/data1")
    rtpath = Path("/local/data/itg/fe/prod/data1")
    # rtpath = Path("$RT") # did not work...
    while True:
        try:
            Ymd_str = file_date.strftime("%Y%m%d")
            Y_str = file_date.strftime("%Y")
            m_str = file_date.strftime("%m")
            d_str = file_date.strftime("%d")
            file = Path.joinpath(rtpath,
                f"HistoricalFiles/MarketConditions/2.0/{iso}/{mytype}/{metric}/{file_date.year}/{metric}_{iso}_2.0_{Ymd_str}.dat.gz")
            log.info(f"Reading {file}")
            # df = pd.read_csv(file, compression='gzip', sep=',')
            df = pd.read_csv(file, compression='gzip', header=8, sep=',', quotechar='"')
            df.rename(columns={'#<Column Name>Ticker': 'ticker'}, inplace=True)
            #df['Ticker'] = df['Ticker'].astype(str)
        except IOError:
            if days_back > 10:
                log.fatal("Can't find file for the previous 10 days")
                raise
            else:
                file_date = file_date - datetime.timedelta(days=1)
                days_back += 1
                log.warn(f"Can't find file for {file_date}. Looking back 1 day")
                continue
        except:
            log.fatal('Unexpected error')
            raise

        # print('Loaded vola weights:',file)
        break

    return df

def ji_read_FX_ECN_trades(ccy,date):
    """
    Read the quotes from ECN (3 vendors) give the date
    """
    file_date = date
    days_back = 0
    #rtpath = Path("/export/prod/data1/FETAQ/FX/Trade")
    # rtpath = Path("$RT") # did not work...
    rtpath = "/export/prod/data1/FETAQ/FX/Trade"

    #while True:
    try:
        Ymd_str = file_date.strftime("%Y%m%d")
        Y_str = file_date.strftime("%Y")
        m_str = file_date.strftime("%m")
        d_str = file_date.strftime("%d")
        #file = Path.joinpath(rtpath,f"/{file_date.year}/{ccy}_{Ymd_str}_ALL.dat.gz")
        #file = f"/{file_date.year}/{ccy}_{Ymd_str}_ALL.dat.gz"
        file = os.path.join(rtpath,Y_str,m_str,d_str,'Trade_'+ccy+'_'+Ymd_str+'_ALL.dat.gz')
        #print('Loading file:',file)
        log.info(f"Reading {file}")
        #df = pd.read_csv(file, compression='gzip', sep=',')
        #df = pd.read_csv(file, compression='gzip', header=6, sep=',', quotechar='"')
        df = pd.read_csv(file, compression='gzip', header=6,sep=",", quoting=csv.QUOTE_NONE) # Solution was to add csv.QUOTE_NONE
        df.rename(columns={'#<Column Name>id': 'id'}, inplace=True)
        #df['Ticker'] = df['Ticker'].astype(str)
    except:
        file_date = file_date
        log.warn(f"Can't find file for {file_date}. Skipping...")
        return pd.DataFrame()
        #continue

        """
        except IOError:
            if days_back > 10:
                log.fatal("Can't find file for the previous 10 days")
                raise
            else:
                file_date = file_date - datetime.timedelta(days=1)
                days_back += 1
                log.warn(f"Can't find file for {file_date}. Looking back 1 day")
                continue
        except:
            log.fatal('Unexpected error')
            raise
        """
        # print('Loaded vola weights:',file)
        #break
    return df

def ji_read_FX_ECN_quotes(ccy,date):
    """
    Read the quotes from ECN (3 vendors) give the date
    """
    file_date = date
    days_back = 0
    #rtpath = Path("/export/prod/data1/FETAQ/FX/Trade")
    # rtpath = Path("$RT") # did not work...
    rtpath = "/export/prod/data1/FETAQ/FX/Quote"

    while True:
        try:
            Ymd_str = file_date.strftime("%Y%m%d")
            Y_str = file_date.strftime("%Y")
            m_str = file_date.strftime("%m")
            d_str = file_date.strftime("%d")
            #file = Path.joinpath(rtpath,f"/{file_date.year}/{ccy}_{Ymd_str}_ALL.dat.gz")
            #file = f"/{file_date.year}/{ccy}_{Ymd_str}_ALL.dat.gz"
            file = os.path.join(rtpath,Y_str,m_str,d_str,'Quote_'+ccy+'_'+Ymd_str+'_ALL.dat.gz')
            print('Loading file:',file)
            log.info(f"Reading {file}")
            #df = pd.read_csv(file, compression='gzip', sep=',')
            #df = pd.read_csv(file, compression='gzip', header=6, sep=',', quotechar='"')
            df = pd.read_csv(file, compression='gzip', header=6,sep=",", quoting=csv.QUOTE_NONE) # Solution was to add csv.QUOTE_NONE
            df.rename(columns={'#<Column Name>Timestamp': 'stamp'}, inplace=True)
            # Remove the milisec field in stamp
            df.stamp = df.stamp.apply(lambda s: s[:8])
            #df['Ticker'] = df['Ticker'].astype(str)
        except:
            file_date = file_date - datetime.timedelta(days=1)
            log.warn(f"Can't find file for {file_date}. Skipping...")
            raise
        """
        except IOError:
            if days_back > 10:
                log.fatal("Can't find file for the previous 10 days")
                raise
            else:
                file_date = file_date - datetime.timedelta(days=1)
                days_back += 1
                log.warn(f"Can't find file for {file_date}. Looking back 1 day")
                continue
        except:
            log.fatal('Unexpected error')
            raise

        """
        break

    return df
# Wrapper to importAll (P5)
from ji_abd_importAll import*


def main():
    """In this script, the following steps are executed:
    - Load and convert the previously computed df2plot (step 1), from 5min to 30min bins.
    More than a mere conversion, stats are re-computed in 30min bins.
    - Final df2plot are saved by strategy and HLG."""

    ## Hardcoded parameters (TODO: parser)
    df2plotpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/tmp'

    isTest = True
    if isTest:
        savelabel = 'HLGlarger_test'
        allrange = [[20180201, 20180229], [20180301, 20180331]]
        allHLG = [0,1]
        myStrategies = ['SCHEDULED']
        shortNames = ['SCHE']
        weeks = [1, 2, 3, 4]
        # df2plot = itg.p5_helper_load_df2plot_April9(df2plotpath,cols2include,allrange,allHLG,myStrategies)
        # df2plot = jifin.p5_helper_load_compute_df2plot30min_ABD2018(df2plotpath, allrange, allHLG, myStrategies)
    else:
        savelabel = 'HLGlarger_2017_2018'
        ## Jan-Dec
        allrange = [[20180101, 20180131], [20180201, 20180229], [20180301, 20180331], [20180401, 20180431],
                    [20180501, 20180531], [20180601, 20180631],
                    [20180701, 20180731], [20180801, 20180831], [20180901, 20180931], [20181001, 20181031],
                    [20181101, 20181131], [20181201, 20181231]]
        allHLG = [0, 1, 2, 3]
        myStrategies = ['SCHEDULED', 'DARK', 'IS', 'OPPORTUNISTIC']
        shortNames = ['SCHE', 'DARK', 'IS', 'LS']
        weeks = [1, 2, 3, 4]
        # df2plot = jifin.p5_helper_load_compute_df2plot30min_ABD2018(df2plotpath, allrange, allHLG, myStrategies)
    #    df2plot = itg.p5_helper_load_df2plot_April9(df2plotpath,cols2include,allrange,allHLG,myStrategies)


    #### PART A: Check if was computed
    print('\n======================================================================================')
    print('=======    PART 1: Check if all the supposed df2plot were computed          ==========')
    print('======================================================================================')

    complete = True
    for t0t1 in allrange:
        t0 = t0t1[0]
        t1 = t0t1[1]
        print('\n=== Range: %d-%d ===' % (t0, t1))
        for HLG in allHLG:
            print(
                '\n--- HLG=%d --------------------------------------------------------------------------------------' % HLG)
            slabel = '%d_%d_HLG%d' % (t0, t1, HLG)
            for strategy, shortName in zip(myStrategies, shortNames):
                print('%s: \t' % shortName, end='')
                for week in weeks:
                    fname = os.path.join(df2plotpath,'df2plot_ABD_HLGlarger_%s_%s_week%d.csv.gz' % (slabel, strategy, week))
                    #print('fname:',fname)
                    if os.path.isfile(fname):
                        print('%d-' % week, end='')
                    else:
                        complete = False
                        print('  ', end='')
                print('\t', end='')
            print('')

    if complete:
        print('\n- Ready to continue! All df2plot were computed successfully.')
    else:
        print('\n(*) Warning! Not all df2plot were computed. So better to wait to concatenate...')

    #### PART B: Compute the 30min bin version
    print('\n======================================================================================')
    print('=======    PART 2: Compute the 30min bin df2plot                            ==========')
    print('======================================================================================')
    print('- Obs: the weekly, not the merged, file has the information necessary to compute in 30min bins.')
    cols2include = ['token', 'tradeDate', 'arriveBin', 'relSize', 'HLG', 'algoType', 'SpreadCapture',
                    'binDuration', 'bin', 'ExPriceForBin', 'FillNbrForBin',
                    'midQuote', 'LowPrice', 'HighPrice', 'histVola5min', 'empCost', 'prate',
                    'normStyle', 'signedReturn',
                    'realVola', 'surVola',
                    'adjHistVola5min', 'adjHistVolu5min', 'adjRealVola', 'adjSizeShare', 'sizeShare',
                    'firstTrade', 'gapDuration', 'very1stTrade', 'adjSignedReturn', 'adjSurVola',
                    'adjSurVolume', 'normAdjSignedReturn', 'normSignedReturn',
                    'normEmpCost', 'pred5min', 'pred30min', 'adjPred5min', 'adjPred30min']

    df2plot = jifin.p5_helper_load_compute_df2plot30min_ABD2018(df2plotpath, allrange, allHLG, myStrategies)
    itg.ji_p5_check_df2plot(df2plot)

    for HLG in allHLG:
        print('\n----------------------------- Processing HLG=%d ----------------------------------' % HLG)
        for strategy in myStrategies:
            print('----------------- Strategy: %s ------------------' % strategy)
            sperc = 0.7  # NOTE: assume that all strategies were already filtered at perc>0.7
            ## 1) Retrieve the strategy to be saved
            df2plot = itg.p5_helper_retrieveStrategy(df2plot, strategy, sperc)
            imask = (df2plot.HLG == HLG)
            df2plot = df2plot.loc[imask, cols2include]
            print('- Total loaded: %d trades' % df2plot.shape[0])

            nn = 'df2plot30min_ABD_%s_%s_HLG%d.csv.gz' % (savelabel, strategy, HLG)
            fname = os.path.join(df2plotpath, nn)
            print('- Saving %s...' % fname)
            df2plot.to_csv(fname, index=False, compression='gzip')

    print('------------------ Done saving df2plot by HLG & Strategy --------------------------')


if __name__ == "__main__":
    main()
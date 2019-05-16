# Wrapper to importAll (P5)
from ji_abd_importAll import*


def main():
    """In this script, the following steps are executed:
    - Load and concatenate all the previously computed df2plot (step 1) for quicker analyses
    - Use step 3 to generate the plots and save tables"""

    ## Hardcoded parameters (TODO: parser)
    istest = True
    if istest:
        savelabel = 'HLGlarger_test'
        #allyy = [2018]
        #allqq = ['Q1']
        #allrange = [[20180301, 20180331]] # for simplicity ignores 30 or 31
        allrange = [[20180201, 20180229],[20180301, 20180331]]  # for simplicity ignores 30 or 31
        allHLG = [0,1]
        myStrategies = ['SCHEDULED']
        shortNames = ['SCHE']
        allperc = [80] # percent to filter for each strategy type
        weeks = [1,2,3,4]
        ## SET DATE RANGES, used to generate input plans for DyCE
        myweeks = [[1, 7],[8,15],[16,23],[24,31]]

        #weeks = [1, 2, 3, 4]
        ## SET DATE RANGES, used to generate input plans for DyCE
        #myweeks = [[1, 7], [8, 15], [16, 23], [24, 31]]
    else:
        savelabel = 'HLGlarger_2017_2018'
        #allyy = [2017,2018] # year to load from ABD
        #allqq = ['Q1', 'Q2', 'Q3', 'Q4'] # clusters to load from ABD
        allrange = [[20180101, 20180131], [20180201, 20180229], [20180301, 20180331], [20180401, 20180431],
                    [20180501, 20180531], [20180601, 20180631],
                    [20180701, 20180731], [20180801, 20180831], [20180901, 20180931], [20181001, 20181031],
                    [20181101, 20181131], [20181201, 20181231]]
        allHLG = [0, 1, 2, 3]
        myStrategies = ['SCHEDULED', 'DARK', 'IS', 'OPPORTUNISTIC']
        shortNames = ['SCHE', 'DARK', 'IS', 'LS']
        allperc = [80,71,80,80]  # percent to filter for each strategy type
        weeks = [1, 2, 3, 4]
        ## SET DATE RANGES, used to generate input plans for DyCE
        myweeks = [[1, 7], [8, 15], [16, 23], [24, 31]]
        # myweeks = [[1, 7]]

    ## SET PATHS
    #dpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/tmp/'  # Where df2go is saved to , loaded from
    df2plotpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/tmp/'  # Where df2plot will be saved

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
                    fname = os.path.join(df2plotpath,
                                         'df2plot_ABD_HLGlarger_%s_%s_week%d.csv.gz' % (slabel, strategy, week))
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

    runPart2 = True
    if runPart2:
        #### PART B: merge df2plot and save by HLG and strategy groups####
        print('\n========================================================================================')
        print('===  PART 2: Concatenate the weekly generated df2plot, save by {HLG,strategy} groups ===')
        print('========================================================================================')

        cols2include = ['token', 'tradeDate', 'arriveBin', 'relSize', 'HLG', 'algoType', 'SpreadCapture',
                        'binDuration', 'bin', 'ExPriceForBin', 'FillNbrForBin',
                        'midQuote', 'LowPrice', 'HighPrice', 'histVola5min', 'empCost', 'prate',
                        'normStyle', 'signedReturn',
                        'realVola', 'surVola',
                        'adjHistVola5min', 'adjHistVolu5min', 'adjRealVola', 'adjSizeShare', 'sizeShare',
                        'firstTrade', 'gapDuration', 'very1stTrade', 'adjSignedReturn', 'adjSurVola',
                        'adjSurVolume', 'normAdjSignedReturn', 'normSignedReturn',
                        'normEmpCost', 'pred5min', 'pred30min', 'adjPred5min', 'adjPred30min']

        df2plot = itg.p5_helper_load_df2plot_April9(df2plotpath, cols2include, allrange, allHLG, myStrategies)

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
                # Consistency check
                itg.ji_p5_check_df2plot(df2plot)

                nn = 'df2plot_ABD_%s_%s_HLG%d.csv.gz' %(savelabel,strategy,HLG)
                fname = os.path.join(df2plotpath, nn)
                print('- Saving %s...' % fname)
                df2plot.to_csv(fname, index=False, compression='gzip')

        print('------------------ Done saving df2plot by HLG & Strategy --------------------------')

    print('\n========================================  END  ======================================= \n')

            # DEBUGGING
            #df2go.to_csv('df2go_62cols_ABD_HLGlarger_%s.csv' % (slabel))

if __name__ == "__main__":
    main()
# Wrapper to importAll (P5)
from ji_abd_importAll import*


def main():
    """
    Generate plots and tables from df2plot (30min bins)
    """

    ## Hardcoded parameters (TODO: parser)
    # UPDATE as needed
    quantilesx = [0.2, 0.4, 0.6, 0.8, 0.9] # size quantiles for the aggregation
    sizeVarx = 'adjSizeShare'
    sizeLabelx = 'adjSizeShare'
    condVarx = 'adjPred30min'
    condLabelx = 'adjPredVolaSur30min'

    istest = True
    if istest:
        df2plotlabel = 'HLGlarger_test' # label used to save df2plot
        allHLG = [[0],[1]]
        allStrat = ['SCHEDULED']
        allS = ['SCHE'] # short name

    else:
        df2plotlabel = 'HLGlarger_2017_2018'
        #allHLG = [[0], [1], [2],[3]]
        allHLG = [[0], [1], [2,3]]
        myStrategies = ['SCHEDULED', 'DARK', 'IS', 'OPPORTUNISTIC']
        shortNames = ['SCHE', 'DARK', 'IS', 'LS']

    ## SET PATHS
    dpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/tmp/'  # Where df2go is saved to , loaded from
    # path to save plots and tables
    savepath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/tmp'

    #### PART A: Check if was computed
    print('\n======================================================================================')
    print('=======    Generating {Cost,Return} x size plots and tables  (30min bins)   ==========')
    print('======================================================================================')
    print('- Note: You might have to setup your $DISPLAY variable to generate the plots.')

    for strat, ss in zip(allStrat, allS):
        for cHLG in allHLG:
            ## 0) Load
            df2plot = pd.DataFrame()
            auxlabel = '%s:HLG=' % (ss)
            savelabel = '30min_%s_HLG' % (ss)

            ## Retrieve and concatenate
            for HLG in cHLG:
                df2plot = df2plot.append(
                    pd.read_csv(os.path.join(dpath, 'df2plot30min_ABD_%s_%s_HLG%d.csv.gz' %(df2plotlabel,strat, HLG))))
                if HLG == cHLG[-1]:  # last
                    auxlabel += '%d' % HLG
                else:
                    auxlabel += '%d,' % HLG
                savelabel += '%d' % HLG

            ## Add corrected prediction
            df2plot = itg.ji_p5_add_corrected_prediction_1stTrade_30min(df2plot)

            ## Remove empCost outliers... observed between 20181026 and 20181031
            nn = 5
            imask = (df2plot.normEmpCost.abs() < nn)
            print('- Removing %d (%2.4f of total) bins out of bound, i.e., Abs>%d)...' % (
            np.sum(df2plot.normEmpCost.abs() > nn), np.sum(df2plot.normEmpCost.abs() > nn) / df2plot.shape[0], nn))
            df2plot = df2plot[imask]

            ## Prepare plot
            try:
                ## Basic bin stats

                itg.p5_helper_print_binStats(df2plot, 'HLG=%d:%s' % (HLG, ss))
                print('2017Q1Q2:', np.sum(df2plot.tradeDate.between(20170101, 20170631)))
                print('2017Q3Q4:', np.sum(df2plot.tradeDate.between(20170701, 20171201)))
                print('2018Q1Q2:', np.sum(df2plot.tradeDate.between(20180101, 20180631)))
                print('2018Q3Q4:', np.sum(df2plot.tradeDate.between(20180701, 20181201)))
                ## Plot
                set2run = [cHLG]  # Pass a single instance because it was already iterated before to concatenate data
                alllabels = [auxlabel]
                allsavelabel = [savelabel]
                #itg.p5_wrapper_generate_plots_April3(df2plot, set2run, alllabels, quantilesx, sizeVarx, sizeLabelx,
                itg.p5_wrapper_generate_plots_30min_April16(df2plot, set2run, alllabels, quantilesx, sizeVarx, sizeLabelx,
                                                     condVarx, condLabelx, savepath, allsavelabel)
            except:
                print('** WARNING! ** Problem with HLG=%d:%s. Not sufficient #bins? Check...' % (HLG, ss))

    print('\n========================================  END  ======================================= \n')

if __name__ == "__main__":
    main()
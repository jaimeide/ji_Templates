# Wrapper to importAll (P5)
from ji_abd_importAll import*

USAGE = """
In this script, the following steps are executed:
- Load ABD clusters
- Compute df2go: with all the stats, cost, return, etc
- Compute df2plot: it includes the vola predictions from DyCE output as well as auxiliary variables for analyses such as indicators for 1stTraded and market conditions.
- Use step 2 to merge the df2plot for quicker analyses
- Use step 3 to generate the plots and save tables
"""

def main():
    """In this script, the following steps are executed:
    - Load ABD clusters
    - Compute df2go: with all the stats, cost, return, etc
    - Compute df2plot: it includes the vola predictions from DyCE output as well as auxiliary variables for analyses such as indicators for 1stTraded and market conditions.
    - Use step 2 to merge the df2plot for quicker analyses
    - Use step 3 to generate the plots and save tables"""

    parser = argparse.ArgumentParser(description='ABD Analyzer',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=USAGE)
    # required_args = parser.add_argument_group("required arguments")
    # required_args.add_argument("-j", "--json", dest="config", required=True,
    #                            help="json file with all config", metavar="config", type=str)
    parser.add_argument("-j","--json", dest = "jsonfile", default = "config_step1_2017_2018_full.json", type = str,
                                help = "json file")

    arg = parser.parse_args()
    #inputFile = "/local/data/itg/fe/prod/data1/mborkove-local/DyCE/FX/clusterdata_byClientBySide_Feb11.csv"
    jsonfile = arg.jsonfile
    if os.path.isfile(jsonfile):
        # Read json
        with open(jsonfile, "r") as rfile:
            config = json.load(rfile)
        print('- Loaded json file from:', jsonfile)
        # Convert dic to variables
        ns = Namespace(**config)
        allyy, allqq, allrange, allHLG, myStrategies, shortNames, allperc, weeks, myweeks, runPart1, runPart2 = ns.allyy, ns.allqq, ns.allrange, ns.allHLG, ns.myStrategies, ns.shortNames, ns.allperc, ns.weeks, ns.myweeks, ns.runPart1, ns.runPart2
        ## SET PATHS
        RTpath = ns.RTpath
        dpath = ns.dpath
        planpath = ns.planpath
        df2plotpath = ns.df2plotpath
        # newfe-s18: DyCE location and working directory
        DyCEpath = ns.DyCEpath
        ppath = ns.ppath
        # for key, val in config.items(): # (NOT WORKING...)
        #    exec(key + '=val')
        #locals().update(config)
    else:
        print('(*) Warning! json file %s not found. Run default 2017+2018 experiment...'%jsonfile)

        allyy = [2017,2018] # year to load from ABD
        allqq = ['Q1', 'Q2', 'Q3', 'Q4'] # clusters to load from ABD
        allrange = [[20180101, 20180131], [20180201, 20180229], [20180301, 20180331], [20180401, 20180431],
                    [20180501, 20180531], [20180601, 20180631],
                    [20180701, 20180731], [20180801, 20180831], [20180901, 20180931], [20181001, 20181031],
                    [20181101, 20181131], [20181201, 20181231]]
        allHLG = [0, 1, 2, 3]
        myStrategies = ['SCHEDULED', 'DARK', 'IS', 'OPPORTUNISTIC']
        shortNames = ['SCHE', 'DARK', 'IS', 'LS']
        allperc = [80,71,80,80]  # percent to filter for each strategy type
        ## SET DATE RANGES, used to generate input plans for DyCE
        weeks = [1, 2, 3, 4]
        myweeks = [[1, 7], [8, 15], [16, 23], [24, 31]]
        # myweeks = [[1, 7]]

        ## SET PATHS
        #RTpath = Path('/local/data/itg/fe/prod/data1')  # $RT (is there a better way to pass this?)
        RTpath = '/local/data/itg/fe/prod/data1'  # $RT (is there a better way to pass this?)
        dpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/tmp/'  # Where df2go is saved to , loaded from
        planpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/tmp/'  # Where input plans will be saved
        df2plotpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/tmp/'  # Where df2plot will be saved
        # newfe-s18: DyCE location and working directory
        DyCEpath = "/local/data/itg/fe/prod/data1/jide-local/DyCE/ref/DyCE/bin/DyCE"
        ppath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/tmp/' # project path where DyCE related output is saved in directory Out
        # What to run
        runPart1 = True
        runPart2 = True

    #### PART A: GENERATE df2go ####

    #runPart1 = True

    if runPart1:
        print('\n===================================================================================')
        print('=======    PART 1: Load ABD clusters and Save df2go                      ==========')
        print('===================================================================================')

        ## Step 1: Load cluster data into dataFrame
        # Set paths
        #abdpath = Path.joinpath(RTpath, 'DTCM_Shared', 'ClusterData', 'ABD', 'USA', 'ALL')
        abdpath = os.path.join(RTpath, 'DTCM_Shared', 'ClusterData', 'ABD', 'USA', 'ALL')
        print('- Loading ABD cluster data from:',abdpath)
        iso = 'USA'
        colsToRead = [1, 2, 4, 5, 7, 8, 14, 15, 17, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 41,
                      45, 48, 49, 54, 55, 57, 65]  # algoType[49] MUST be included
        dd = {"nrows": None}
        # Load DF
        df_ABD = itg.ji_load_ABD_clusters_all(abdpath, iso, allyy, allqq, colsToRead, dd)

        ## Step 2: compute and save df2go
        dfall = df_ABD
        for t0t1 in allrange:
            for HLG in allHLG:
                t0 = t0t1[0]
                t1 = t0t1[1]
                print('\n====== Computing df2go for HLG=%d, range: %d to %d ======\n' % (HLG, t0, t1))
                imask = (dfall['tradeDate[17]'].between(t0, t1)) & (dfall['HLG[31]'] == HLG)
                slabel = '%d_%d_HLG%d' % (t0, t1, HLG)
                print('np.sum(imask):',np.sum(imask))
                if np.sum(imask)>0:
                    dfallx = dfall[imask].copy()
                    print('dfall.shape:', dfallx.shape)
                    # mytype = ['CommonStock']
                    mytype = ['Equity', 'REIT', 'ADR', 'Other', 'Fund', 'GDR', 'Unit']  # Removed: 'ETF'
                    sizemask0 = (dfallx['HLG[31]'] == 0) & (dfallx['secType[14]'].isin(mytype))
                    sizemask1 = (dfallx['HLG[31]'] == 1) & (dfallx['relSize[28]'] > 1) & (dfallx['secType[14]'].isin(mytype)) # larger ones
                    sizemask2 = (dfallx['HLG[31]'] == 2) & (dfallx['relSize[28]'] > 0.5) & (dfallx['secType[14]'].isin(mytype)) # larger ones
                    sizemask3 = (dfallx['HLG[31]'] == 3) & (dfallx['relSize[28]'] > 0.25) & (dfallx['secType[14]'].isin(mytype)) # larger ones
                    imask = ((sizemask0) | (sizemask1) | (sizemask2) | (sizemask3))

                    print('Selected clusters: %d (%d%%)' % (np.sum(imask), 100 * np.sum(imask) / dfallx.shape[0]))
                    imask = ((sizemask0) | (sizemask1) | (sizemask2) | (sizemask3))
                    print('Selected clusters: %d \n(%d%% of total number of parent orders)' % (np.sum(imask), 100 * np.sum(imask) / dfallx.shape[0]))
                    print('sizemask HLG=0: %d (%d%%)' % (np.sum(sizemask0), 100 * np.sum(sizemask0) / dfallx.shape[0]))
                    print('sizemask HLG=1: %d (%d%%)' % (np.sum(sizemask1), 100 * np.sum(sizemask1) / dfallx.shape[0]))
                    print('sizemask HLG=2: %d (%d%%)' % (np.sum(sizemask2), 100 * np.sum(sizemask2) / dfallx.shape[0]))
                    print('sizemask HLG=3: %d (%d%%)' % (np.sum(sizemask3), 100 * np.sum(sizemask3) / dfallx.shape[0]))

                    dfalgo = dfallx[imask].reset_index(drop=True)
                    # REDcluster
                    # df2go = itg.ji_wrapper_create_compute_allFeatures(dfalgo,'RED')
                    # ABD
                    # df2go = itg.ji_wrapper_create_compute_allFeatures(dfalgo,'ABD')
                    ## MAIN COMPUTATION
                    df2go = itg.ji_wrapper_create_compute_allFeatures_COMPACT(dfalgo, 'ABD')
                    print('- Final df2go shape:', df2go.shape)
                    # SAVE
                    fname = os.path.join(dpath, 'df2go_62cols_ABD_HLGlarger_%s.csv' % (slabel))
                    print('- Saving and zipping: %s' % (fname))
                    df2go.to_csv(fname)  # (~??GB size...)
                    subprocess.call('gzip -f {}'.format(fname), shell=True)
                else:
                    print('(*) Warning! There is no trade for %s. Skip computation...'%(slabel))
    else:
        print('(Skipping Part 1. Already computed?...)')

    #runPart2 = True
    if runPart2:
        #### PART B: Load saved df2go, create input plans, run DyCE and save df2plot ####
        print('\n====================================================================================')
        print('======= PART 2: Load saved df2go, create input plans, run DyCE and save df2plot ========')
        print('========================================================================================')
        if not os.path.exists(os.path.join(ppath,'Out')):
            os.mkdir(os.path.join(ppath, 'Out'))
            print('- Creating directory:', os.path.join(ppath, 'Out'))

        ######## 1) Get df2go
        for t0t1 in allrange:
            for HLG in allHLG:
                t0 = t0t1[0]
                t1 = t0t1[1]
                slabel = '%d_%d_HLG%d' % (t0, t1, HLG)
                fname = os.path.join(dpath, 'df2go_62cols_ABD_HLGlarger_%s.csv.gz' % (slabel))

                if os.path.isfile(fname):
                    print('- Loading: %s...' % (fname))
                    df2go = pd.read_csv(fname)  #
                    for strategy, sperc in zip(myStrategies, allperc):
                        print('==================================================================================')
                        print('=== Range: %d-%d HLG=%d Strategy:%s ===' % (t0, t1, HLG, strategy))
                        print('==================================================================================')

                        ## a) Filter by Strategy
                        # sperc = 80 # 50% percentile
                        df2gox = itg.p5_helper_retrieveStrategy(df2go.copy(), strategy, sperc)

                        ## b) Create input plan for engine
                        print(
                            '(*) OBS1: Use the "bin" time as the starttime, not the parent-order start time (to simplify)')
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
                            planfile = os.path.join(planpath,'input_ABD_HLGlarger_%s_%s_week%d.plan' % (datalabel, strategy, i + 1))
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
                                    itg.ji_wrapper_save_df2plot_volaPrediction_fromDf2go(df2gox.copy(), outpath, df2plotpath,mylabel)
                                except:
                                    print('** Warning ** Failed to save df2plot.. Check DyCE output...')
                            except:
                                print('** Warning ** Failed to run DyCE:', cmd)
                                print('**  - df2plot not saved for:', mylabel)
                                print('**  - Check why...')

                    print('=========================== Done generating df2plot=========================')
                else:
                    print('(*) Warning! File %s does not exist. Check if it was not computed or if there is no trade in this range and HLG...'%fname)
                    print('    Suggested solution: include all the strategies or have SCHEDULED when running this script.')

    print('\n===================================== END ===================================== \n')

            # DEBUGGING
            #df2go.to_csv('df2go_62cols_ABD_HLGlarger_%s.csv' % (slabel))

if __name__ == "__main__":
    main()
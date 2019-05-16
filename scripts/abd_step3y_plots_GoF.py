# Wrapper to importAll (P5)
from ji_abd_importAll import*

USAGE = """
        Given the df2plot file (5min version), it generates Goodness to Fit plots with the following parameters:
    - binstart
    - binend
    - HLG
    - tgroup: set of tickers within that HLG
        """

def main():

    parser = argparse.ArgumentParser(description='ABD Analyzer',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=USAGE)
    # required_args = parser.add_argument_group("required arguments")
    # required_args.add_argument("-j", "--json", dest="config", required=True,
    #                            help="json file with all config", metavar="config", type=str)
    parser.add_argument("-j","--json", dest = "jsonfile", default = "config_step3y_SCH_HLG2.json", type = str,
                                help = "json file")

    arg = parser.parse_args()
    jsonfile = arg.jsonfile
    if os.path.isfile(jsonfile):
        # Read json
        with open(jsonfile, "r") as rfile:
            config = json.load(rfile)
        print('- Loaded json file from:', jsonfile)
        # Convert dic to variables
        ns = Namespace(**config)
        allHLG ,myStrategies,shortNames,tickers,df2plotpath,savepath = ns.allHLG ,ns.myStrategies,ns.shortNames,ns.tickers,ns.df2plotpath,ns.savepath
        allbins,binlabels,quantiles = ns.allbins,ns.binlabels,ns.quantiles

    else:
        print('(*) Warning! json file %s not found. Run default values...'%jsonfile)

        allHLG = [2]
        myStrategies = ['SCHEDULED']
        shortNames = ['SCHE']

        #tickers = ['VC','AAN']
        tickers = None

        ## SET PATHS
        df2plotpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data_df2plot'  # Where df2plot will be saved
        savepath = './tmp'

        ## Other settings (UPDATE as necessary)
        allbins = [[1, 78], [7, 18], [19, 36], [37, 60], [61, 75]]
        binlabels = ['all day', '9:30-11:00', '11:00-12:30', '12:30-14:30', '14:30-15:45']
        ## 25 points with more for small values
        quantiles = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                     0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975]

    ## Load df2plot
    for HLG in allHLG:
        for strategy,sname in zip(myStrategies,shortNames):
            dffile = os.path.join(df2plotpath, f'df2plot_ABD_HLGlarger_2017_2018_{strategy}_HLG{HLG}.csv.gz')
            mainlabel = f'{sname}:HLG{HLG}'
            print('- Loading file %s...' % dffile)
            df2plot = pd.read_csv(dffile)
            imask = [False] * df2plot.shape[0]
            extralabel = ''
            savelabel = dffile.split("/df2plot_")[1].split(".csv")[0]
            if tickers is not None:
                for yourticker in tickers:
                    imask = imask | df2plot.token.str.contains('_' + yourticker + '_')
                    extralabel = extralabel + '_%s'%yourticker
                df2plot = df2plot[imask]
                pdffile = os.path.join(savepath, 'plots_GoF_v2_%s%s.pdf' % (savelabel, extralabel))
            else:
                pdffile = os.path.join(savepath, 'plots_GoF_v2_%s%s.pdf' % (savelabel, extralabel))

            ## GoF plots for different times of the day
            pdfobj = PdfPages(pdffile)
            print('- Creating file:', pdffile)
            for bb, blabel in zip(allbins, binlabels):
                binstart = bb[0]
                binend = bb[1]
                if tickers is not None:
                    mylabel = mainlabel + ' (%s, ticker: %s)' % (blabel, extralabel)
                else:
                    mylabel = mainlabel + ' (%s)' % (blabel)

                sizefloor = 0
                sizecap = 1000000
                tablelabel = '%s%s_bin%d_%d' % (savelabel, extralabel, binstart, binend)
                jifin.plot_GoF_fullVersion(df2plot, mylabel, savepath, tablelabel, binstart, binend, sizefloor, sizecap,
                                           pdfobj, quantiles)

    print('- Figure saved in', pdffile)
    pdfobj.close()

if __name__ == "__main__":
    main()
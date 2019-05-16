# Wrapper to importAll (P5)
from ji_abd_importAll import*

USAGE = """
        Given the strategy, HLG and ticker, it generates Goodness to Fit plots from the df2plot files (5min bins)
        for different times of the day.
        - Files will be saved with names based on the parsed input files.
        """

def main():

    parser = argparse.ArgumentParser(description='ABD Analysis',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=USAGE)
    # required_args = parser.add_argument_group("required arguments")
    parser.add_argument("-f","--dffile", dest = "dffile",
                        default = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data_df2plot/df2plot_ABD_HLGlarger_2017_2018_SCHEDULED_HLG2.csv.gz',
                        type = str, help = "df2plot file to load")
    parser.add_argument("-l", "--label", dest="mainlabel",
                        default='SCH:HLG2',
                        type=str, help="Label for the titles. Example SCH:HLG=2")
    parser.add_argument("-t", "--ticker", dest="yourticker",
                        default='VC',
                        type=str, help="Selected ticker. Example: VC")
    parser.add_argument("-o", "--output", dest="savepath",
                        default='./',
                        type=str, help="Output folder where to save plots.")
    arg = parser.parse_args()

    dffile = arg.dffile
    mainlabel = arg.mainlabel
    yourticker = arg.yourticker
    savepath = arg.savepath

    ## Other settings (UPDATE as necessary)
    allbins = [[1, 78], [7, 18], [19, 36], [37, 60], [61, 75]]
    binlabels = ['all day', '9:30-11:00', '11:00-12:30', '12:30-14:30', '14:30-15:45']

    ## Load df2plot
    print('- Loading file %s...'%dffile)
    df2plot = pd.read_csv(dffile)
    imask = df2plot.token.str.contains('_' + yourticker + '_')

    ## GoF plots for different times of the day
    savelabel = dffile.split("/df2plot_")[1].split(".csv")[0]
    pdffile = os.path.join(savepath, 'plots_GoF_%s_%s.pdf' % (savelabel,yourticker))
    pdfobj = PdfPages(pdffile)
    print('- Creating file:',pdffile)
    for bb,blabel in zip(allbins,binlabels):
        binstart = bb[0]
        binend = bb[1]
        mylabel = mainlabel+' (%s, ticker: %s)'%(blabel,yourticker)
        sizefloor = 0
        sizecap = 1000000
        realVar = 'surVola'
        realFloor = 0.0001
        realCap = 100
        jifin.plot_GoF(df2plot[imask],mylabel,binstart, binend, sizefloor, sizecap, realVar, realFloor, realCap, pdfobj)

    print('- Figure saved in', pdffile)
    pdfobj.close()

if __name__ == "__main__":
    main()
#!/usr/bin/python

##----------------------------------------------------------------------------##
## \file  distrFit.py                                                         ##
## \brief Fit a sample with a theoretical distribution                        ##
##                                                                            ##
## Fits a set of probabilistic distributions to a sample, optimizing their    ##
## parameters using maximum likelihood. The results are ranked using the      ##
## Kolmogorov-Smirnov statistic.                                              ##
##----------------------------------------------------------------------------##

from datetime          import datetime as dt
from Calendar.common   import dtFmtLog
from Utils.logUtils    import setupLog


import logging, os, sys

def makePaths(runDir, prefix, runTime, log):
    runDir = os.path.normpath(runDir)
    if not os.path.exists(runDir):
        os.makedirs(runDir)
    if not os.path.isdir(runDir):
        msg = runDir + ' is invalid - not a directory!'
        log.exception(msg)
        raise ValueError(msg)

    filePrefix = prefix + '.' + runTime.strftime(dtFmtLog)
    fileName   = os.path.join(runDir, filePrefix)
    return runDir, filePrefix, fileName

def parseCommandLine(argv, log, logToFile = True):
    import argparse

    parser = argparse.ArgumentParser(
                 description = 'Run sample distribution fitting')

    parser.add_argument("-c", "--csv", action = "store_true", default = False,
                        help = "produce csv output")
    parser.add_argument("-d", "--distrs", nargs='+', metavar="distr",
                        help = "list of distributions to run on, from " \
                               + ", ".join(DistributionFitter.GetDistrList()))
    parser.add_argument("-D", "--outDir", default='.',
                        help = "directory for output files")
    parser.add_argument("-G", "--graphAll", action = "store_true",
                        default = False,
                        help = "graph all distributions on the same plot")
    parser.add_argument("-g", "--graphEach", action = "store_true",
                        default = False,
                        help = "graph each distribution separately")
    parser.add_argument("-i", "--inFile", default='sample.txt',
                        help = "input text file with sample data [sample.txt]")
    parser.add_argument("-P", "--minProb", type = float,
                        help = "minimum probability of match in the K-S test,"
                               " as a decimal (use 0.5 for 1/2)")
    parser.add_argument("-p", "--prefix", default = 'distrFit',
                        help = "output file prefix, default distrFit")
    parser.add_argument("-S", "--maxStat", type = float,
                        help = "maximum threshold for the K-S statistic")
    parser.add_argument("-t", "--timeStamp",
                        help = "run as of a YYYYMMDDHHMMSS (default: now)")
    parser.add_argument("-x", "--excel", action = "store_true", default = False,
                        help = "produce Excel output")

    opts = parser.parse_args(argv[1:]) # ignore program name

    opts.runTime = dt.strptime(opts.timeStamp, dtFmtLog) if opts.timeStamp \
                   else dt.now()

    opts.outDir, opts.filePrefix, opts.fileName \
       = makePaths(runDir  = opts.outDir,
                   prefix  = opts.prefix,
                   runTime = opts.runTime,
                   log     = log)

    setupLog(log      = log,
             fileName = opts.fileName + '.log' if logToFile else None)

    if not opts.csv and not opts.excel:
        log.warn('Neither CSV -c nor Excel -x output specified, assuming CSV')
        opts.csv = True

    return opts

def processInputFile(inFileName):
    with open(inFileName, 'r') as inFile:
        sample = [float(x) for x in inFile]

    return sample

def main(argv = None):
    if argv is None:
        argv = sys.argv

    log  = logging.getLogger('distributionFitting')
    opts = parseCommandLine(argv = argv, log = log, logToFile = True)
    if opts is None:
        return 0

    try:
        progName = os.path.splitext( os.path.basename(argv[0]) )[0]
        log.info('Starting %s in %s',  progName, os.getcwd())
        log.info('Arguments %s', ' '.join(argv))

        sample = processInputFile(inFileName = opts.inFile)
        log.info('Read %d points from the input file %s',
                 len(sample), opts.inFile)

        log.info('Constructing the distribution fitter')
        fitter = DistributionFitter()

        log.info('Fitting distributions')
        out = fitter(sample     = sample, log = log, distrs  = opts.distrs,
                     graphEach  = opts.graphEach,    minProb = opts.minProb,
                     graphAll   = opts.graphAll,     maxStat = opts.maxStat,
                     filePrefix = os.path.join(opts.outDir, opts.filePrefix))

        if opts.csv:
            outFileName = os.path.join(opts.outDir, opts.filePrefix + '.csv')
            log.info('Writing CSV results to %s', outFileName)
            out.toCSV(outFileName = outFileName)

        if opts.excel:
            outFileName = os.path.join(opts.outDir, opts.filePrefix + '.xlsx')
            log.info('Writing Excel results to %s', outFileName)
            out.toExcel(outFileName = outFileName)

        log.info('Completed')

    except Exception as e:
        log.exception('Type %s; args %s', type(e), str(e))

    return 0

if __name__ == '__main__':
    sys.exit(main())

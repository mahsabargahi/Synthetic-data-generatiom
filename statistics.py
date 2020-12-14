## ----------------------------------------------------------------------------
# \file  statUtils.py
# \brief Statistical utilities
#
# $URL: https://trstwprsvnapv01.treasury.corp.ge.com/svn/Treasury/QUANTS%20GROUP/Branches/Dima/mdm0/Utils/Python/Maths/statistics.py $
# $Id: statistics.py 24330 2015-06-19 18:48:56Z 212361829 $
#  ------------------------------------------------------------------------------

from scipy.stats      import distributions, kstest
from sortedcontainers import SortedSet
from xlsxwriter       import Workbook

import csv, inspect, itertools, matplotlib.pyplot as plt, numpy as np, random

def Rsqr(observ, model):
    y = np.array(observ)
    f = np.array(model)

    SStot = sum(y**2) - np.size(y)*np.average(y)**2
    SSres = sum((y-f)**2)
    return 1. - SSres/SStot

def sampleWithReplacement(population, choiceSize):
    """Generates a random sample with replacement.

    Chooses \a choiceSize elements with replacement from the \a population
    From http://code.activestate.com/recipes/273085-sample-with-replacement

    \param[in] population  The container to sample from
    \param[in] choiceSize  The size of the random sample to make.
    \return    List of choices of \a choiceSize
    """

    n = len(population)
    _random, _int = random.random, int # speed hack
    return [_int(_random()*n) for _ in itertools.repeat(None, choiceSize)]

class DistributionFitterResult:
    columns = ['Distribution', 'K-S Stat', 'Prob.Match']

    def __init__(self):
        self.result    = dict()
        self.maxParams = 0

    def add(self, name, result, param):
        self.result[name] = (result, param)
        self.maxParams    = max(self.maxParams, len(param))

    def getHeaderList(self):
        paramHeaders = ['Param'+str(idx) for idx in range(1, self.maxParams+1)]
        return self.columns + paramHeaders

    def toCSV(self, outFileName):
        """
        Write the distribution fitter results to a CSV file
        :param outFileName: output CSV file name
        :type  outFileName: str
        :return: Nothing
        :rtype:  None
        """
        with open(outFileName, 'wb') as outFile:
            writer = csv.writer(outFile)
            writer.writerow(self.getHeaderList())
            for name, (results, params) in self.result.iteritems():
                writer.writerow([name] + list(results) + list(params))

    def toExcel(self, outFileName):
        """
        Write the distribution fitter results to an Excel spreadsheet
        :param outFileName: output spreadsheet name
        :type  outFileName: str
        :return: Nothing
        :rtype:  None
        """
        workbook = Workbook(outFileName, {'constant_memory': True})
        workbook.use_zip64() # allow large size Excels just in case

        wks    = workbook.add_worksheet('Distribution Fitting')
        hdrFmt = workbook.add_format({'bold'      : True,
                                      'underline' : True,
                                      'align'     : 'center'})
        resultFormats = [workbook.add_format({'num_format' : fmtStr}) \
                             for fmtStr in ['0.000000', '0.0000%']]

        row = 0
        wks.set_column(0, 0, 11)
        wks.set_column(1, 1,  8,   resultFormats[0])
        wks.set_column(2, 2, 10.6, resultFormats[1])
        for col, headerName in enumerate(self.getHeaderList()):
            wks.write_string(row, col, headerName, hdrFmt)

        for distrName, (results, params) in self.result.iteritems():
            row += 1
            col = 0
            wks.write_string(row, col, distrName)
            for col, (result, outFormat) in \
                     enumerate(itertools.izip(results, resultFormats), col+1):
                wks.write_number(row, col, result, outFormat)
            for col, paramValue in enumerate(params, col+1):
                wks.write_number(row, col, paramValue)

        workbook.close()

class DistributionFitter:
    distrs = SortedSet()

    @classmethod
    def InitDistrList(cls):
        for name, _ in inspect.getmembers(distributions):
            if name[-4:] != '_gen' or name[0] == '_':
                continue
            name = name[:-4]

            distr = getattr(distributions, name)
            if hasattr(distr, 'fit') and hasattr(distr, 'name') and\
               distr.__class__.__base__.__name__ == 'rv_continuous':
                cls.distrs.add(distr)

    @classmethod
    def GetDistrList(cls):
        if not cls.distrs:
            cls.InitDistrList()

        return [distr.name for distr in cls.distrs]

    def __init__(self):
        if not self.distrs:
            self.InitDistrList()

    def __call__(self, sample, distrs    = None, log = None, filePrefix = None,
                               graphAll  = False, minProb = None,
                               graphEach = False, maxStat = None):
        if (graphAll or graphEach) and filePrefix is None:
            raise ValueError('Must pass a valid file prefix for graphing')

        #sample = norm.rvs(loc   = np.mean(sample),
        #                  scale = np.std(sample),
        #                  size  = len(sample))
        fitResult = DistributionFitterResult()

        if graphAll or graphEach:
            sampleMin = min(sample)
            sampleMax = max(sample)
            abscissas = np.linspace(sampleMin, sampleMax, 1000)

        if graphAll:
            figSummary = plt.figure()
            axSummary  = figSummary.add_subplot(1, 1, 1)
            axSummary.set_title('Distribution Fits')
            plt.hist(sample, 20, normed = True, histtype='stepfilled')

        distrs = {getattr(distributions, distr) for distr in distrs} \
                 if distrs else self.distrs

        for idx, distr in enumerate(distrs):
            # find best fitting maximum likelihood parameters by minimization
            # and use them to perform a 2-sided Kolmogorov-Smirnov test
            # WARNING: using this test is an upper bound on the p-value, not
            #          its true value, since its distribution is affected by
            #          estimating parameters.
            if log:
                log.info('Fitting and testing %s', distr.name)
            try:
                param    = distr.fit(sample)
                resultKS = kstest(sample, distr.name, args = param)
            except Exception as e:
                if log:
                    log.exception('Caught exception %s fitting %s; args %s',
                                  type(e).__name__, distr.name, str(e))
                    log.info('Distribution %s skipped', distr.name)
                continue

            # test if resulting statistic clears the maximum threshold
            # formulating through a negative to reject NaN in the comparison
            if not (maxStat is None or resultKS[0] < maxStat):
                if log:
                    log.info('% rejected: K-S stat %f should be < %f',
                             distr.name, resultKS[0], maxStat)
                    continue

            # test if resulting match probability clears the minimum threshold
            # formulating through a negative to reject NaN in the comparison
            if not (minProb is None or resultKS[1] > minProb):
                if log:
                    log.info('% rejected: '
                             'prob match %2.4f\% should be > %2.4f\%',
                             distr.name, 100.*resultKS[1], 100.*minProb)
                    continue

            fitResult.add(name = distr.name, param = param, result = resultKS)

            if graphAll or graphEach:
                pdf = distr.pdf(abscissas, *param)
                if graphAll:
                    axSummary.plot(abscissas, pdf, '-',
                                   linewidth = 2, label = distr.name)
                if graphEach:
                    singlePlotName = filePrefix + '.%s.png' % distr.name
                    if log:
                        log.info('Writing single plot to %s', singlePlotName)
                    figSingle = plt.figure()
                    axSingle = figSingle.add_subplot(1, 1, 1)
                    axSingle.set_title(distr.name + ' Distribution Fit')
                    plt.hist(sample, 20, normed = True, histtype='stepfilled')
                    axSingle.plot(abscissas, pdf, '-',
                                  linewidth = 2, label = distr.name)
                    figSingle.savefig(singlePlotName)
                    plt.close(figSingle)

        if graphAll:
            plt.legend(bbox_to_anchor = (1, 1),
                       bbox_transform = plt.gcf().transFigure)
            summaryPlotName = filePrefix + '.all.png'
            if log:
                log.info('Writing summary plot to %s', summaryPlotName)
            figSummary.savefig(summaryPlotName)
            plt.close(figSummary)
        return fitResult

if __name__ == '__main__':
    x, yObs = np.array([6,5,11,7,5,4,4]), np.array([2,3,9,1,8,7,5])
    m, b = 2.75/9, 9.5/3
    yMod = m*x + b

    print(Rsqr(observ = yObs, model = yMod)) # should be 0.05795

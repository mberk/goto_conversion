convertAmericanOdds = _convertAmericanOdds_py
efficient_shin_conversion = _efficient_shin_conversion_py
goto_conversion = _goto_conversion_py

try:
    import numpy as np
    convertAmericanOdds = _convertAmericanOdds_np
    efficient_shin_conversion = _efficient_shin_conversion_np
    goto_conversion = _goto_conversion_np
except ImportError:
    pass
    

def _convertAmericanOdds_np(listOfOdds):
    listOfOdds = listOfOdds.astype(float)
    isNegativeAmericanOdds = listOfOdds < 0.0
    listOfOdds[isNegativeAmericanOdds] = 1.0 + ((100.0 / listOfOdds[isNegativeAmericanOdds]) * -1.0)
    listOfOdds[~isNegativeAmericanOdds] = 1.0 + (listOfOdds[~isNegativeAmericanOdds] / 100.0)
    return listOfOdds

def _convertAmericanOdds_py(listOfOdds):
    for i in range(len(listOfOdds)):
        currOdds = listOfOdds[i]
        isNegativeAmericanOdds = currOdds < 0.0
        if isNegativeAmericanOdds:
            currDecimalOdds = 1.0 + (100.0/(currOdds*-1.0))
        else: #Is non-negative American Odds
            currDecimalOdds = 1.0 + (currOdds/100.0)
        listOfOdds[i] = currDecimalOdds
    return listOfOdds

def errorCatchers(listOfOdds):
    if len(listOfOdds) < 2:
        raise ValueError('len(listOfOdds) must be >= 2')
    try:
        isAllOddsAbove1 = np.all(listOfOdds > 1.0)
    except:
        isAllOddsAbove1 = all([x > 1.0 for x in listOfOdds])
    if not isAllOddsAbove1:
        raise ValueError('All odds must be > 1.0, set isAmericanOdds parameter to True if using American Odds')

def _efficient_shin_conversion_np(listOfOdds, total = 1.0, multiplicativeIfUnprudentOdds = False, isAmericanOdds = False):

    #Convert American Odds to Decimal Odds
    if isAmericanOdds:
        listOfOdds = convertAmericanOdds(listOfOdds)

    #Error Catchers
    errorCatchers(listOfOdds)

    #Compute parameters
    listOfPies = 1.0 / listOfOdds
    beta = np.sum(listOfPies)
    listOfComplementPies = listOfPies - (beta - listOfPies)

    #Compute vectors
    listOfZ = ((beta - 1.0) * (listOfComplementPies ** 2.0 - beta)) / (beta * (listOfComplementPies ** 2.0 - 1.0))
    listOfPStars = ((np.sqrt(listOfZ**2.0 + 4.0 * (1.0 - listOfZ) * (listOfPies**2 / beta)) - listOfZ) / (2.0 * (1.0 - listOfZ)))
    normalizer = np.sum(listOfPStars) / total
    outputListOfProbabilities = listOfPStars / normalizer

    return outputListOfProbabilities

def _efficient_shin_conversion_py(listOfOdds, total = 1.0, multiplicativeIfUnprudentOdds = False, isAmericanOdds = False):
    
    #Convert American Odds to Decimal Odds
    if isAmericanOdds:
        listOfOdds = convertAmericanOdds(listOfOdds)

    #Error Catchers
    errorCatchers(listOfOdds)

    #Compute parameters
    listOfPies = [1.0/x for x in listOfOdds]
    beta = sum(listOfPies)
    listOfComplementPies = [x - (beta-x) for x in listOfPies]

    #Compute vectors
    listOfZ = [((beta - 1.0)*(x**2.0 - beta))/(beta*(x**2.0 - 1.0)) for x in listOfComplementPies]
    listOfPStars = [(((z_i**2.0 + 4.0*(1.0-z_i)*(pi_i**2.0/beta))**0.5) - z_i)/(2.0*(1.0 - z_i)) for pi_i,z_i in zip(listOfPies, listOfZ)]
    normalizer = sum(listOfPStars)/total
    outputListOfProbabilities = [x/normalizer for x in listOfPStars]

    return outputListOfProbabilities

def _goto_conversion_np(listOfOdds, total = 1.0, multiplicativeIfUnprudentOdds = False, isAmericanOdds = False):

    #Convert American Odds to Decimal Odds
    if isAmericanOdds:
        listOfOdds = convertAmericanOdds(listOfOdds)

    #Error Catchers
    errorCatchers(listOfOdds)

    listOfProbabilities = 1.0 / listOfOdds
    listOfSe = np.sqrt((listOfProbabilities - listOfProbabilities**2.0) / listOfProbabilities)
    step = (np.sum(listOfProbabilities) - total) / np.sum(listOfSe)
    outputListOfProbabilities = listOfProbabilities - (listOfSe * step)
    if np.any(outputListOfProbabilities <= 0.0) or (np.sum(listOfProbabilities) <= 1.0):
        if multiplicativeIfUnprudentOdds:
            normalizer = np.sum(listOfProbabilities) / total
            outputListOfProbabilities = np.array(listOfProbabilities) / normalizer
        else:
            print('Odds must have a positive low bookmaker margin to be prudent.')
            raise ValueError('Set multiplicativeIfUnprudentOdds argument to True to use multiplicative conversion for unprudent odds.')

    return outputListOfProbabilities

def _goto_conversion_py(listOfOdds, total = 1.0, multiplicativeIfUnprudentOdds = False, isAmericanOdds = False):
    
    #Convert American Odds to Decimal Odds
    if isAmericanOdds:
        listOfOdds = convertAmericanOdds(listOfOdds)

    #Error Catchers
    errorCatchers(listOfOdds)

    listOfProbabilities = [1.0/x for x in listOfOdds] #initialize probabilities using inverse odds
    listOfSe = [pow((x-x**2.0)/x,0.5) for x in listOfProbabilities] #compute the standard error (SE) for each probability
    step = (sum(listOfProbabilities) - total)/sum(listOfSe) #compute how many steps of SE the probabilities should step back by
    outputListOfProbabilities = [x - (y*step) for x,y in zip(listOfProbabilities, listOfSe)]
    if any(0.0 >= x for x in outputListOfProbabilities) or (sum(listOfProbabilities) <= 1.0):
        if multiplicativeIfUnprudentOdds:
            normalizer = sum(listOfProbabilities)/total
            outputListOfProbabilities = [x/normalizer for x in listOfProbabilities]
        else:
            print('Odds must have a positive low bookmaker margin to be prudent.')
            raise ValueError('Set multiplicativeIfUnprudentOdds argument to True to use multiplicative conversion for unprudent odds.')

    return outputListOfProbabilities

def zero_sum(listOfPrices, listOfVolumes):
    listOfSe = [x**0.5 for x in listOfVolumes] #compute standard errors assuming standard deviation is same for all stocks
    step = sum(listOfPrices)/sum(listOfSe)
    outputListOfPrices = [x - (y*step) for x,y in zip(listOfPrices, listOfSe)]
    return outputListOfPrices

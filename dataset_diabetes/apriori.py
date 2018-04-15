import numpy as np
import pandas as pd

def gettransactionset(data):
    tranlist = []
    for i in data.index:
        itemset = []
        for k,v in dict(data.iloc[i]).items():
            if isinstance(v, str) or not np.isnan(v): #Handle missing values
                itemset.append(k + ' - ' + str(v))
        tranlist.append(frozenset(itemset))
    return tranlist



def frequent1itemset(transactions, minsup):
    '''
    transactions: list of transaction set
    minsup: absolute minimum support count
    
    returns list one itemsets as well as a dictionary one itemsets with support counts.
    '''
    count = len(transactions)
    oneitems = defaultdict(int)
    oneitemsupport = set()
    
    for transaction in transactions:
        for item in transaction:
            oneitems[item] += 1
            
    for k, v in oneitems.items():
        if v >= (minsup * count):
            oneitemsupport.add(frozenset([k]))
        #else:
        #    del oneitems[k]
            
    return oneitemsupport, oneitems


def selfjoin(prevfreqset, targetlength):
    '''
    This function performs a self joining to determine the K itemset
    prevfreqset: frequent K - 1 itemset
    targetlength: The K value i.e the length of the frequent items
    returns a set of items.
    '''
    #return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == targetlength])
    return set([i.union(j) for i in prevfreqset for j in prevfreqset if len(i.union(j)) == targetlength])

def hasinfrequentsubset(freqitems, previousL):
    '''
    ***Checks if any of the joined frequent items is infrequent.
    freqitems: an item set
    previousL: previous list of frequent itemset.
    returns a boolean value corresponding if the itemset has infrequent subset
    
    '''
    for i in freqitems:
        #seti = set([i])
        if previousL.issuperset([i]):
            return False
    return True


def getfrequentitemsfromL(L, transactions, minsupport):
    '''
    Scans transactions to determine itemsets which are frequent.
    L: List of frequent itemsets
    transactions: list of transactions
    minsupport: absolute minimum support
    
    returns set of frequent itemsets as well as a dictionary frequent items and a support count.
    '''
    
    freqitems = set()
    #itemcounts = set()
    itemcount = defaultdict(int)
    for i in L:
        #itemcount = defaultdict(int)
        for itemset in transactions:
            if i.issubset(itemset):
                itemcount[i] += 1
        if itemcount[i] / len(transactions) >= minsupport:
            freqitems.add(i)
        #else:
        #    del itemcount[i]
    
    return freqitems, itemcount

def generatefrequentitemsApriori(transactions, minsupport):
    '''
    Runs full process to get all frequent itemsets.
    transactions: full list of all transactions
    minsupport: absolute minimum support
    
    returns a set of frequent itemsets and set of dictionary of frequent itemset and each support count.
    '''
    
    starttime = time.time()
    #Get frequent 1-Item sets
    K = 1
    #L1 = frequent1itemset(transactions, minsupport)
    Fulllist = set([])
    CurrentSet, Fulldict = frequent1itemset(transactions, minsupport)
    Lkprev = set()
    K += 1
    while CurrentSet != set([]):
        #print('Loop %i, %i'%(K, len(Fulllist)))
        Lkprev.union(CurrentSet)
        CurrentSet = selfjoin(CurrentSet, K)
    #Using Apriori rule
        inf = set()
        for i in CurrentSet:
            if hasinfrequentsubset(i, Lkprev):
                inf.add(i)
        CurrentSet.discard(inf)
    #Scan transaction list for minimum support
        CurrentSet, CurrentD = getfrequentitemsfromL(CurrentSet, transactions, minsupport)
        Fulllist = Fulllist.union(CurrentSet)
        Fulldict.update(CurrentD)
        K += 1
        
    return Fulllist, Fulldict, time.time() - starttime


def subsets(myset):
    """ Returns non empty subsets of arr"""
    cs = chain(*[combinations(myset, i + 1) for i, a in enumerate(myset)])
    return map(frozenset, [x for x in cs])


def getffrequentassociations(frequentset, frequentsetcounters, minconf, numoftrans):
    '''
    frequentset: Set of frequent items from which rules are to be obtained.
    frequentsetcounters: Set of frequent items with their support count.
    minconf: minumum confidence which must hold for a rule to be frequent.
    numoftrans: number of transactions which is being mined.
    '''
    def getsupport(item):
        '''
        item: Item in frequent itemset whose minimums support is to be determined.
        Returns float value of support value.
        '''
        if len(item) == 1:
            item = list(item)[0]
        return float(frequentsetcounters[item])/numoftrans
    
    frequentrules = []
    for freqitem in frequentset:
        #Get Subsets for frequent item
        subs = subsets(freqitem)
        for element in subs:
            rem = freqitem.difference(element)
            if len(rem) > 0:
                conf = getsupport(freqitem)/getsupport(element)
                if conf >= minconf:
                    frequentrules.append(((tuple(element), tuple(rem)), conf, getsupport(freqitem)))
                    
    #return sorted(frequentrules, reverse=True, key = lambda rule, conf, sup: sup)
    return frequentrules


## Frequent Itemset Miner for Apriori Algorithms...

def associationruleminer(transaction, minsup, minconf):
    '''
    transaction: The list of transactions to be mined in pandas dataframe
    minsupport: The absolute minimum support
    minconf: The absolute value of the minimum confidence
    '''
    import time
    start_time = time.time()
    #Convert transaction dataframe into a set of items.
    transactions = gettransactionset(transaction)
    
    #Get frequent itemsets and itemset support count
    itemset, itemsetsupport, t = generatefrequentitemsApriori(transactions, minsup)
    
    #Mine association rules from frequent itemsets
    assocrules = getffrequentassociations(itemset, itemsetsupport, minconf=minconf, numoftrans=len(transactions))
    
    #print out association rules
    
    for i, rule in enumerate(assocrules):
        print('Rule %i: %s => %s - support: %0.2f, confidence: %0.2f' %(i, rule[0][0], rule[0][1], rule[2], rule[1]))
        
    return (time.time() - start_time)
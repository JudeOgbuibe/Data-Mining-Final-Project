import numpy as np
import pandas as pd

def gettransactionset(data):
    tranlist = []
    for i in data.index:
        itemset = []
        for k,v in dict(data.loc[i]).items():
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
    
    #starttime = time.time()
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
        
    return Fulllist, Fulldict#, time.time() - starttime


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
        #if len(item) == 1:
        #    item = frozenset([list(item)[0]])
        try:
            result = float(frequentsetcounters[item])/numoftrans
        except:
            #print 'error obtained for item' + str(item)
            result = 0
        return result #float(frequentsetcounters[item])/numoftrans
    
    frequentrules = []
    for freqitem in frequentset:
        #Get Subsets for frequent item
        subs = subsets(freqitem)
        for element in subs:
            rem = freqitem.difference(element)
            if len(rem) > 0:
                conf = getsupport(freqitem)/getsupport(element)
                support = getsupport(freqitem)
                lift = conf/getsupport(frozenset(rem))
                if conf >= minconf:
                    frequentrules.append(((tuple(element), tuple(rem)),conf,support,lift))    
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
        
    return assocrules#(time.time() - start_time)

class FPNode(object):
    '''
    Implements a node of the FP tree
    '''

    def __init__(self, value, count, parent):
        """
        Initializes the node.
        """
        self.value = value
        self.count = count
        self.parent = parent
        self.link = None
        self.children = []

    def has_child(self, value):
        """
        Checks if the node has children
        """
        for node in self.children:
            if node.value == value:
                return True

        return False

    def get_child(self, value):
        """
        Return a child node with a particular value.
        """
        for node in self.children:
            if node.value == value:
                return node

        return None

    def add_child(self, value):
        """
        Add a node as a child node.
        """
        child = FPNode(value, 1, self)
        self.children.append(child)
        return child


class FPTree(object):
    """
    A frequent pattern tree.
    """

    def __init__(self, transactions, threshold, root_value, root_count):
        """
        Initialize the tree.
        """
        self.frequent = self.find_frequent_items(transactions, threshold)
        self.headers = self.build_header_table(self.frequent)
        self.root = self.build_fptree(
            transactions, root_value,
            root_count, self.frequent, self.headers)

    @staticmethod
    def find_frequent_items(transactions, threshold):
        """
        Create a dictionary of items with occurrences above the threshold.
        """
        items = {}

        for transaction in transactions:
            for item in transaction:
                if item in items:
                    items[item] += 1
                else:
                    items[item] = 1

        for key in list(items.keys()):
            if items[key] < threshold:
                del items[key]

        return items

    @staticmethod
    def build_header_table(frequent):
        """
        Build the header table.
        """
        headers = {}
        for key in frequent.keys():
            headers[key] = None

        return headers

    def build_fptree(self, transactions, root_value,
                     root_count, frequent, headers):
        """
        Build the FP tree and return the root node.
        """
        root = FPNode(root_value, root_count, None)

        for transaction in transactions:
            sorted_items = [x for x in transaction if x in frequent]
            sorted_items.sort(key=lambda x: frequent[x], reverse=True)
            if len(sorted_items) > 0:
                self.insert_tree(sorted_items, root, headers)

        return root

    def insert_tree(self, items, node, headers):
        """
        Recursively grow FP tree.
        """
        first = items[0]
        child = node.get_child(first)
        if child is not None:
            child.count += 1
        else:
            # Add new child.
            child = node.add_child(first)

            # Link it to header structure.
            if headers[first] is None:
                headers[first] = child
            else:
                current = headers[first]
                while current.link is not None:
                    current = current.link
                current.link = child

        # Call function recursively.
        remaining_items = items[1:]
        if len(remaining_items) > 0:
            self.insert_tree(remaining_items, child, headers)

    def tree_has_single_path(self, node):
        """
        If there is a single path in the tree,
        return True, else return False.
        """
        num_children = len(node.children)
        if num_children > 1:
            return False
        elif num_children == 0:
            return True
        else:
            return True and self.tree_has_single_path(node.children[0])

    def mine_patterns(self, threshold):
        """
        Mine the constructed FP tree for frequent patterns.
        """
        if self.tree_has_single_path(self.root):
            return self.generate_pattern_list()
        else:
            return self.zip_patterns(self.mine_sub_trees(threshold))

    def zip_patterns(self, patterns):
        """
        Append suffix to patterns in dictionary if
        we are in a conditional FP tree.
        """
        suffix = self.root.value

        if suffix is not None:
            # We are in a conditional tree.
            new_patterns = {}
            for key in patterns.keys():
                new_patterns[tuple(sorted(list(key) + [suffix]))] = patterns[key]

            return new_patterns

        return patterns

    def generate_pattern_list(self):
        """
        Generate a list of patterns with support counts.
        """
        patterns = {}
        items = self.frequent.keys()

        # If we are in a conditional tree,
        # the suffix is a pattern on its own.
        if self.root.value is None:
            suffix_value = []
        else:
            suffix_value = [self.root.value]
            patterns[tuple(suffix_value)] = self.root.count

        for i in range(1, len(items) + 1):
            for subset in itertools.combinations(items, i):
                pattern = tuple(sorted(list(subset) + suffix_value))
                patterns[pattern] = \
                    min([self.frequent[x] for x in subset])

        return patterns

    def mine_sub_trees(self, threshold):
        """
        Generate subtrees and mine them for patterns.
        """
        patterns = {}
        mining_order = sorted(self.frequent.keys(),
                              key=lambda x: self.frequent[x])

        # Get items in tree in reverse order of occurrences.
        for item in mining_order:
            suffixes = []
            conditional_tree_input = []
            node = self.headers[item]

            # Follow node links to get a list of
            # all occurrences of a certain item.
            while node is not None:
                suffixes.append(node)
                node = node.link

            # For each occurrence of the item, 
            # trace the path back to the root node.
            for suffix in suffixes:
                frequency = suffix.count
                path = []
                parent = suffix.parent

                while parent.parent is not None:
                    path.append(parent.value)
                    parent = parent.parent

                for i in range(frequency):
                    conditional_tree_input.append(path)

            # Now we have the input for a subtree,
            # so construct it and grab the patterns.
            subtree = FPTree(conditional_tree_input, threshold,
                             item, self.frequent[item])
            subtree_patterns = subtree.mine_patterns(threshold)

            # Insert subtree patterns into main patterns dictionary.
            for pattern in subtree_patterns.keys():
                if pattern in patterns:
                    patterns[pattern] += subtree_patterns[pattern]
                else:
                    patterns[pattern] = subtree_patterns[pattern]

        return patterns


def find_frequent_patterns(transactions, min_sup):
    """
    Transactions: As a list of transaction sets
    min_sup: Absolute minimum support 
    """
    
    starttime = time.time()
    support_threshold =  min_sup * len(transactions)
    tree = FPTree(transactions, support_threshold, None, None)
    
    result = tree.mine_patterns(support_threshold)
    freqitems = set([])
    resultdict = {}
    for itemset, values in result.items():
        freqitems.add(frozenset(itemset))
        resultdict[freqitems.add(frozenset(itemset))] = values
        
    return freqitems, resultdict#, time.time() - starttime
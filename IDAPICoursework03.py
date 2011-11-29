#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
import copy

##############################
#  Coursework 1 begins here  #
##############################

# PLEASE NOTE: The code was optimized for readability, rather than absolute
#              efficency.

# Function to compute the prior distribution of the variable root from the
# data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
# Coursework 1 task 1 should be inserted here
    for val in theData[0:, root]:
        prior[val] += 1
    prior /= theData.shape[0]

# end of Coursework 1 task 1
    return prior

# Function to compute a CPT with parent node varP and xchild node varC from
# the data array it is assumed that the states are designated by consecutive
# integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
# Coursework 1 task 2 should be inserte4d here
    # extract only the relevant columns
    vecC = theData[0:, varC]
    vecP = theData[0:, varP]

    # count N(C&P)
    for i in range(theData.shape[0]):
        cPT[vecC[i], vecP[i]] += 1

    # divide by N(P)
    for i, row in enumerate(cPT.transpose()):
        # divide by <count the number of states equal to |i|>
        row /= len(filter(lambda state: state == i, vecP))

# end of coursework 1 task 2
    return cPT

# Function to calculate the joint probability table of two variables in the
# data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here 
    # extract only the relevant columns
    vecRow = theData[0:, varRow]
    vecCol = theData[0:, varCol]

    # count the number of the occurences of the pairs: N(R & C)
    for i in range(theData.shape[0]):
        jPT[vecRow[i], vecCol[i]] += 1

    # divide by number of data points
    for row in jPT.transpose():
        row /= theData.shape[0]

# end of coursework 1 task 3
    return jPT

#
# Function to convert a joint probability table to a conditional probability
# table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here 
    # normalize the JPT so columns sum to 1
    for row in aJPT.transpose():
        row /= sum(row)

# coursework 1 taks 4 ends here
    return aJPT

#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
# Coursework 1 task 5 should be inserted here
    prior = naiveBayes[0]
    
    # A function returning the i-th CPT (indexed from 0)
    CPT = lambda i: naiveBayes[i + 1]

    # calculate distribution (without normalizing)
    for rootState, _ in enumerate(rootPdf):
        # A function returning P(var | root) from the CPT for that var
        ConditionalProbability = lambda var: CPT(var)[theQuery[var], rootState]

        rootPdf[rootState] = prior[rootState]
        rootPdf[rootState] *= multiply.reduce(map(ConditionalProbability,
                                                      range(0, len(theQuery)) ))

    # normalize
    alpha = sum(rootPdf)
    for i, p in enumerate(rootPdf):
        rootPdf[i] /= alpha

# end of coursework 1 task 5
    return rootPdf

#
# End of Coursework 1
#
def Cw1Main(log):
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
    theData = array(datain)

    filename = "Results01.txt"

    # Clear the contents of the file
    open(filename, 'w').close()

    # Produce the results and write to the file
    AppendString(filename,"Coursework One Results by mlo08")
    AppendString(filename,"") #blank line
    AppendString(filename,"The prior probability of node 0")

    prior = Prior(theData, 0, noStates)
    AppendList(filename, prior)
    if (log) : print(prior)

    cPT = CPT(theData, 2, 0, noStates)
    AppendArray(filename, cPT)
    if (log) : print(cPT)

    jPT = JPT(theData, 2, 0, noStates)
    AppendArray(filename, jPT)
    if (log) : print(jPT)

    JPT2CPT(jPT)
    AppendArray(filename, jPT)
    if (log) : print(jPT)

    # Prepare the network input for the Query
    naiveBayes = [prior] + map(lambda c: CPT(theData, c, 0, noStates), range(1, 6))
    if (log) : print(naiveBayes)

    dist = Query([4,0,0,0,5], naiveBayes)
    AppendList(filename, dist)
    if (log) : print(dist)

    dist = Query([6,5,2,5,5], naiveBayes)
    AppendList(filename, dist)
    if (log) : print(dist)

##############################
#  Coursework 2 begins here  #
##############################

# PLEASE NOTE: The code was sometimes optimized for readability, rather than
#              absolute efficency.

def memoize(f):
    cache = {}
    def memoizedF(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    return memoizedF

# Calculate the mutual information from the joint probability table of two
# variables
def MutualInformation(jP):
    mi=0.0
# Coursework 2 task 1 should be inserted here
    def log2(x):
        if x == 0:
            return 0
        return math.log(x, 2)

    @memoize
    def PCol(col):
        return sum(jP[0:, col])
    
    @memoize
    def PRow(row):
        return sum(jP[row])

    mi = sum([jP[row][col] * log2(jP[row][col] / (PCol(col) * PRow(row))) \
                for row in range(jP.shape[0]) \
                for col in range(jP.shape[1]) \
                if jP[row][col] != 0])
# end of coursework 2 task 1
    return mi


# Constructs a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    for row in range(MIMatrix.shape[0]):
        for col in range(MIMatrix.shape[1]):
            MIMatrix[row][col] = MutualInformation(JPT(theData, row, col, noStates))
# end of coursework 2 task 2
    return MIMatrix


# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    depList = [ [depMatrix[row, col], row, col] \
                    for col in range(depMatrix.shape[1]) \
                    for row in range(col, depMatrix.shape[0]) ]
    depList.sort(key = lambda arc: arc[0], reverse=True)
# end of coursework 2 task 3
    return array(depList)


# Functions implementing the spanning tree algorithm
# It implements the Kruskal's algorithm for finding maximum spanning tree,
# using python sets
def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
# Coursework 2 task 4 should be inserted here
    
    # A a collection of clusters (i.e. a forest of trees) in Kruskal's
    # algorithm
    verticesClusters = []

    def AddsCycle(edgeVertices):
        # all the clusters that the edge has at least one common vertex with
        # (the size of this list is between 0 and 2)
        candidateClusters \
            = [[cluster, size] \
                    for cluster in verticesClusters \
                    for size in [len(cluster.intersection(edgeVertices))] \
                    if size > 0]
        VerticesOf             = lambda cls: cls[0]
        NumberOfEdgeVerticesIn = lambda cls: cls[1]

        # if the list is empty -> add new equivalence class (i.e. new cluster)
        if not candidateClusters:
            verticesClusters.append(edgeVertices)
            return False

        clusterOne = candidateClusters[0]

        # if the edge belongs to two clusters -> merge them together
        if len(candidateClusters) == 2:
            clusterTwo = candidateClusters[1]
            VerticesOf(clusterOne).update(VerticesOf(clusterTwo))
            verticesClusters.remove(VerticesOf(clusterTwo))
            return False

        # both vertices of the edge are already in the same cluster -> a loop!
        if NumberOfEdgeVerticesIn(clusterOne) == 2:
            return True

        # just one vertex in one cluster -> add a vertex to this cluster
        VerticesOf(clusterOne).update(edgeVertices)
        return False

    # Main loop of Kruskal's algorithm. We greedily add edges (but not self
    # loops) to the graph if they don't add a cycle (a loop) until we have a
    # spanning tree.
    for edge in depList:
        if edge[1] == edge[2]: continue
        if len(spanningTree) == noVariables - 1: break

        if not AddsCycle(set(edge[1:])):
            spanningTree.append(edge)

# end of coursework 2 task 4
    return array(spanningTree)

#
# End of coursework 2
#
def Cw2Main(log):
    inputFile = "HepatitisC.txt"
    noVariables, noRoots, noStates, noDataPoints, datain \
            = ReadFile(inputFile)

    theData = array(datain)

    filename = "Results02.txt"

    # Clear the contents of the file
    open(filename, 'w').close()

    # Produce the results and write to the file
    AppendString(filename,"Coursework Two Results by mlo08")
    AppendString(filename,"") #blank line

    dM = DependencyMatrix(theData, noVariables, noStates)
    headline = "Dependency Matrix for " + inputFile + ": "
    if (log) :
        print headline
        print(dM)
        print ""
    AppendString(filename, headline)
    AppendArray(filename, dM)

    dL = DependencyList(dM)
    headline = "Dependency List for " + inputFile + ": "
    if (log) :
        print headline
        print(dL)
        print ""
    AppendString(filename, headline)
    AppendArray(filename, dL)

    sTA = SpanningTreeAlgorithm(dL, noVariables)
    headline = "Maximum Spanning Tree for " + inputFile + ": "
    if (log) :
        print headline
        print(sTA)
    AppendString(filename, headline)
    AppendArray(filename, sTA)

############################
# Coursework 3 begins here #
############################

# PLEASE NOTE: The code was sometimes optimized for readability, rather than
#              absolute efficency.

# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers
# starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here
    # extract only the relevant columns
    vecChild   = theData[0:, child]
    vecParent1 = theData[0:, parent1]
    vecParent2 = theData[0:, parent2]

    # this will keep the sum of the 1s in the given columns
    # (assuming we treat the CPT as 2D matrix P(C|P1&P2), like
    # in the lecture notes)
    colSums = zeros([noStates[parent1], noStates[parent2]])

    # count N(C&P1&P2) and colSums
    for i in range(theData.shape[0]):
        cPT[vecChild[i], vecParent1[i], vecParent2[i]] += 1
        colSums[vecParent1[i], vecParent2[i]] += 1

    for p1 in range(noStates[parent1]):
        for p2 in range(noStates[parent2]):
            if (colSums[p1, p2] != 0):
                cPT[:, p1, p2] /= colSums[p1, p2]
            else:
                cPT[:, p1, p2] = 1.0 / noStates[child]
# End of Coursework 3 task 1           
    return cPT

# Function to compute the list of CPTs from arcList for
# the definitions of example Bayesian Networks
def getCPTList(theData, noStates, arcList):
    cptList = []
    for arcs in arcList:
        if len(arcs) == 1:
            cptList.append(Prior(theData, arcs[0], noStates))
        elif len(arcs) == 2:
            cptList.append(CPT(theData, arcs[0], arcs[1], noStates))
        elif len(arcs) == 3:
            cptList.append(CPT_2(theData, arcs[0], arcs[1], arcs[2], noStates))
    return cptList;

# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,2,1],[4,3],[5,3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList

# Coursework 3 task 2 begins here
def HepatitisCBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,4],[4,1],[5,4],[6,1],[7,0,1],[8,7]]
    ''' The data entered manually:
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT(theData, 3, 4, noStates)
    cpt4 = CPT(theData, 4, 1, noStates)
    cpt5 = CPT(theData, 5, 4, noStates)
    cpt6 = CPT(theData, 6, 1, noStates)
    cpt7 = CPT_2(theData, 7, 0, 1, noStates)
    cpt8 = CPT(theData, 8, 7, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5, cpt7, cpt8]
    '''
    # The data computed from arcList
    cptList = getCPTList(theData, noStates, arcList)
    return arcList, cptList
# end of coursework 3 task 2

# An example from Tutorial 5
def TutorialBayesianNetwork(theData, noStates):
    arcList = [[0],[1,0],[2,0],[3,1]]
    '''
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = CPT(theData, 1, 0, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT(theData, 3, 1, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3]
    '''
    cptList = getCPTList(theData, noStates, arcList)
    return arcList, cptList

# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here
    for cpt in cptList:
        size = cpt.shape[0] - 1
        for s in cpt.shape[1:]:
            size *= s
        mdlSize += size

    mdlSize *= math.log(noDataPoints, 2) / 2;
# Coursework 3 task 3 ends here 
    return mdlSize 

# Function to calculate the joint probability of a single data point
# in a Network
def JointProbabilityB(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here
    for (index, arcs) in enumerate(arcList):
        cpt = cptList[index]
        noNodes = len(arcs)
        ind = zeros(noNodes)
        for i in range(noNodes):
            node = arcs[i]
            state = dataPoint[node]
            ind[i] = state
        prob = cpt[tuple(ind)]
        jP *= prob
# Coursework 3 task 4 ends here 
    return jP

# Function to calculate the joint probability of a single data point
# in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here
    for (i, arcs) in enumerate(arcList):
        cpt          = cptList[i]
        cptCoords    = zeros(cpt.ndim)
        for crd in range(cpt.ndim):
            cptCoords[crd] = dataPoint[arcs[crd]]

        jP *= cpt[tuple(cptCoords)]
# Coursework 3 task 4 ends here 
    return jP

# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy = 0
# Coursework 3 task 5 begins here
    for dataPoint in theData:
        mdlAccuracy += \
                math.log(JointProbability(dataPoint, arcList, cptList), 2)
# Coursework 3 task 5 ends here 
    return mdlAccuracy

# Function to calculate MDLScore from a data set
def MDLScore(theData, noDataPoints, noStates, arcList, cptList):
    mDLSize     = MDLSize(arcList, cptList, noDataPoints, noStates)
    mDLAccuracy = MDLAccuracy(theData, arcList, cptList)

    return MDLScore2(mDLSize, mDLAccuracy)

# Function to calculate MDLScore from size and accuracy
def MDLScore2(mDLSize, mDLAccuracy):
    return mDLSize - mDLAccuracy

# Function to find the best scoring network formed by deleting one arc
# from the spanning tree
def FindBestNetwork(theData, noDataPoints, noStates, arcList):
    bestMDLScore = 0.0;
    bestArcList  = []
    bestCPTList  = []

    for arcs in arcList:
        for parent in arcs[1:]:
            arcs.remove(parent)

            cptList  = getCPTList(theData, noStates, arcList)
            mDLScore = \
                    MDLScore(theData, noDataPoints, noStates, arcList, cptList)

            if ((-1) * mDLScore < bestMDLScore):
                bestMDLScore = mDLScore
                bestArcList  = copy.deepcopy(arcList)
                bestCPTList  = cptList

            arcs.append(parent)

    return bestMDLScore, bestArcList, bestCPTList
#
# End of coursework 3
#
def Cw3Main(log):
    inputFile = "HepatitisC.txt"
    noVariables, noRoots, noStates, noDataPoints, datain \
            = ReadFile(inputFile)

    theData = array(datain)

    filename = "Results03.txt"

    # Clear the contents of the file
    open(filename, 'w').close()

    # Produce the results and write to the file
    AppendString(filename,"Coursework Three Results by mlo08")
    AppendString(filename,"") #blank line

    arcList, cptList = HepatitisCBayesianNetwork(theData, noStates)

    mDLSize = MDLSize(arcList, cptList, noDataPoints, noStates)
    headline = "MDL size for " + inputFile + ": "
    if (log) :
        print headline
        print(mDLSize)
        print ""
    AppendString(filename, headline)
    AppendString(filename, mDLSize)
    AppendString(filename,"") #blank line

    mDLAccuracy = MDLAccuracy(theData, arcList, cptList)
    headline = "MDL accuracy for " + inputFile + ": "
    if (log) :
        print headline
        print(mDLAccuracy)
        print ""
    AppendString(filename, headline)
    AppendString(filename, mDLAccuracy)
    AppendString(filename,"") #blank line

    mDLScore = MDLScore2(mDLSize, mDLAccuracy)
    headline = "MDL score for " + inputFile + ": "
    if (log) :
        print headline
        print(mDLScore)
        print ""
    AppendString(filename, headline)
    AppendString(filename, mDLScore)
    AppendString(filename,"") #blank line

    bestMDLScore, bestArcList, bestCPTList = \
            FindBestNetwork(theData, noDataPoints, noStates, arcList)
    headline = "Best network score for " + inputFile + ": "
    if (log) :
        print headline
        print(bestMDLScore)
        print(bestArcList)
        print(bestCPTList)
        print ""
    AppendString(filename, headline)
    AppendString(filename, bestMDLScore)
    AppendArray(filename, bestArcList)

# main program part for Coursework 3
#
Cw3Main(True)

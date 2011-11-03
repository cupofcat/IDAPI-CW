#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
#
# Coursework 1 begins here
#

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

def memoize(f):
    cache = {}
    def memoizedF(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    return memoizedF

def divide(a, b):
    if a == 0:
        return 0
    return a/b

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

    for row in range(jP.shape[0]):
        for col in range(jP.shape[1]):
            mi += jP[row][col] * log2(divide(jP[row][col], (PCol(col) * PRow(row))) )
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
    for col in range(depMatrix.shape[1]):
        for row in range(col, depMatrix.shape[0]):
            depList.append([depMatrix[row, col], row, col])
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
    clusters = []

    def AddsCycle(edge):
        # all the clusters that the edge has at least one common vertex with
        # (the size of this list is between 0 and 2)
        candidateClusters = [[cluster, size] \
                                for cluster in clusters \
                                for size in [len(cluster.intersection(edge))] \
                                if size > 0]
        Cluster = lambda cls: cls[0]
        Size    = lambda cls: cls[1]

        # if the list is empty -> add new equivalence class (i.e. new cluster)
        if not candidateClusters:
            clusters.append(edge)
            return False

        clusterOne = candidateClusters[0]

        # if the edge belongs to two clusters -> merge them together
        if len(candidateClusters) == 2:
            clusterTwo = candidateClusters[1]
            Cluster(clusterOne).update(Cluster(clusterTwo))
            clusters.remove(Cluster(clusterTwo))
            return False

        # both vertices of the edge are already in the same cluster -> a loop!
        if Size(clusterOne) == 2:
            return True

        # just one vertex in one cluster -> add a vertex to this cluster
        Cluster(clusterOne).update(edge)
        return False

    # Main loop of Kruskal's algorithm. We greedily add edges to the graph
    # if they don't add a cycle (a loop) until we have a spanning tree.
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
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
    theData = array(datain)

    filename = "Results02.txt"

    # Clear the contents of the file
    #open(filename, 'w').close()

    # Produce the results and write to the file
    #AppendString(filename,"Coursework Two Results by mlo08")
    #AppendString(filename,"") #blank line

    jPT = JPT(theData, 0, 0, noStates)
    if (log) : print(jPT)

    mI = MutualInformation(jPT)
    if (log) : print(mI)

    dM = DependencyMatrix(theData, noVariables, noStates)
    if (log) : print(dM)

    dL = DependencyList(dM)
    if (log) : print(dL)

    sTA = SpanningTreeAlgorithm(dL, noVariables)
    if (log) : print(sTA)

    #print(DependencyMatrix(theData, noVariables, noStates))

#
# main program part for Coursework 1
#
Cw2Main(True)
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

#
# main program part for Coursework 1
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
theData = array(datain)

log = False
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

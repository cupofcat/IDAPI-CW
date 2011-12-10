#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *

##############################
#  Coursework 4 begins here  #
##############################

# PLEASE NOTE: The code was optimized for readability, rather than absolute
#              efficency.

#
# Calculates the mean vector of the data set
#
def Mean(theData):
    return numpy.mean(theData.astype(float), 0)

#
# Calculates the covariance matrix of the data set
#
def Covariance(theData):
    return numpy.cov(theData.astype(float), rowvar = 0)

#
# Creates eigenfaces' files from the basis
#
def CreateEigenfaceFiles(theBasis, fileNamePrefix):
    fileNames = []
    for i, eigenFace in enumerate(theBasis):
        fileName = fileNamePrefix + str(i) + ".jpg"
        fileNames.append(fileName)
        SaveEigenface(eigenFace, fileName)
    return fileNames

#
# Projects an image file onto the PCA space
#
def ProjectFace(theBasis, theMean, theFaceImage):
    face = array(ReadOneImage(theFaceImage))
    meanFace = face - theMean
    return array(dot(meanFace, transpose(theBasis)))

#
# Reconstructs the image from PCA space, saving all the
# intermidiate results
#
def CreatePartialReconstructions(aBasis, aMean, componentMags, fileNamePrefix):
    fileNames = []
    reconstruction = aMean;

    fileName = fileNamePrefix + str(0) + ".jpg"
    fileNames.append(fileName)
    SaveEigenface(reconstruction, fileName)

    for i in range(aBasis.shape[0] - 1): 
        reconstruction += componentMags[i] * aBasis[i, :]

        fileName = fileNamePrefix + str(i + 1) + ".jpg"
        fileNames.append(fileName)
        SaveEigenface(reconstruction, fileName)
    
    return fileNames

#
# Performs PCA of the data
#
def PrincipalComponents(theData):
    # Karhunen-Lowe trick
    u = theData - Mean(theData)
    uuT = dot(u, u.T)
    uuTEigenVals, uuTEigenVecs = linalg.eig(uuT)
    uTuEigenVecs = dot(u.T, uuTEigenVecs)

    # Normalize
    for i in range(uTuEigenVecs.shape[1]):
        col = uTuEigenVecs[:, i]
        uTuEigenVecs[:, i] = col / sqrt(dot(col, col.T))
    
    # Sort
    l = zip(uuTEigenVals, uTuEigenVecs.T)
    list.sort(l, reverse = True)

    _, result = zip(*l)
    
    return array(result)

#
# End of Coursework 4
#
def Cw4Main(log):
    inputFile = "HepatitisC.txt"
    noVariables, noRoots, noStates, noDataPoints, datain \
            = ReadFile(inputFile)

    theData = array(datain)

    filename = "IDAPIResults04.txt"

    # Clear the contents of the file
    open(filename, 'w').close()

    # Produce the results and write to the file
    AppendString(filename,"Coursework Four Results by mlo08")
    AppendString(filename,"") #blank line

    meanHepC = Mean(theData)
    headline = "Mean vector for " + inputFile + ": "
    if (log) :
        print headline
        print(meanHepC)
        print ""
    AppendString(filename, headline)
    AppendString(filename, meanHepC)
    AppendString(filename,"") #blank line

    covHepC = Covariance(theData)
    headline = "Covaraince matrix for " + inputFile + ": "
    if (log) :
        print headline
        print(covHepC)
        print ""
    AppendString(filename, headline)
    AppendString(filename, covHepC)
    AppendString(filename,"") #blank line

    # Given

    givenBasis = ReadEigenfaceBasis()
    givenMean = array(ReadOneImage("MeanImage.jpg"))
    givenMags = ProjectFace(givenBasis, givenMean, "c.pgm")

    headline = "Components magnitudes for c.pgm: "
    if (log) :
        print headline
        print(givenMags)
        print ""
    AppendString(filename, headline)
    AppendList(filename, givenMags)
    AppendString(filename,"") #blank line

    givenEigenFaces = CreateEigenfaceFiles(givenBasis, "GivenEigenFace")
    headline = "Given eigenfaces:"
    if (log) :
        print headline
        print(givenEigenFaces)
        print ""

    givenReconstructions \
            = CreatePartialReconstructions(givenBasis, givenMean, givenMags, "GivenReconstructionC")
    headline = "Partial reconstructions of c.pgm from given eigenfaces:"
    if (log) :
        print headline
        print(givenReconstructions)
        print ""
    
    # 4.6 Calculations

    images = array(ReadImages())
    afBasis = PrincipalComponents(images)
    afMean = Mean(images)
    afMags = ProjectFace(afBasis, afMean, "c.pgm")

    afEigenFaces = CreateEigenfaceFiles(afBasis, "AFEigenFace")
    headline = "A-F eigenfaces:"
    if (log) :
        print headline
        print(afEigenFaces)
        print ""

    afReconstructions \
            = CreatePartialReconstructions(afBasis, afMean, afMags, "AFReconstructionC")
    headline = "Partial reconstructions of c.pgm from A-F eigenfaces:"
    if (log) :
        print headline
        print(afReconstructions)
        print ""

# main program part for Coursework 4
#
Cw4Main(True)

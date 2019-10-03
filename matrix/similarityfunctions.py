import math

def calculateCosine(vect1, vect2):
    cosine = float(0)
    if len(vect1) < len(vect2):
        for j in vect1:
            if j in vect2:
                cosine += (vect1[j] * vect2[j])
    else:
        for j in vect2:
            if j in vect1:
                cosine += (vect1[j] * vect2[j])
    return cosine
    
def calculateSkewDivergence(vect1, vect2, alpha = 0.99):
    skewDiv = float(0)
    featList = []
    featList.extend(vect1.keys())
    featList.extend(vect2.keys())
    featList = list(set(featList))
    mixVect = {}
    for j in featList:
        if j in vect2 and j in vect1:
            mixVect[j] = (alpha * vect2[j]) + ( (1 - alpha) * vect1[j] )
        elif j in vect2 and not j in vect1:
            mixVect[j] = alpha * vect2[j]
        elif j in vect1 and not j in vect2:
            mixVect[j] = (1 - alpha) * vect1[j]
    for j in vect1:
        skewDiv += vect1[j] * math.log(vect1[j] / mixVect[j])
    return skewDiv

def calculateJSDivergence(vect1,vect2):
    JSDiv1 = float(0)
    JSDiv2 = float(0)
    featList = []
    featList.extend(vect1.keys())
    featList.extend(vect2.keys())
    featList = list(set(featList))
    averageVect = {}
    for j in featList:
        if j in vect2 and j in vect1:
            averageVect[j] = (vect1[j] + vect2[j]) / 2
        elif j in vect2 and not j in vect1:
            averageVect[j] = vect2[j] / 2
        elif j in vect1 and not j in vect2:
            averageVect[j] = vect1[j] / 2
    for j in vect1:
        JSDiv1 += vect1[j] * math.log(vect1[j] / averageVect[j])
    for j in vect2:
        JSDiv2 += vect2[j] * math.log(vect2[j] / averageVect[j])
    JSDiv = ( 0.5 * JSDiv1 ) + ( 0.5 * JSDiv2 )
    return JSDiv

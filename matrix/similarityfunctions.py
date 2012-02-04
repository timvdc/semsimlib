import math

def calculateCosine(vect1, vect2):
    cosine = float(0)
    if len(vect1) < len(vect2):
        for j in vect1:
            if vect2.has_key(j):
                cosine += (vect1[j] * vect2[j])
    else:
        for j in vect2:
            if vect1.has_key(j):
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
        if vect2.has_key(j) and vect1.has_key(j):
            mixVect[j] = (alpha * vect2[j]) + ( (1 - alpha) * vect1[j] )
        elif vect2.has_key(j) and not vect1.has_key(j):
            mixVect[j] = alpha * vect2[j]
        elif vect1.has_key(j) and not vect2.has_key(j):
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
        if vect2.has_key(j) and vect1.has_key(j):
            averageVect[j] = (vect1[j] + vect2[j]) / 2
        elif vect2.has_key(j) and not vect1.has_key(j):
            averageVect[j] = vect2[j] / 2
        elif vect1.has_key(j) and not vect2.has_key(j):
            averageVect[j] = vect1[j] / 2
    for j in vect1:
        JSDiv1 += vect1[j] * math.log(vect1[j] / averageVect[j])
    for j in vect2:
        JSDiv2 += vect2[j] * math.log(vect2[j] / averageVect[j])
    JSDiv = ( 0.5 * JSDiv1 ) + ( 0.5 * JSDiv2 )
    return JSDiv

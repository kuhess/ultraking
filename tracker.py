import numpy
import cv


def warpImg(inImg, polygon):
    """Extract the subimage of the input image inside the polygon."""
    mapMatrix = polygon.mapMatrix()
    outImg = cv.CreateImage(polygon.outSize, inImg.depth, inImg.channels)
    cv.GetQuadrangleSubPix(inImg, outImg, mapMatrix)
    return outImg


def skl(data, U0=None, D0=None, mu0=None, n0=None, ff=1.0, K=16):
    """SKL algorithm according to the article of A. Levy and M. Lindenbaum: 'Sequential Karhunen-Loeve basis extraction and its application to images.'"""
    n = data.shape[1]
    if U0 == None:
        U = numpy.zeros((data.shape[0], data.shape[1]))
        U[0, 0] = 1
        D = numpy.array([0])
        mu = numpy.mean(data, 1)
    else:
        if not(n0):
            n0 = n
        mu1 = numpy.mean(data, axis=1)
        data = data - mu1
        data = numpy.hstack((data, numpy.sqrt(n * n0 / (n + n0)) * (mu0 - mu1)))
        mu = (ff * n0 * mu0 + n * mu1) / (n + ff * n0)
        n = n + ff * n0
        if D0.shape[0] == 1:
            D = numpy.array([D0])
        else:
            D = numpy.diag(D0)
        data_proj = numpy.dot(U0.T, data)
        data_res = data - numpy.dot(U0, data_proj)
        q, r = numpy.linalg.qr(data_res)
        Q = numpy.hstack((U0, q))
        Rtop = numpy.hstack((ff * D, data_proj))
        Rbot = numpy.hstack((numpy.zeros((data.shape[1], D.shape[0])), numpy.dot(q.T, data_res)))
        R = numpy.vstack((Rtop, Rbot))
        U, D, V = numpy.linalg.svd(R)
        keep = numpy.arange(0, numpy.min((K, D.shape[0])))
        D = D[keep]
        U = U[:, keep]
        U = numpy.dot(Q, U)
    return U, D, mu, n


def maxLikelihood(mat, U, D, mu):
    """Get the maximum likelihood and the index in the matrix."""
    if U != None:
        diff = mat - mu
        coef = numpy.dot(numpy.mat(U).T, diff)
        diff = diff - numpy.dot(U, coef)
        invLikelihoods = numpy.sum(numpy.power(diff, 2), axis=0)
        idx = numpy.argmin(invLikelihoods)
    else:
        idx = 0
    return idx

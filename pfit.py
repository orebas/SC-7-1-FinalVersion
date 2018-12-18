import numpy
import ptest

def pFit(xsubk, fsubk):
    n =len(xsubk)
    vandermonde = numpy.zeros((n,n))
    for i in range(n):
        for k  in range(n):
            vandermonde[i][k] = xsubk[i] ** k
    asubk = numpy.linalg.solve(vandermonde,fsubk)
    cond = numpy.linalg.cond(vandermonde)
    return (asubk,cond)


#below fits a cubic spline to input data.  We assume the xs are sorted in increasing order.
# we return them as part of our data as we expect them to be passed around as part of the
#spline definition.
#some notes to make it readable:
# the polynomials are sum of a(i,k)(x-x_k)^i for i=0 to 3
# a(i,k) is stored in coeffs[4*k+i]


def splinefit(xsubk,fsubk):
    A = []
    B = []
    n = len(xsubk)
    coeffs = [0.0] * (4 * (n-1))  #there's a cubic poly for each interval, i.e. for all but the last x
    for k in range(0, n-1):
        row = [0.0] * (4 * (n-1))
        row[4*k+0] = 1
        A.append(row)
        B.append(fsubk[k])
    for k in range(0, n-1):
        t = xsubk[k+1] - xsubk[k]
        row = [0.0] * (4 * (n - 1))
        row[4 * k + 0] = 1
        row[4 * k + 1] = t
        row[4 * k + 2] = t**2
        row[4 * k + 3] = t**3
        A.append(row)
        B.append(fsubk[k+1])
    for k in range(0,n-2):
        t = xsubk[k + 1] - xsubk[k]
        row = [0.0] * (4 * (n - 1))
        row[4*k+1] = 1.0
        row[4*k+2] = 2.0*t
        row[4 * k + 3] = 3.0 * t * t
        row[4 * (k+1) + 1] = -1
        A.append(row)
        B.append(0)
    for k in range(0,n-2):
        t = xsubk[k + 1] - xsubk[k]
        row = [0.0] * (4 * (n - 1))
        row[4*k+2] = 2.0
        row[4*k+3] = 6.0*t
        row[4 * (k+1) + 2] = -2
        A.append(row)
        B.append(0)

    t = xsubk[k + 1] - xsubk[k]
    row = [0.0] * (4 * (n - 1))
    row[4 * 0 + 2] = 2.0
    A.append(row)
    B.append(0)

    t = xsubk[-1] - xsubk[-2]
    row = [0.0] * (4 * (n - 1))
    row[4 * (n-2) + 2] = 2.0
    row[4 * (n - 2) + 3] = 6.0 * t
    A.append(row)
    B.append(0)
    A = numpy.array(A)
    B = numpy.array(B)
    #print(A)
    #print(B)

    coeffs = numpy.linalg.solve(A,B)
    cond = numpy.linalg.cond(A)
    #print (coeffs)
    return (coeffs,cond)

def rfit(xs,fs,l):
    n = len(xs)
    A = []
    B = []
    for i in range(n):
        row = [0] * n
        for k in range(n):
            row[k] = ptest.phi(xs[i] - xs[k], l)
        A.append(row)
        B.append(fs[i])
    w = numpy.linalg.solve(A,B)
    return(w, numpy.linalg.cond(A))

#dummy comment
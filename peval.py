import numpy as np
import ptest

def pEval(x, asubk):
    x = x * 1.0
    f = asubk[-1] * 1.0
    k = len(asubk) - 2
    while (k >= 0):
        f = f * x + asubk[k] * 1.0
        k = k - 1
    return f


#below evaluates a cubuc spline.
#coeffs should have 4 coeffs per entry in x_k except for the last one.
#we are putting constant term on the left, cubic term on the right
# we use horners method again
def splineEval(x,xs,coeffs):
    i = np.searchsorted(xs,x)-1
    if(x <= xs[0]):
        i =0
    if(x > xs[-1]):
        i = len(xs)-2
    t = x - xs[i]
    return ((coeffs[4*i+3] *t + coeffs[4*i+2]) * t + coeffs[4*i+1]) * t + coeffs[4*i]


def rEval(x,xs,ws,l):
    sum =0
    for i in range(len(xs)):
        sum += ws[i] * ptest.phi(x - xs[i], l)
    return sum

#dummy comment
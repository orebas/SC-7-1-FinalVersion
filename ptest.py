import peval
import pfit
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import findiff

def gauss(x):
    return math.exp(x*x*-0.5)

def phi(x,l):
    return math.exp(x * x /(-2 * l * l))

def ptest():
    #test Horner's method implementation vs Pythons
    maxabserr = 0
    maxpolyval=0
    for i in range(10):
        d =    random.randint(2,30)
        xs = [0]*(d+1)
        for j in range(d+1):
            xs[j] = random.random() * 10 - 5
        #print(xs)
        for j in range(-10,10):
            t1 = peval.pEval(j * 1.0, xs)
            ys = np.flipud(xs)
            t2 = np.polyval(ys,j * 1.0)
            maxabserr = max(maxabserr, abs(t1-t2))
            maxpolyval = max(maxpolyval,t1)
            #print (t1,t2)
    print (maxabserr)
    print (maxpolyval)

#test interpolation
    avgfiterr = []
    avgcond = []
    runs = 40
    dims = range(3,40)
    for i in dims:
        avgfiterr.append(0)
        avgcond.append(0)
        for r in range(0, runs):
            d = i
            xs = [0] * (d + 1)
            fs = [0] *(d+1)
            checks = [0] * (d+1)
            for j in range(d + 1):
                xs[j] = random.random() * 10 - 5
                fs[j] = random.random() * 10 - 5
            sol = pfit.pFit(xs, fs)
            avgcond[-1] += sol[1] / (runs*1.0)
            for j in range(d+1):
                checks[j] = peval.pEval(xs[j], sol[0])
                avgfiterr[-1] +=  abs(checks[j] - fs[j]) / (runs * 1.0 * (d+1))
    plt.plot(dims, avgfiterr)
    plt.yscale("log")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Mean absolute error of interp for random polynomials (log)")
    plt.title("Interpolation error of high-degree polynomial interpolation")
    plt.show()
    plt.clf()
    plt.plot(dims, avgcond)
    plt.xlabel("Polynomial degree")
    plt.ylabel("Mean condition number of random polynomials (log)")
    plt.title("Condition number of Vandermonde matrix for high-d polynomial ")
    plt.yscale("log")
    plt.show()
    #print (avgfiterr)
    #print (avgcond)

# interpolate gauss function at various d and plot
#first, we allow the interp function to use all of [-2,2]
    plt.clf()
    dset = [2,3,4,10,20]
    sol =[]
    solcond = []
    plotpoints = 400
    for d in dset:
        fs = [0] * (d + 1)
        xs = np.linspace(-2,2,d+1)
        for i in range(d+1):
            fs[i] = (  gauss(xs[i]))
        soldata = pfit.pFit(xs, fs)
        sol.append(soldata[0])
        solcond.append(soldata[1])
    xplot = np.linspace(-2,2,plotpoints)
    res = []
    for x in xplot:
        res.append(gauss(x))
    plt.plot(xplot,res,label ="Gauss" )
    for i in range(len(dset)):
        res = []
        for x in xplot:
            res.append(peval.pEval(x, sol[i]))
        plt.plot(xplot,res,label=dset[i])
    plt.xlabel("X")
    plt.ylabel("Interpolated Y")
    plt.title("Polynomial interp performance, in-sample, for various degrees")
    plt.legend()
    plt.show()

#next, we only give the interp functions data from [-1,1]
    plt.clf()
    dset = [2,3,4,5,6,10,15]
    sol =[]
    solcond = []
    plotpoints = 400
    for d in dset:
        fs = [0] * (d + 1)
        xs = np.linspace(-1,1,d+1)
        for i in range(d+1):
            fs[i] = (  gauss(xs[i]))
        soldata = pfit.pFit(xs, fs)
        sol.append(soldata[0])
        solcond.append(soldata[1])
    xplot = np.linspace(-2,2,plotpoints)
    res = []
    for x in xplot:
        res.append(gauss(x))
    plt.plot(xplot,res,label ="Gauss" )
    for i in range(len(dset)):
        res = []
        for x in xplot:
            res.append(peval.pEval(x, sol[i]))
        plt.plot(xplot,res,label=dset[i])
    plt.legend()

    plt.xlabel("X")
    plt.ylabel("Interpolated Y")
    plt.title("Polynomial interp performance, out-of-sample, for various degrees")
    plt.show()




def stest():
    xs = []
    fs = []
    for i in np.linspace(-2,2,21):
        xs.append(i)
        fs.append(gauss(i))
    over = 0.2
    plotx=[]
    ploty=[]
    eval = [0] * len(xs)
    coeffs = pfit.splinefit(xs, fs)[0]
    #plt.plot(xs,fs)
    #plt.show()
    xs = np.array(xs)
    fs = np.array(fs)
    def i(x):
        return peval.splineEval(x, xs, coeffs)
    domain =  np.linspace(np.min(xs) - over,np.max(xs) + over,903)
    dx = domain[1] - domain[0]
    for x in domain:
       plotx.append(x)
       ploty.append(i(x))
    plotx = np.array(plotx)
    ploty = np.array(ploty)
    plt.plot(xs,fs,label = "Gauss")
    plt.plot(plotx,ploty, label = "Spline interp"),
    plt.title("Accuracy of spline interpolation for F(x)")
    plt.legend()
    plt.show()
    plt.clf()
    d_dx = findiff.FinDiff(0,dx,1)
    d_dx2 = findiff.FinDiff(0, dx, 2)
    d_dx3 = findiff.FinDiff(0, dx, 3)
    d_dx4 = findiff.FinDiff(0, dx, 4)
    fp = d_dx(ploty)
    fp2 = d_dx2(ploty)
    fp3 = d_dx3(ploty)
    fp4 = d_dx4(ploty)

    # plt.plot(plotx,ploty, label = "base")
    # plt.plot(plotx,fp, label = 1)
    # plt.plot(plotx,fp2, label = 2)
    # plt.plot(plotx,fp3, label = 3)
    # #plt.plot(plotx,fp4, label = 4)
    # plt.legend()
    # plt.show()

#first, verify all interpolation conditions:
    maxerr = 0
    for k in range(len(xs)):
        maxerr = max(maxerr, abs(i(xs[k]) - fs[k]))
    print ("Max error at knot points is %f" % maxerr)
    print ( "Max absolute value of first derivative is  %f" % abs(max(fp,key=abs))  )
    print ( "Max absolute value of second derivative is  %f" % abs(max(fp2,key=abs))  )
    print ( "Max absolute value of third derivative is  %f" % abs(max(fp3,key=abs))  )
    print("Max absolute value of fourth derivative is  %f" % abs(max(fp4, key=abs)))
    modfp4 = np.zeros_like(fp4)
    for k in range(len(fp4)):
        modfp4[k] = fp4[k]
        if( np.abs(xs - plotx[k] ).min() < dx*1.1 ):
            modfp4[k] = 0
    print("Max absolute value of fourth derivative, when away from knot point is  %f" % abs(max(modfp4, key=abs)))

    plt.plot(plotx,ploty, label = "base")
    plt.plot(plotx,fp, label = 1)
    plt.plot(plotx,fp2, label = 2)
    plt.plot(plotx,fp3, label = 3)
    plt.plot(plotx,modfp4, label = 4)
    plt.title("Derivatives of various orders of interpolating spline for F(x)")
    plt.legend()
    plt.show()

def scondgraph():
    dims = range(3, 40)
    conds = []
    for d in dims:
        xs = np.linspace(-2, 2, d)
        ys = [gauss(x) for x in xs]
        conds.append (pfit.splinefit(xs, ys)[1])
    plt.plot(dims,conds)
    plt.title("Condition number of spline matrix as a function of d")
    plt.ylabel("Condition number")
    plt.xlabel("d")
    plt.show()

def sgaussplots():
    dims = (3,4,5,6,7,10,20,40,80,160,320)
    maxerr=[]
    plotpts = 403
    plotx = np.linspace(-1,1,plotpts)
    truey = [gauss(x) for x in plotx]
    #plt.plot(plotx,truey,label = "true")
    for d in dims:
        xs = np.linspace(-1, 1, d)
        ys = [gauss(x) for x in xs]
        coeffs = pfit.splinefit(xs, ys)[0]
        ploty = [peval.splineEval(x, xs, coeffs) - gauss(x) for x in plotx]
        plt.plot(plotx,ploty,label = d)
        plt.ylabel("Error vs true function")
        plt.title("Spline interpolation error for various knot point counts")
        #plt.scatter(xs,ys)
        maxerr.append(max ([abs(x) for x in ploty]) )
    plt.legend()
    plt.show()
    plt.plot(dims,maxerr)
    plt.xlabel("Knot point count")
    plt.ylabel("Max absolute error of spline interp")
    plt.title("Log-log plot of max error vs number of knots")
    plt.yscale("log")
    plt.xscale("log")
    plt.show()

def rtest():
    dims = (4,8,16,32,64)
    ppts = 403
    l = 0.8
    domain = np.linspace(-2, 2, ppts)
    truey = [gauss(x) for x in domain]
    plt.plot(domain, truey, label="true")

    for d in dims:
        xs = np.linspace(-2,2,d)
        fs = [gauss(x) for x in xs]
        sol = pfit.rfit(xs, fs, l)
        w = sol[0]
        def i(x):
            return peval.rEval(x, xs, w, l)
        err = [abs(i(x) - gauss(x)) for x in xs]
        print("Max absolute err is %f" % max(err))
        testy = [i(x) for x in domain]
        plt.plot(domain,testy,label = "%i interp" %d)
    plt.legend()
    plt.show()

def rcondgraph():
    dims = range(3,20)
    conds = []
    l = 0.8
    for d in dims:
        xs =  np.linspace(-2,2,d)
        fs= [gauss(x) for x in xs]
        conds.append(pfit.rfit(xs, fs, l)[1])
    print(conds)
    plt.plot(dims,conds)
    plt.yscale("log")
    #plt.xscale("log")
    plt.title("Condition number as a function of d, rdf interp")
    plt.xlabel("D")
    plt.ylabel("Condition number (log)")
    plt.show()
    d = 10
    lconds=[]
    xs = np.linspace(-2, 2, d)
    fs = [gauss(x) for x in xs]
    lvalues = np.linspace(0.1,2,20)
    for lt in lvalues:
        lconds.append(pfit.rfit(xs, fs, lt)[1])
    plt.plot(lvalues,lconds)
    plt.yscale("log")
    plt.title("Condition number as a function of L, rdf interp")
    plt.xlabel("L")
    plt.ylabel("Condition number (log)")
    plt.show()

def plotrdf():
    dlset = ((4,0.2),
        (4,0.8),
        (4,2),
        (10,0.2),
        (10,0.8),
        (10,2)
    )
    ppts = 403
    pdomain = np.linspace(-2, 2, ppts)
    truefunc = [gauss(x) for x in pdomain]
    plt.plot(pdomain,truefunc,label = "True function")
    for (d,l) in dlset:
        xs = np.linspace(-2,2,d)
        fs = [gauss(x) for x in xs]
        ws = pfit.rfit(xs, fs, l)[0]
        def int(x):
            return peval.rEval(x, xs, ws, l)
        intfunc = [int(x) for x in pdomain]
        plt.plot(pdomain,intfunc,label = "D = %i, L = %f"%(d,l))
    plt.legend()
    plt.show()

if __name__== "__main__":
    #ptest()
    #stest()
    #scondgraph()
    sgaussplots()
    #rtest()
    #rcondgraph()
    #plotrdf()
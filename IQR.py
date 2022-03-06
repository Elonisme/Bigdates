# coding: utf-8
import scipy.stats

class IQR:
    def Calculate_IQR(selfs):
        Q1 = scipy.stats.norm(0,1).ppf(0.25)
        Q3 = scipy.stats.norm(0,1).ppf(0.75)
        Upperfence = scipy.stats.norm(0,1).cdf(Q3+1.5*(Q3-Q1))
        Lowerfence = scipy.stats.norm(0,1).cdf(Q1-1.5*(Q3-Q1))
        probUL = round(Upperfence-Lowerfence,4)
        probOutLiers = 1-probUL
        print(u'Q1-μ= %.4f\u03C3,Q3-μ=%.4f'%(Q1,Q3))
        print(u'IQR = Q3-Q1= %.4f\u03C3'%(Q3-Q1))
        print(u'Q3+1.5xIQR-μ=%.4f\u03C3'%(Q3+1.5*(Q3-Q1)))
        print(u'Q1-1.5xIQR-μ=%.4fu03C3'%(Q1-1.5*Q3-Q1))
        print(u'P(Q1-1.5xIPR<x<Q3+1.5xIQR)=%.4f'%(probUL))
        print(u'在上下限之外的概率=%.4f%%'%(100*probOutLiers))

if __name__=='__main__':
    I = IQR()
    I.Calculate_IQR()

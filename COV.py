# coding: utf-8
import numpy as np

class COV:
    def Calculate_COV(selfs):
        Adult_group = np.array([177, 169, 171, 171, 173, 175, 170, 173, 169, 172, 173, 175,
        179, 176, 166, 170, 167, 171, 171 ,169])
        Children_group = np.array([72,76,72,70,69,76,77,72,68,74,72,70,71,73,
        75,71,72,72,71,67])
        print(u'成人组标准差：%.2f  幼儿园标准差： %.2f'
        %(np.std(Adult_group,ddof=1),np.std(Children_group,ddof=1)))
        print(u'成人组均差：%.2f  幼儿园均差： %.2f'
        %(np.mean(Adult_group),np.mean(Children_group)))
        print(u'成人组离散系数：%.4f  幼儿园离散系数： %.4f'
        %(np.std(Adult_group,ddof=1)/np.mean(Adult_group),np.std(Children_group,ddof=1)/np.mean(Children_group)))

if __name__ == '__main__':
    C = COV()
    
    C.Calculate_COV()
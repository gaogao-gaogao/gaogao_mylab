# -*- coding：utf-8 -*-

import pandas as pd
import numpy as np


data = pd.read_csv(r'D:\pythonProject\Boosting\AdaBoost\17.csv')



def DeleteTargetNan(exdata, targetstr):

    if exdata[targetstr].isnull().any():
     
        loc = exdata[targetstr][data[targetstr].isnull().values == True].index.tolist()
       
        exdata = exdata.drop(loc)
  
    exdata = exdata.fillna(exdata.mean())
  
    targetnum = exdata[targetstr].copy()
    del exdata[targetstr]
    exdata[targetstr] = targetnum
    return exdata




def Shanchu(exdata, aiduan=['NO']):
    for ai in aiduan:
        if ai in exdata.keys():
            del exdata[ai]
    return exdata



def Digit(eadata):
   
    for jj in eadata:
        try:
            eadata[jj].values[0] + 1
        except TypeError:
            
            numlist = list(set(list(eadata[jj].values)))
            zhuan = [numlist.index(jj) for jj in eadata[jj].values]
            eadata[jj] = zhuan
    return eadata




first = DeleteTargetNan(data, 'mv')
two = Shanchu(first)
third = Digit(two)


def fenge(exdata, k=10, per=[0.6, 0.4]):
 
    lent = len(exdata)
    alist = np.arange(lent)
    np.random.shuffle(alist)


    xunlian_sign = int(lent * per[0])

    xunlian = np.random.choice(alist, xunlian_sign, replace=False)

    
    yuce = np.array([i for i in alist if i not in xunlian])

 
    save_dict = {}
    for jj in range(k):
        save_dict[jj] = {}
        length = len(xunlian)
      
        yuzhi = int(length / k)
        yan = np.random.choice(xunlian, yuzhi, replace=False)
        tt = np.array([i for i in xunlian if i not in yan])
        save_dict[jj]['train'] = exdata[tt]
        save_dict[jj]['test'] = exdata[yan]

    return save_dict, exdata[yuce]

deeer = fenge(third.values)


dt_data = deeer[0]

predict_data = deeer[1]

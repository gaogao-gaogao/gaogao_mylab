# -*- coding：utf-8 -*-

import AdaBoost_Data as data


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
import numpy as np


from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] 
mpl.rcParams['axes.unicode_minus'] = False  
import matplotlib.pyplot as plt


cengs = [20, 30, 50]


models = [100, 300, 500]



def Train(data, modelcount, censhu, yanzhgdata):
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=censhu),
                              n_estimators=modelcount, learning_rate=0.8)

    model.fit(data[:, :-1], data[:, -1])
   
    train_out = model.predict(data[:, :-1])
  
    train_mse = mse(data[:, -1], train_out)

 
    add_yan = model.predict(yanzhgdata[:, :-1])
    
    add_mse = mse(yanzhgdata[:, -1], add_yan)
    print(train_mse, add_mse)
    return train_mse, add_mse


def Zuhe(datadict, tre=models, tezhen=cengs):
  
    savedict = {}
   
    sacelist = {}
    for t in tre:
        for te in tezhen:
            print(t, te)
            sumlist = []
          
            ordelist = sorted(list(datadict.keys()))
            for jj in ordelist:
                xun, ya = Train(datadict[jj]['train'], t, te, datadict[jj]['test'])
              
                sumlist.append(ya)
            sacelist['%s-%s' % (t, te)] = sumlist
            savedict['%s-%s' % (t, te)] = np.mean(np.array(sumlist))

   
    zuixao = sorted(savedict.items(), key=lambda fu: fu[1])[0][0]
   
    xiao = sacelist[zuixao].index(min(sacelist[zuixao]))
    return zuixao, xiao, sacelist


def duibi(exdict, you):
    plt.figure(figsize=(11, 7))
    for ii in exdict:
        plt.plot(list(range(len(exdict[ii]))), exdict[ii], \
                 label='%s,%d-Fold MSE Mean Value:%.3f' % (ii, len(exdict[ii]), np.mean(np.array(exdict[ii]))), lw=2)
    plt.legend()
    plt.title('MSE comparison curves of different parameters[Optimal:%s]' % you)
    plt.savefig(r'D:\pythonProject\Boosting\AdaBoost\adaboost_VV_VH_NDWI.jpg')
    return 'The comparison of different methods is complete'


def recspre(exstr, predata, datadict, zhe, count=100):
    tree, te = exstr.split('-')
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=int(te)),
                              n_estimators=int(tree), learning_rate=0.8)
    model.fit(datadict[zhe]['train'][:, :-1], datadict[zhe]['train'][:, -1])


    yucede = model.predict(predata[:, :-1])



    np.save('./pre_VV_VH_NDWI.npy', yucede)
    np.save('./true_VV_VH_NDWI.npy', predata[:, -1])



    zongleng = np.arange(len(yucede))
    randomnum = np.random.choice(zongleng, count, replace=False)


    yucede_se = list(np.array(yucede)[randomnum])

    yuce_re = list(np.array(predata[:, -1])[randomnum])


    plt.figure(figsize=(17, 9))
    plt.subplot(2, 1, 1)
    plt.plot(list(range(len(yucede_se))), yucede_se, 'r--', label='Predicted', lw=2)
    plt.scatter(list(range(len(yuce_re))), yuce_re, c='b', marker='.', label='Measured', lw=2)
    plt.xlim(-1, count + 1)
    plt.legend()
    plt.title('Measured and Predicted value comparison[Maximum tree:%d]' % int(tree))

    plt.subplot(2, 1, 2)
    plt.plot(list(range(len(yucede_se))), np.array(yuce_re) - np.array(yucede_se), 'k--', marker='s', label='Measured-Predicted', lw=2)
    plt.legend()
    plt.title('The relative error between Predicted and Measured value')

    plt.savefig(r'D:\pythonProject\Boosting\AdaBoost\duibi_VV_VH_NDWI.jpg')
    return 'The prediction is complete'



if __name__ == "__main__":
    zijian, zhehsu, xulie = Zuhe(data.dt_data)

    duibi(xulie, zijian)
    recspre(zijian, data.predict_data, data.dt_data, zhehsu)
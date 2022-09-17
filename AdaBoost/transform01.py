import csv
import numpy as np


true = np.load('./true_VV_VH_NDWI.npy')  # 367
predicts = np.load('./pre_VV_VH_NDWI.npy')  # 367

true = np.around(true, 2)
predicts = np.around(predicts, 2)

true = np.reshape(true, (len(true), 1))
predicts = np.reshape(predicts, (len(true), 1))


with open('./true_VV_VH_NDWI.csv', 'w', encoding='utf-8') as file_obj:
    writer = csv.writer(file_obj)
    for p in true:
        writer.writerow(p)
        
        
with open('./predict_VV_VH_NDWI.csv', 'w', encoding='utf-8') as file_obj:
    writer = csv.writer(file_obj)
    for p in predicts:
        writer.writerow(p)
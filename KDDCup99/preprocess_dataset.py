import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pickle


encoder = preprocessing.LabelEncoder()
# NSLKDD test set preprocess
kdd = pd.read_csv("./data/KDDTest+.csv")

kdd.drop(["service","is_host_login","num_outbound_cmds", "num_learners"],axis=1,inplace=True)

for c in kdd.columns:
    if str(kdd[c].dtype) == 'object': 
        kdd[c] = encoder.fit_transform(kdd[c])

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
kdd[['dst_bytes','src_bytes']] = scaler.fit_transform(kdd[['dst_bytes','src_bytes']])

with open('./data/KDDTest+.pth','wb') as f:
    pickle.dump(kdd,f) 
    
# NSLKDD train set preprocess
kdd = pd.read_csv("./data/KDDTrain+.csv")

kdd.drop(["service","is_host_login","num_outbound_cmds", "num_learners"],axis=1,inplace=True)

for c in kdd.columns:
    if str(kdd[c].dtype) == 'object': 
        kdd[c] = encoder.fit_transform(kdd[c])

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
kdd[['dst_bytes','src_bytes']] = scaler.fit_transform(kdd[['dst_bytes','src_bytes']])

with open('./data/KDDTrain+.pth','wb') as f:
    pickle.dump(kdd,f) 
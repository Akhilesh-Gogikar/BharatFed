#  -*-coding:utf8 -*-
import json
import requests
import time
import uuid
import datetime
import time

'''
feature_data = dict()
feature_data['FlowBytesSent'] = 1203
feature_data['FlowSentRate'] = 16725.68622516372
feature_data['FlowBytesReceived'] = 8305
feature_data['FlowReceivedRate'] = 115473.92277637965
feature_data['PacketLengthVariance'] = 327826.87603305787
feature_data['PacketLengthStandardDeviation'] = 572.5616788024307
feature_data['PacketLengthMean'] = 432.181818181818
feature_data['PacketLengthMedian'] = 63.0
feature_data['PacketLengthMode'] = 54
feature_data['PacketLengthSkewFromMedian'] = 1.934368812215996
feature_data['PacketLengthSkewFromMode'] = 0.66050843460712
feature_data['PacketLengthCoefficientofVariation'] = 1.3248166737119769
feature_data['PacketTimeVariance'] = 0.00035945262665289246
feature_data['PacketTimeStandardDeviation'] = 0.018959235919543082
feature_data['PacketTimeMean'] = 0.048442727272727255
feature_data['PacketTimeMedian'] = 0.049658
feature_data['PacketTimeMode'] = 0.0
feature_data['PacketTimeSkewFromMedian'] = -0.19229773801485806
feature_data['PacketTimeSkewFromMode'] = 2.5550991336519404
feature_data['PacketTimeCoefficientofVariation'] = 0.3913742472178465
feature_data['ResponseTimeTimeVariance'] = 9.582381066666662e-05
feature_data['ResponseTimeTimeStandardDeviation'] = 0.009788963717711218
feature_data['ResponseTimeTimeMean'] = 0.01143
feature_data['ResponseTimeTimeMedian'] = 0.012999
feature_data['ResponseTimeTimeMode'] = 5e-05
feature_data['ResponseTimeTimeSkewFromMedian'] = -0.4808476296100
feature_data['ResponseTimeTimeSkewFromMode'] = 1.1625336785557914
feature_data['ResponseTimeTimeCoefficientofVariation'] = 0.85642727189074
'''
import pandas as pd

df = pd.read_csv('cyber_security_eval_data.csv', index_col='id')

y_true = df.loc[:, df.columns == 'y'].values

print(y_true)

records = df.loc[:, df.columns != 'y'].to_json(orient='records')

records = json.loads(records)
#print(records)

response_list = []

for record in records:
    #print(record)
    feature_data = record
    url1 = "http://172.19.0.3:8059/federation/1.0/inference"

    request_data_tmp = {
        "head": {
            "serviceId": "cyber_security_inference",
        },
        "body": {
            "featureData": feature_data,
            "sendToRemoteFeatureData": {
                "party_id": 10000
                }
        }
    }

    request_data = {
        "head": {
            "serviceId": "cyber_security_inference",
        },
        "body": {
            "featureData": feature_data,
        }
    }
    #print(request_data)
    headers = {"Content-Type": "application/json"}
    response = requests.post(url1, json=request_data_tmp, headers=headers)
    #print(response.text)

    response_list.append(json.loads(response.text)["data"]["score"])


y_preds = [1 if x > 0.5 else 0 for x in response_list]

print(y_preds)

from sklearn.metrics import accuracy_score, roc_auc_score

acc_score = accuracy_score(y_true, y_preds, normalize=True, sample_weight=None)

auc = roc_auc_score(y_true, y_preds)
print("accuracy: {}".format(acc_score))

print("AUC: {}".format(auc))
'''
print(response_list)

print("url:", url1)
print("request:\n", request_data_tmp)
print()
print("response:\n", response.text)
print()
#time.sleep(0.1)
'''

'''
    "head": {
        "serviceId": "cyber_security_inference.py",
    },
    "body": {
          "featureData": feature_data,
    }
'''

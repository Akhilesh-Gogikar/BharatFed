import pandas as pd

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import preprocessing
from diffprivlib.models import LinearRegression, LogisticRegression
from ml_utils import train_save_model, return_predictions

date_list = []
amount_list = []

amount_cat_list = []

with open('data_response.txt') as json_file:
    data = json.load(json_file)

    print(data['body'][0]['fiObjects'][0]['Profile']['Holders']['Holder']['pan'])

    x = train_save_model(data)

    print(x)

    stat = return_predictions(data)

    print(stat)

    #start_date = data['body'][0]['fiObjects'][0]['Transactions']['start_date']
    #end_date = data['body'][0]['fiObjects'][0]['Transactions']['start_date']
'''
    for p in data['body'][0]['fiObjects'][0]['Transactions']['Transaction']:
        if p['type'] == 'CREDIT':
            date_list.append(p['valueDate'])
            amount_list.append(p['amount'])
            amount_cat_list.append(p['narration'])
        else:
            date_list.append(p['valueDate'])
            amount_list.append(-1 * p['amount'])
            amount_cat_list.append(p['narration'])
            
'''

'''
data = {'dates':date_list,'category': amount_cat_list, 'amount': amount_list}
df = pd.DataFrame(data=data)
le = preprocessing.LabelEncoder()
min_max_scaler = preprocessing.QuantileTransformer(random_state=0)
robust_scaler = preprocessing.MaxAbsScaler()#.RobustScaler()
df['category'] = min_max_scaler.fit_transform(le.fit_transform(df['category']).reshape(-1,1))
df['amount'] = robust_scaler.fit_transform(df['amount'].values.reshape(-1,1))
df['dates'] = df['dates'].astype('datetime64[ns]')

cols = []

for lag in range(1, 14 + 1):
    col = 'lag_{}'.format(lag)
    #df['cat_'+col] = df['category'].shift(lag)
    #cols.append('cat_'+col)
    df['amt_'+col] = df['amount'].shift(lag)
    cols.append('amt_'+col)

df.dropna(inplace=True)

split = int(len(df) * 0.60)

print(df.head())

print(df.describe())

train = df.iloc[:split].copy()

test = df.iloc[split:].copy()

model_c = LinearRegression()#LogisticRegression()#

model_c.fit(train[cols], train[['category', 'amount']])
#model_a.fit(train[cols], train[['amount']])
print("R2 score for epsilon=%.2f: %.2f" % (model_c.epsilon, model_c.score(test[cols], test[['category', 'amount']])))

for entry in test[cols].iterrows():
    preds = model_c.predict(entry[1].to_numpy().reshape(1, -1))
    #print(preds)
    print(le.inverse_transform(min_max_scaler.inverse_transform(preds[0][0].reshape(1, -1)).astype(int)), robust_scaler.inverse_transform(preds[0][1].reshape(1, -1)))

epsilons = np.logspace(-2, 2, 50)

accuracy = list()

for epsilon in epsilons:
    clf = LinearRegression(epsilon=epsilon)
    clf.fit(train[cols], train[['category', 'amount']])

    accuracy.append(clf.score(test[cols], test[['category', 'amount']]))

plt.semilogx(epsilons, accuracy)
plt.title("Differentially private Naive Bayes accuracy")
plt.xlabel("epsilon")
plt.ylabel("Accuracy")
plt.show()

model = LinearRegression()
model.fit(df[cols], df[['category', 'amount']])
#model_a.fit(train[cols], train[['amount']])
print("R2 score for epsilon=%.2f: %.2f" % (model.epsilon, model.score(df[cols], df[['category', 'amount']])))
'''
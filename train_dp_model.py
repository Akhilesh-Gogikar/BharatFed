import pandas as pd

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import preprocessing
from diffprivlib.models import LinearRegression, LogisticRegression
from ml_utils import train_save_model, return_predictions
from find_spare_balance import find_spare_balance

date_list = []
amount_list = []

amount_cat_list = []

def intervaled_cumsum(a, trigger_val=1, start_val = 0, invalid_specifier=-1):
    out = np.ones(a.size,dtype=int)
    idx = np.flatnonzero(a==trigger_val)
    if len(idx)==0:
        return np.full(a.size,invalid_specifier)
    else:
        out[idx[0]] = -idx[0] + 1
        out[0] = start_val
        out[idx[1:]] = idx[:-1] - idx[1:] + 1
        np.cumsum(out, out=out)
        out[:idx[0]] = invalid_specifier
        return out

with open('data_response.txt') as json_file:
    data = json.load(json_file)

    print(data['body'][0]['fiObjects'][0]['Profile']['Holders']['Holder']['pan'])

    x = train_save_model(data)

    print(x)

    stat = return_predictions(data)

    balance = int(stat[1])

    preds = json.loads(stat[0])

    #print(preds['prediction'])

    spare_bal = find_spare_balance(preds['prediction'], balance)

    print("The user has {} rupees in his account which the prediction model forecasts won't be spent".format(spare_bal))







'''
    cr_list = set()

    deb_list = set()

    #start_date = data['body'][0]['fiObjects'][0]['Transactions']['start_date']
    #end_date = data['body'][0]['fiObjects'][0]['Transactions']['start_date']
    for p in data['body'][0]['fiObjects'][0]['Transactions']['Transaction']:
        if p['type'] == 'CREDIT':
            date_list.append(p['valueDate'])
            amount_list.append(p['amount'])
            amount_cat_list.append(p['narration'])
            cr_list.add(p['narration'])

        else:
            date_list.append(p['valueDate'])
            amount_list.append(p['amount'])
            amount_cat_list.append(p['narration'])
            deb_list.add(p['narration'])


data = {'dates':date_list,'category': amount_cat_list, 'amount': amount_list}
df = pd.DataFrame(data=data)
le = preprocessing.LabelEncoder()

cat_scaler = preprocessing.QuantileTransformer(random_state=0)
robust_scaler = preprocessing.MaxAbsScaler()#.RobustScaler()
df['category'] = le.fit_transform(df['category']).reshape(-1,1)

#print(le.classes_)

cols = []

for cls in le.classes_:
    #print(cls, le.transform([cls])[0])
    cols.append('Recency_'+cls)
    df['Recency_'+cls] = intervaled_cumsum(df.category.values, trigger_val=le.transform([cls])[0], start_val=0, invalid_specifier=365)/365


df[cols] = df[cols].shift(1)

df['amount'] = robust_scaler.fit_transform(df['amount'].values.reshape(-1,1))
df['dates'] = df['dates'].astype('datetime64[ns]')

for lag in range(1, 14 + 1):
    col = 'lag_{}'.format(lag)

    df['amt_'+col] = df['amount'].shift(lag)

    cols.append('amt_'+col)

df.dropna(inplace=True)

split = int(len(df) * 0.60)

#print(df.head())

#print(df.describe())

train = df.iloc[:split].copy()

test = df.iloc[split:].copy()

list1 = ['category', 'amount']
list1.extend(cols)
#print(df[list1].values)
norm = np.linalg.norm(df[list1].values, ord=1, axis=1).max()

#norm=None



y_bounds = (np.min(df[['category', 'amount']], axis=0), np.max(df[['category', 'amount']], axis=0))

#y_bounds = None

#print(norm)

bounds = ([0 for i in range(len(cols))],[1 for i in range(len(cols))])
x_bounds=bounds

model_c = LinearRegression(data_norm=norm, bounds_X=x_bounds, bounds_y=y_bounds)

model_c.fit(train[cols], train[['category', 'amount']])

print("R2 score for epsilon=%.2f: %.2f" % (model_c.epsilon, model_c.score(test[cols], test[['category', 'amount']])))

#print(cr_list, deb_list)

for entry in test[cols].iterrows():
    preds = model_c.predict(entry[1].to_numpy().reshape(1, -1))
    cat = le.inverse_transform(preds[0][0].reshape(1, -1).astype(int))
    if cat[0] in cr_list:
        amt = abs(robust_scaler.inverse_transform(preds[0][1].reshape(1, -1)))
    else:
        amt = -1*abs(robust_scaler.inverse_transform(preds[0][1].reshape(1, -1)))

    print(cat, amt)

epsilons = np.logspace(-2, 2, 50)

accuracy = list()

for epsilon in epsilons:
    clf = LinearRegression(data_norm=norm, bounds_X=bounds, bounds_y=(0,1), epsilon=epsilon)
    clf.fit(train[cols], train[['category', 'amount']])

    accuracy.append(clf.score(test[cols], test[['category', 'amount']]))

plt.semilogx(epsilons, accuracy)
plt.title("Differentially private model accuracy")
plt.xlabel("epsilon")
plt.ylabel("Accuracy")
plt.show()

model = LinearRegression(data_norm=norm, bounds_X=bounds, bounds_y=(0,1))
model.fit(df[cols], df[['category', 'amount']])
#model_a.fit(train[cols], train[['amount']])
print("R2 score for epsilon=%.2f: %.2f" % (model.epsilon, model.score(df[cols], df[['category', 'amount']])))
'''
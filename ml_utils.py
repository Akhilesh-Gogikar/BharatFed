import pandas as pd
import json
from sklearn import preprocessing
from diffprivlib.models import LinearRegression, GaussianNB, standard_scaler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from datetime import datetime, timedelta

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


def train_save_model(data):
    '''
    Input: the json response of the data request from the fiu api
    :return:
    True if the model is successfully saved else False
    '''

    date_list = []
    amount_list = []

    amount_cat_list = []

    bal_list = []

    current_balance = 0

    for p in data['body'][0]['fiObjects'][0]['Transactions']['Transaction']:
        if p['type'] == 'CREDIT':
            date_list.append(p['valueDate'])
            amount_list.append(p['amount'])
            amount_cat_list.append(p['narration'])
            bal_list.append(p['currentBalance'])
            current_balance = p['currentBalance']
        else:
            date_list.append(p['valueDate'])
            amount_list.append(-1 * p['amount'])
            amount_cat_list.append(p['narration'])
            bal_list.append(p['currentBalance'])
            current_balance = p['currentBalance']

    m_data = {'dates': date_list, 'category': amount_cat_list, 'amount': amount_list, 'currentBalance': bal_list}
    df = pd.DataFrame(data=m_data)
    le = preprocessing.LabelEncoder()
    cat_scaler = preprocessing.QuantileTransformer(random_state=0)
    amt_scaler = standard_scaler.StandardScaler()
    bal_scaler = standard_scaler.StandardScaler()
    df['category'] = cat_scaler.fit_transform(le.fit_transform(df['category']).reshape(-1, 1))
    df['amount'] = amt_scaler.fit_transform(df['amount'].values.reshape(-1, 1))
    df['currentBalance'] = bal_scaler.fit_transform(df['currentBalance'].values.reshape(-1, 1))
    df['dates'] = df['dates'].astype('datetime64[ns]')

    cols = []

    for lag in range(1, 14+1):
        col = 'lag_{}'.format(lag)
        df['cat_'+col] = df['category'].shift(lag)
        cols.append('cat_'+col)
        df['bal_' + col] = df['currentBalance'].shift(lag)
        cols.append('bal_' + col)

    df.dropna(inplace=True)

    model = LinearRegression()#GaussianNB()#LinearRegression()
    model.fit(df[cols], df[['category', 'amount']])
    print("R2 score for epsilon=%.2f: %.2f" % (model.epsilon, model.score(df[cols], df[['category', 'amount']])))

    out = False

    models = {
        'model': model,
        'category_scaler': cat_scaler,
        'amount_scaler': amt_scaler,
        'balance_scaler': bal_scaler,
        'category_encoder': le
    }

    modelfile = '{}.sav'.format(data['body'][0]['fiObjects'][0]['Profile']['Holders']['Holder']['pan'])

    try:
        pickle.dump(models, open(modelfile, 'wb'))
        out = True
    except Exception as e:
        print(e)

    return out


def return_predictions(data, days=30):
    '''

    :param data: The json of the request data available to make prediction must have length greater than 14 entries
    :param days: The number of days for which prediction has to be made
    :return: The json of the expense category and amount prediction for each date since the last date in data
    '''

    date_list = []
    amount_list = []

    amount_cat_list = []

    bal_list = []

    cr_list = set()

    deb_list = set()

    current_balance = 0

    for p in data['body'][0]['fiObjects'][0]['Transactions']['Transaction']:
        if p['type'] == 'CREDIT':
            date_list.append(p['valueDate'])
            amount_list.append(p['amount'])
            amount_cat_list.append(p['narration'])
            bal_list.append(p['currentBalance'])
            cr_list.add(p['narration'])
            current_balance = p['currentBalance']
        else:
            date_list.append(p['valueDate'])
            amount_list.append(p['amount'])
            amount_cat_list.append(p['narration'])
            bal_list.append(p['currentBalance'])
            deb_list.add(p['narration'])
            current_balance = p['currentBalance']

    return_dict = {'prediction': [{'dates': None, 'category': None, 'amount': None, 'type': None}]}

    initial_balance = current_balance

    if len(date_list) < 14:
        return return_dict
    else:

        modelfile = '{}.sav'.format(data['body'][0]['fiObjects'][0]['Profile']['Holders']['Holder']['pan'])

        try:
            models = pickle.load(open(modelfile, 'rb'))
        except Exception as e:
            print(e)
            return None

    data = {'dates': date_list, 'category': amount_cat_list, 'amount': amount_list, 'currentBalance': bal_list}
    df = pd.DataFrame(data=data)
    le = models['category_encoder']
    cat_scaler = models['category_scaler']
    amt_scaler = models['amount_scaler']
    bal_scaler = models['balance_scaler']
    df['category'] = cat_scaler.transform(le.transform(df['category']).reshape(-1, 1))
    df['amount'] = amt_scaler.transform(df['amount'].values.reshape(-1, 1))
    df['currentBalance'] = bal_scaler.transform(df['currentBalance'].values.reshape(-1, 1))
    df['dates'] = df['dates'].astype('datetime64[ns]')

    model = models['model']

    cols = []

    for lag in range(1, 14 + 1):
        col = 'lag_{}'.format(lag)
        df['cat_'+col] = df['category'].shift(lag)
        cols.append('cat_'+col)
        df['bal_' + col] = df['currentBalance'].shift(lag)
        cols.append('bal_' + col)

    df.dropna(inplace=True)

    entry = df.iloc[-1].copy()

    for day in range(days):
        if day == 0:

            next_date = entry['dates'].date() + timedelta(day+1)
            #print(next_date)
            preds = model.predict(entry[cols].to_numpy().reshape(1, -1))
            expense = le.inverse_transform(cat_scaler.inverse_transform(preds[0][0].reshape(1, -1)).astype(int))
            amount = abs(amt_scaler.inverse_transform(preds[0][1].reshape(1, -1)))

            current_balance = int(current_balance)

            if expense[0] in deb_list:
                if amount[0][0] < current_balance:
                    current_balance = current_balance - amount[0][0]
                    preds[0][1] = bal_scaler.transform(current_balance.reshape(1, -1))
                else:
                    amount[0][0] = current_balance
                    current_balance = current_balance - amount[0][0]
                    preds[0][1] = bal_scaler.transform(current_balance.reshape(1, -1))
            else:
                current_balance = current_balance + amount[0][0]
                preds[0][1] = bal_scaler.transform(current_balance.reshape(1, -1))

            type_t = 'CREDIT' if expense[0] in cr_list else 'DEBIT'

            next_entry = {'dates': next_date}

            for i in range(1,27):
                try:
                    next_entry[cols[i-1]] = entry[cols[i+1]]
                except Exception as e:
                    continue

            next_entry[cols[26]] = preds[0][0]
            next_entry[cols[27]] = preds[0][1]

            return_dict['prediction'][0]['dates'] = str(next_date)
            return_dict['prediction'][0]['category'] = str(expense[0])
            return_dict['prediction'][0]['amount'] = int(amount[0][0])
            return_dict['prediction'][0]['type'] = type_t

            entry = pd.Series(next_entry)




        else:

            next_date = entry['dates'] + timedelta(1)

            preds = model.predict(entry[cols].to_numpy().reshape(1, -1))
            expense = le.inverse_transform(cat_scaler.inverse_transform(preds[0][0].reshape(1, -1)).astype(int))
            amount = abs(amt_scaler.inverse_transform(preds[0][1].reshape(1, -1)))

            current_balance = int(current_balance)

            if expense[0] in deb_list:
                if amount[0][0] < current_balance:
                    current_balance = current_balance - amount[0][0]
                    preds[0][1] = bal_scaler.transform(current_balance.reshape(1, -1))
                else:
                    amount[0][0] = current_balance
                    current_balance = current_balance - amount[0][0]
                    preds[0][1] = bal_scaler.transform(current_balance.reshape(1, -1))
            else:
                current_balance = current_balance + amount[0][0]
                preds[0][1] = bal_scaler.transform(current_balance.reshape(1, -1))

            type_t = 'CREDIT' if expense[0] in cr_list else 'DEBIT'

            next_entry = {'dates': next_date}

            return_dict['prediction'].append(
                {'dates': str(next_date), 'category': str(expense[0]), 'amount': int(amount[0][0]),
                 'type': str(type_t)})

            for i in range(1, 27):
                try:
                    next_entry[cols[i - 1]] = entry[cols[i + 1]]
                except Exception as e:
                    continue

            next_entry[cols[26]] = preds[0][0]
            next_entry[cols[27]] = preds[0][1]

            entry = pd.Series(next_entry)

    return json.dumps(return_dict), initial_balance






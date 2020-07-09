import pandas as pd
import json
from sklearn import preprocessing
from diffprivlib.models import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from datetime import datetime, timedelta


def train_save_model(data):
    '''
    Input: the json response of the data request from the fiu api
    :return:
    True if the model is successfully saved else False
    '''

    date_list = []
    amount_list = []

    amount_cat_list = []

    for p in data['body'][0]['fiObjects'][0]['Transactions']['Transaction']:
        if p['type'] == 'CREDIT':
            date_list.append(p['valueDate'])
            amount_list.append(p['amount'])
            amount_cat_list.append(p['narration'])
        else:
            date_list.append(p['valueDate'])
            amount_list.append(-1 * p['amount'])
            amount_cat_list.append(p['narration'])

    m_data = {'dates': date_list, 'category': amount_cat_list, 'amount': amount_list}
    df = pd.DataFrame(data=m_data)
    le = preprocessing.LabelEncoder()
    cat_scaler = preprocessing.QuantileTransformer(random_state=0)
    amt_scaler = preprocessing.MaxAbsScaler()
    df['category'] = cat_scaler.fit_transform(le.fit_transform(df['category']).reshape(-1, 1))
    df['amount'] = amt_scaler.fit_transform(df['amount'].values.reshape(-1, 1))
    df['dates'] = df['dates'].astype('datetime64[ns]')

    cols = []

    for lag in range(1, 14+1):
        col = 'lag_{}'.format(lag)
        df['cat_'+col] = df['category'].shift(lag)
        cols.append('cat_'+col)
        df['amt_' + col] = df['amount'].shift(lag)
        cols.append('amt_' + col)

    df.dropna(inplace=True)

    model = LinearRegression()
    model.fit(df[cols], df[['category', 'amount']])
    print("R2 score for epsilon=%.2f: %.2f" % (model.epsilon, model.score(df[cols], df[['category', 'amount']])))

    out = False

    models = {
        'model': model,
        'category_scaler': cat_scaler,
        'amount_scaler': amt_scaler,
        'category_encoder': le
    }

    #print(data['body'])

    modelfile = '{}.sav'.format(data['body'][0]['fiObjects'][0]['Profile']['Holders']['Holder']['pan'])

    try:
        pickle.dump(models, open(modelfile, 'wb'))
        out = True
    except Exception as e:
        print(e)

    return out


def return_predictions(data, days=60):
    '''

    :param data: The json of the request data available to make prediction must have length greater than 14 entries
    :param days: The number of days for which prediction has to be made
    :return: The json of the expense category and amount prediction for each date since the last date in data
    '''

    date_list = []
    amount_list = []

    amount_cat_list = []

    for p in data['body'][0]['fiObjects'][0]['Transactions']['Transaction']:
        if p['type'] == 'CREDIT':
            date_list.append(p['valueDate'])
            amount_list.append(p['amount'])
            amount_cat_list.append(p['narration'])
        else:
            date_list.append(p['valueDate'])
            amount_list.append(-1 * p['amount'])
            amount_cat_list.append(p['narration'])

    return_dict = {'prediction': [{'date': None, 'category': None, 'amount': None}]}

    if len(date_list) < 14:
        return return_dict
    else:

        modelfile = '{}.sav'.format(data['body'][0]['fiObjects'][0]['Profile']['Holders']['Holder']['pan'])

        try:
            models = pickle.load(open(modelfile, 'rb'))
        except Exception as e:
            print(e)

    data = {'dates': date_list, 'category': amount_cat_list, 'amount': amount_list}
    df = pd.DataFrame(data=data)
    le = models['category_encoder']
    cat_scaler = models['category_scaler']
    amt_scaler = models['amount_scaler']
    df['category'] = cat_scaler.fit_transform(le.fit_transform(df['category']).reshape(-1, 1))
    df['amount'] = amt_scaler.fit_transform(df['amount'].values.reshape(-1, 1))
    df['dates'] = df['dates'].astype('datetime64[ns]')

    model = models['model']

    cols = []

    for lag in range(1, 14 + 1):
        col = 'lag_{}'.format(lag)
        df['cat_'+col] = df['category'].shift(lag)
        cols.append('cat_'+col)
        df['amt_' + col] = df['amount'].shift(lag)
        cols.append('amt_' + col)

    df.dropna(inplace=True)

    entry = df.iloc[-1].copy()

    for day in range(days):
        if day == 0:

            next_date = entry['dates'].date() + timedelta(day+1)
            #print(next_date)
            preds = model.predict(entry[cols].to_numpy().reshape(1, -1))
            expense = le.inverse_transform(cat_scaler.inverse_transform(preds[0][0].reshape(1, -1)).astype(int))
            amount = amt_scaler.inverse_transform(preds[0][1].reshape(1, -1))

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

            entry = pd.Series(next_entry)


        else:
            next_date = entry['dates'] + timedelta(day + 1)

            preds = model.predict(entry[cols].to_numpy().reshape(1, -1))
            expense = le.inverse_transform(cat_scaler.inverse_transform(preds[0][0].reshape(1, -1)).astype(int))
            amount = amt_scaler.inverse_transform(preds[0][1].reshape(1, -1))

            return_dict['prediction'].append({'dates': str(next_date), 'category': str(expense[0]), 'amount': int(amount[0][0])})

            next_entry = {'dates': next_date}

            for i in range(1, 27):
                try:
                    next_entry[cols[i - 1]] = entry[cols[i + 1]]
                except Exception as e:
                    continue

            next_entry[cols[26]] = preds[0][0]
            next_entry[cols[27]] = preds[0][1]

            entry = pd.Series(next_entry)

    return json.dumps(return_dict)






import pandas as pd
from datetime import datetime, timedelta
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def convert_fig_to_html(fig):
    """ Convert Matplotlib figure 'fig' into a <img> tag for HTML use using base64 encoding. """
    import urllib
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from io import BytesIO
    import base64

    sio = BytesIO()

    fig.savefig(sio, format="png")

    html = """<img src="data:image/png;base64,{}"/>""".format(base64.encodebytes(sio.getvalue()).decode())

    return html

def earnings_per_week(data):
    week_data = data.groupby(['type', 'category']).resample('W-Mon', on='date').sum().reset_index().sort_values(
        by='date')  #
    week_data = week_data.loc[week_data['type'] == 'CREDIT']
    week_data = week_data.drop(columns=['type'])
    week_data = week_data.set_index('date')
    pivot_df = week_data.pivot(columns='category', values='amount')
    pivot_df['week'] = pivot_df.index.week
    pivot_df['month'] = pivot_df.index.month_name()
    pivot_df['year'] = pivot_df.index.to_series().dt.strftime('%Y')
    pivot_df.index = pivot_df.index.date
    print(pivot_df)
    # pivot_df.index = pivot_df.index.week
    # ax = pivot_df.plot.bar(stacked=True)

    # fig_html = convert_fig_to_html(plt.figure())

    return pivot_df.to_json(orient='index')

def earnings_per_month(data):
    mon_data = data.groupby(['type', 'category']).resample('M', on='date').sum().reset_index().sort_values(
        by='date')  #
    mon_data = mon_data.loc[mon_data['type'] == 'CREDIT']
    mon_data = mon_data.drop(columns=['type'])
    mon_data = mon_data.set_index('date')
    pivot_df = mon_data.pivot(columns='category', values='amount')
    pivot_df['month'] = pivot_df.index.month_name()
    pivot_df['year'] = pivot_df.index.to_series().dt.strftime('%Y')
    pivot_df.index = pivot_df.index.week
    print(pivot_df)
    # for mon in pivot_df['month']:
    # ax = pivot_df[pivot_df['month']==mon].plot.bar(stacked=True, title=mon)

    # fig_html = convert_fig_to_html(plt.figure())

    return pivot_df.to_json(orient='index')

def earnings_per_year(data):
    year_data = data.groupby(['type', 'category']).resample('Y', on='date').sum().reset_index().sort_values(
        by='date')  #
    year_data = year_data.loc[year_data['type'] == 'CREDIT']
    year_data = year_data.drop(columns=['type'])
    year_data = year_data.set_index('date')
    pivot_df = year_data.pivot(columns='category', values='amount')
    pivot_df['year'] = pivot_df.index.to_series().dt.strftime('%Y')

    pivot_df.index = pivot_df.index.month_name()
    #for year in pivot_df['year']:
        #ax = pivot_df[pivot_df['year'] == year].plot.bar(stacked=True, title=year)
    return data.to_json(orient='index')

def earnings_by_category(data):
    data = data.groupby(['type', 'category']).resample('W-Mon', on='date').sum().reset_index().sort_values(by='date')

    data = data.loc[data['type'] == 'CREDIT']

    data = data.set_index('date')

    data['week'] = data.index.week
    data['month'] = data.index.month_name()
    data['year'] = data.index.to_series().dt.strftime('%Y')

    data = data.groupby(['year', 'month', 'week', 'category']).sum()

    #data = data.groupby(['category']).sum()

    #plot = data.plot.pie(y='amount', figsize=(5, 5))

    #fig_html = convert_fig_to_html(plt.figure())

    return data.to_json(orient='index')

def expense_by_category(data):
    data = data.groupby(['type', 'category']).resample('W-Mon', on='date').sum().reset_index().sort_values(by='date')

    data = data.loc[data['type'] == 'DEBIT']

    data = data.set_index('date')

    data['week'] = data.index.week
    data['month'] = data.index.month_name()
    data['year'] = data.index.to_series().dt.strftime('%Y')

    data = data.groupby(['year', 'month', 'week', 'category']).sum()
    #data = data.groupby(['category','week']).sum()
    print(data)

    #plot = data.plot.pie(y='amount', figsize=(5, 5), subplots=True)

    #fig_html = convert_fig_to_html(plt.figure())

    return data.to_json(orient='index')

def expense_per_week(data):
    week_data = data.groupby(['type', 'category']).resample('D', on='date').sum().reset_index().sort_values(by='date')#
    week_data = week_data.loc[week_data['type'] == 'DEBIT']
    week_data = week_data.drop(columns=['type'])
    week_data = week_data.set_index('date')
    pivot_df = week_data.pivot(columns='category', values='amount')
    pivot_df['week'] = pivot_df.index.week
    pivot_df['month'] = pivot_df.index.month_name()
    pivot_df['year'] = pivot_df.index.to_series().dt.strftime('%Y')
    pivot_df.index = pivot_df.index.date
    print(pivot_df)
    #pivot_df.index = pivot_df.index.week
    #ax = pivot_df.plot.bar(stacked=True)

    #fig_html = convert_fig_to_html(plt.figure())

    return pivot_df.to_json(orient='index')

def expense_per_month(data):
    mon_data = data.groupby(['type', 'category']).resample('W-Mon', on='date').sum().reset_index().sort_values(
        by='date')  #
    mon_data = mon_data.loc[mon_data['type'] == 'DEBIT']
    mon_data = mon_data.drop(columns=['type'])
    mon_data = mon_data.set_index('date')
    pivot_df = mon_data.pivot(columns='category', values='amount')
    pivot_df['month'] = pivot_df.index.month_name()
    pivot_df['year'] = pivot_df.index.to_series().dt.strftime('%Y')
    pivot_df.index = pivot_df.index.week
    print(pivot_df)
    #for mon in pivot_df['month']:
        #ax = pivot_df[pivot_df['month']==mon].plot.bar(stacked=True, title=mon)

    #fig_html = convert_fig_to_html(plt.figure())

    return pivot_df.to_json(orient='index')

def expense_per_year(data):
    year_data = data.groupby(['type', 'category']).resample('M', on='date').sum().reset_index().sort_values(
        by='date')  #
    print(year_data)
    year_data = year_data.loc[year_data['type'] == 'DEBIT']
    year_data = year_data.drop(columns=['type'])
    year_data = year_data.set_index('date')
    pivot_df = year_data.pivot(columns='category', values='amount')
    pivot_df['year'] = pivot_df.index.to_series().dt.strftime('%Y')

    pivot_df.index = pivot_df.index.month_name()
    for year in pivot_df['year']:
        ax = pivot_df[pivot_df['year']==year].plot.bar(stacked=True, title=year)
    return ax

def preds_to_df(data):
    '''

    :param data: Predictions from the model
    :return: df for charting purposes
    '''

    date_list = []

    type_list = []

    amount_list = []

    amount_cat_list = []

    for p in data['prediction']:
        if p['type'] == 'CREDIT':
            date_list.append(p['dates'])
            type_list.append(p['type'])
            amount_list.append(p['amount'])
            amount_cat_list.append(p['category'])

        else:
            date_list.append(p['dates'])
            type_list.append(p['type'])
            amount_list.append(p['amount'])
            amount_cat_list.append(p['category'])

    data_d = {'date': date_list, 'category': amount_cat_list, 'type': type_list, 'amount': amount_list}
    df = pd.DataFrame(data=data_d)
    df['date'] = pd.to_datetime(df['date'])

    return df


def json_to_df(data):

    date_list = []

    type_list = []

    amount_list = []

    amount_cat_list = []

    # start_date = data['body'][0]['fiObjects'][0]['Transactions']['start_date']
    # end_date = data['body'][0]['fiObjects'][0]['Transactions']['start_date']
    for p in data['body'][0]['fiObjects'][0]['Transactions']['Transaction']:
        if p['type'] == 'CREDIT':
            date_list.append(p['valueDate'])
            type_list.append(p['type'])
            amount_list.append(p['amount'])
            amount_cat_list.append(p['narration'])

        else:
            date_list.append(p['valueDate'])
            type_list.append(p['type'])
            amount_list.append(p['amount'])
            amount_cat_list.append(p['narration'])

    data_d = {'date': date_list, 'category': amount_cat_list, 'type': type_list, 'amount': amount_list}
    df = pd.DataFrame(data=data_d)
    df['date'] = pd.to_datetime(df['date'])

    return df

#Testing the utils below

if __name__ == '__main__' :

    with open('data_response_bharatfed99@finvu_1yr.txt') as json_file:
        data = json.load(json_file)

        df = json_to_df(data)

        from ml_utils import return_predictions

       #stat = return_predictions(data)

        #balance = int(stat[1])

        #preds = json.loads(stat[0])

        #df = preds_to_df(preds)

    ax = expense_by_category(df)

    plt.show()

    print(ax)

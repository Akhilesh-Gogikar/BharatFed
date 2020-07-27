import json
from sklearn.model_selection import train_test_split
from ml_model_utils import DP_LSTM_model
from find_spare_balance import find_spare_balance

with open('data_response_bharatfed99@finvu_1yr.txt') as json_file:
    data = json.load(json_file)

    DP_module = DP_LSTM_model()

    #print(data['body'][0]['fiObjects'][0]['Profile']['Holders']['Holder']['pan'])

    #pan = data['body'][0]['fiObjects'][0]['Profile']['Holders']['Holder']['pan']

    X, Y, scaler_dict, y_cols, creds, debs = DP_module.train_data_prep(data)

    #print(creds, debs)

    #print(y_cols)

    #split = int(len(X) * 0.75)

    #train_X, test_X, train_y, test_y = train_test_split(X.values, Y.values, test_size=0.15, shuffle=True)

    #train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    #test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    #print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    #model = DP_module.train_model(pan, train_X, train_y, test_X, test_y)

    #rmse = DP_module.prediction_test(model, scaler_dict, test_X, test_y, y_cols)

    #print(rmse)

    #out = DP_module.save_train_model(scaler_dict, data)

    stat = DP_module.return_predictions(data, creds, debs)

    preds = json.loads(stat)

    print(preds)

    balance = int(preds['Balance'])

    print(preds['Predictions'])

    spare_bal = find_spare_balance(preds['Predictions'], balance)

    print("The user has {} rupees in his account which the prediction model forecasts won't be spent".format(spare_bal))
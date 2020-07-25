import json

import pickle
from datetime import datetime, timedelta
import pandas as pd

from sklearn import preprocessing
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot

import numpy as np

from sklearn.metrics import mean_squared_error

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

import tensorflow as tf
import tensorflow.compat.v1 as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from absl import logging
import collections

from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.dp_query import gaussian_query

class DP_LSTM_model():

    def __init__(self):
        self.description = "This is class wrapper for all the functions pertaining to the ML model for BharatFed"
        self.GradientDescentOptimizer = tf.compat.v1.train.AdamOptimizer  # .GradientDescentOptimizer
        self.DPGradientDescentGaussianOptimizer_NEW = self.make_gaussian_optimizer_class(self.GradientDescentOptimizer)

    def make_optimizer_class(self, cls):
        """Constructs a DP optimizer class from an existing one."""
        parent_code = tf.compat.v1.train.Optimizer.compute_gradients.__code__
        child_code = cls.compute_gradients.__code__
        GATE_OP = tf.compat.v1.train.Optimizer.GATE_OP  # pylint: disable=invalid-name
        if child_code is not parent_code:
            logging.warning(
                'WARNING: Calling make_optimizer_class() on class %s that overrides '
                'method compute_gradients(). Check to ensure that '
                'make_optimizer_class() does not interfere with overridden version.',
                cls.__name__)

        class DPOptimizerClass(cls):
            """Differentially private subclass of given class cls."""

            _GlobalState = collections.namedtuple(
                '_GlobalState', ['l2_norm_clip', 'stddev'])

            def __init__(
                    self,
                    dp_sum_query,
                    num_microbatches=None,
                    unroll_microbatches=False,
                    *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
                    **kwargs):
                """Initialize the DPOptimizerClass.

                Args:
                  dp_sum_query: DPQuery object, specifying differential privacy
                    mechanism to use.
                  num_microbatches: How many microbatches into which the minibatch is
                    split. If None, will default to the size of the minibatch, and
                    per-example gradients will be computed.
                  unroll_microbatches: If true, processes microbatches within a Python
                    loop instead of a tf.while_loop. Can be used if using a tf.while_loop
                    raises an exception.
                """
                super(DPOptimizerClass, self).__init__(*args, **kwargs)
                self._dp_sum_query = dp_sum_query
                self._num_microbatches = num_microbatches
                self._global_state = self._dp_sum_query.initial_global_state()
                # TODO(b/122613513): Set unroll_microbatches=True to avoid this bug.
                # Beware: When num_microbatches is large (>100), enabling this parameter
                # may cause an OOM error.
                self._unroll_microbatches = unroll_microbatches

            def compute_gradients(self,
                                  loss,
                                  var_list,
                                  gate_gradients=GATE_OP,
                                  aggregation_method=None,
                                  colocate_gradients_with_ops=False,
                                  grad_loss=None,
                                  gradient_tape=None,
                                  curr_noise_mult=0,
                                  curr_norm_clip=1):

                self._dp_sum_query = gaussian_query.GaussianSumQuery(curr_norm_clip,
                                                                     curr_norm_clip * curr_noise_mult)
                self._global_state = self._dp_sum_query.make_global_state(curr_norm_clip,
                                                                          curr_norm_clip * curr_noise_mult)

                # TF is running in Eager mode, check we received a vanilla tape.
                if not gradient_tape:
                    raise ValueError('When in Eager mode, a tape needs to be passed.')

                vector_loss = loss()
                if self._num_microbatches is None:
                    self._num_microbatches = tf.shape(input=vector_loss)[0]
                sample_state = self._dp_sum_query.initial_sample_state(var_list)
                microbatches_losses = tf.reshape(vector_loss, [self._num_microbatches, -1])
                sample_params = (self._dp_sum_query.derive_sample_params(self._global_state))

                def process_microbatch(i, sample_state):
                    """Process one microbatch (record) with privacy helper."""
                    microbatch_loss = tf.reduce_mean(input_tensor=tf.gather(microbatches_losses, [i]))
                    grads = gradient_tape.gradient(microbatch_loss, var_list)
                    sample_state = self._dp_sum_query.accumulate_record(sample_params, sample_state, grads)
                    return sample_state

                for idx in range(self._num_microbatches):
                    sample_state = process_microbatch(idx, sample_state)

                if curr_noise_mult > 0:
                    grad_sums, self._global_state = (self._dp_sum_query.get_noised_result(sample_state, self._global_state))
                else:
                    grad_sums = sample_state

                def normalize(v):
                    return v / tf.cast(self._num_microbatches, tf.float32)

                final_grads = tf.nest.map_structure(normalize, grad_sums)
                grads_and_vars = final_grads  # list(zip(final_grads, var_list))

                return grads_and_vars

        return DPOptimizerClass


    def make_gaussian_optimizer_class(self, cls):
        """Constructs a DP optimizer with Gaussian averaging of updates."""

        class DPGaussianOptimizerClass(self.make_optimizer_class(cls)):
            """DP subclass of given class cls using Gaussian averaging."""

            def __init__(
                    self,
                    l2_norm_clip,
                    noise_multiplier,
                    num_microbatches=None,
                    ledger=None,
                    unroll_microbatches=False,
                    *args,  # pylint: disable=keyword-arg-before-vararg
                    **kwargs):
                dp_sum_query = gaussian_query.GaussianSumQuery(
                    l2_norm_clip, l2_norm_clip * noise_multiplier)

                if ledger:
                    dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query,
                                                                  ledger=ledger)

                super(DPGaussianOptimizerClass, self).__init__(
                    dp_sum_query,
                    num_microbatches,
                    unroll_microbatches,
                    *args,
                    **kwargs)

            @property
            def ledger(self):
                return self._dp_sum_query.ledger

        return DPGaussianOptimizerClass

    def encode_and_bind(self, original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], prefix=feature_to_encode)
        dum_cols = dummies.columns
        # print(dum_cols)
        res = pd.concat([original_dataframe, dummies], axis=1)
        res = res.drop([feature_to_encode], axis=1)
        return (res, dum_cols)


    def train_data_prep(self, data):
        '''
        Input: the json response of the data request from the fiu api
        :return:
        True if the model is successfully saved else False
        '''

        date_list = []
        amount_list = []

        amount_cat_list = []

        bal_list = []

        cr_list = set()

        deb_list = set()

        type_list = []

        current_balance = 0

        for p in data['body'][0]['fiObjects'][0]['Transactions']['Transaction']:
            if p['type'] == 'CREDIT':
                date_list.append(p['valueDate'])
                amount_list.append(p['amount'])
                type_list.append(p['type'])
                amount_cat_list.append(p['narration'])
                cr_list.add(p['narration'])
                bal_list.append(p['currentBalance'])
                current_balance = p['currentBalance']
            else:
                date_list.append(p['valueDate'])
                amount_list.append(p['amount'])
                type_list.append(p['type'])
                amount_cat_list.append(p['narration'])
                deb_list.add(p['narration'])
                bal_list.append(p['currentBalance'])
                current_balance = p['currentBalance']

        m_data = {'dates': date_list, 'category': amount_cat_list, 'amount': amount_list, 'currentBalance': bal_list, 'type': type_list}
        df = pd.DataFrame(data=m_data)
        mon_scaler = preprocessing.MinMaxScaler()
        amt_scaler = preprocessing.MinMaxScaler()
        day_scaler = preprocessing.MinMaxScaler()

        df['dates'] = df['dates'].astype('datetime64[ns]')

        score = df.groupby(['type', 'category']).resample('M', on='dates').sum().reset_index().sort_values(
            by='dates')

        score = score.groupby(['type', 'category']).mean().reset_index()

        income_sum = score.loc[score['type'] == 'CREDIT'].sum()['amount']

        salary_ratio = float(score.loc[score['category']=='SALARY']['amount']/income_sum)

        expense_income_ratio = score.loc[score['category'].isin(['gym','food','beauty','entertainment'])]['amount'].sum()/income_sum

        desc_purchase_ratio = score.loc[score['category'].isin(['outing','travel','purchase'])]['amount'].sum()/income_sum

        loan_income_ratio = float(score.loc[score['category']=='loan']['amount']/income_sum)

        bill_income_ratio = score.loc[score['category'].isin(['RENT', 'medical', 'recharge'])]['amount'].sum()/income_sum

        fin_score = ((salary_ratio/(expense_income_ratio+desc_purchase_ratio+loan_income_ratio+bill_income_ratio))+(0.001/(expense_income_ratio*desc_purchase_ratio*loan_income_ratio*bill_income_ratio)))

        fin_score = int(1000*(1 / (1 + np.exp(-fin_score))))


        #Write the avg. Monthly income and financial score to the profile
        print(fin_score, income_sum)


        df['amount'] = amt_scaler.fit_transform(df['amount'].values.reshape(-1, 1))

        df['month'] = mon_scaler.fit_transform(df['dates'].dt.strftime('%m').values.reshape(-1, 1))
        df['day'] = day_scaler.fit_transform(df['dates'].dt.strftime('%d').values.reshape(-1, 1))

        cols = []

        # print(df['category'])

        for lag in range(1, 2):
            col = 'lag_{}'.format(lag)

            df['cat_' + col] = df['category'].shift(lag)

            res = self.encode_and_bind(df, 'cat_' + col)

            df = res[0]
            for i in res[1]:
                cols.append(i)
            print(cols)
            df['amt_' + col] = df['amount'].shift(lag)
            cols.append('amt_' + col)

            df['mon_' + col] = df['month'].shift(lag)
            cols.append('mon_' + col)

            df['day_' + col] = df['day'].shift(lag)
            cols.append('day_' + col)

        res = self.encode_and_bind(df, 'category')

        df = res[0]

        y_cols = list(res[1])

        scalers = {
            'day_scaler': day_scaler,
            'amount_scaler': amt_scaler,
            'month_scaler': mon_scaler,
            'categories': y_cols
        }

        y_cols.append('amount')

        df.dropna(inplace=True)

        return df[cols], df[y_cols], scalers, y_cols, cr_list, deb_list


    def train_model(self, pan, train_X, train_y, test_x, test_y):
        # design network
        shp = train_y[:, :-1].shape

        cls_wgts = {}

        for i in range(shp[1]):
            cls_wgts[i] = (1 / np.count_nonzero(train_y[:, i] == 1)) * shp[0] / shp[1]

        print(cls_wgts)

        input = tf.keras.layers.Input(shape=(train_X.shape[1], train_X.shape[2]))
        x10 = Dense(15, activation='relu')(input)
        x20 = Dense(15, activation='relu')(input)
        x1 = LSTM(50)(x10)
        x1 = Dropout(0.1)(x1)
        x2 = Dense(15, activation='relu')(x20)
        x2 = Dense(15, activation='relu')(x2)
        x2 = tf.keras.layers.Flatten()(x2)
        reg_out = Dense(1, name='reg', activation="sigmoid")(x1)
        cls_out = Dense(15, name='cls', activation="softmax")(x2)

        model1 = tf.keras.Model(input, [reg_out])
        model2 = tf.keras.Model(input, [cls_out])
        model = tf.keras.Model(input, [cls_out, reg_out])

        optimizer = self.DPGradientDescentGaussianOptimizer_NEW(
            l2_norm_clip=1.0,
            noise_multiplier=1.2,
            num_microbatches=250,
            learning_rate=0.1)

        model1.compile(loss={'reg': 'mse'}, optimizer=optimizer,
                       metrics=['MeanSquaredError'])  # GaussianNB()#LinearRegression() categorical_crossentropy
        model2.compile(loss={'cls': 'categorical_crossentropy'}, optimizer=optimizer,
                       metrics=['accuracy'])  # GaussianNB()#LinearRegression() categorical_crossentropy

        print(model1.summary())
        checkpoint_filepath = '{}.h5'.format(pan)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True)

        history = model1.fit(train_X, train_y[:, -1], epochs=10, batch_size=64,
                             validation_data=(test_x, [test_y[:, :-1], test_y[:, -1]]), verbose=2,
                             shuffle=False, callbacks=[model_checkpoint_callback])

        model1.load_weights(checkpoint_filepath)

        # plot history
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

        print(model2.summary())
        checkpoint_filepath = '{}.h5'.format(pan)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True)

        history = model2.fit(train_X, train_y[:, :-1], epochs=100, batch_size=64,
                             validation_data=(test_x, [test_y[:, :-1], test_y[:, -1]]), verbose=2,
                             shuffle=False, callbacks=[model_checkpoint_callback], class_weight=cls_wgts)

        model2.load_weights(checkpoint_filepath)

        # plot history
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

        model.save('{}.h5'.format(pan))

        return model


    def prediction_test(self, model, scaler_dict, test_X, test_y, y_cols):
        # make a prediction
        yhat = model.predict(test_X)

        print(yhat)

        yhat1 = concatenate([yhat[0], yhat[1]], axis=1)

        yhat1 = yhat1.reshape((len(yhat1), 16))

        rmse = sqrt(mean_squared_error(test_y, yhat1))

        print('Test RMSE: %.3f' % rmse)

        expenses = np.argmax(test_y[:, :-1], axis=1).astype(int)
        print(expenses)
        amounts = scaler_dict['amount_scaler'].inverse_transform(test_y[:, -1].reshape(-1, 1))

        p_expenses = np.argmax(yhat1[:, :-1], axis=1).astype(int)
        print(p_expenses)
        p_amounts = scaler_dict['amount_scaler'].inverse_transform(yhat1[:, -1].reshape(-1, 1)).astype(int)

        for i in range(len(expenses)):
            print(y_cols[expenses[i]], amounts[i], y_cols[p_expenses[i]], p_amounts[i])

        return rmse


    def save_train_model(self, scalers, data):
        out = False

        modelfile = '{}.sav'.format(data['body'][0]['fiObjects'][0]['Profile']['Holders']['Holder']['pan'])

        try:
            pickle.dump(scalers, open(modelfile, 'wb'))
            out = True
        except Exception as e:
            print(e)

        return out


    def return_predictions(self, data, creds, debs, period=90):
        return_dict = {}
        last = data['body'][0]['fiObjects'][0]['Transactions']['Transaction'][-1]
        date = last['valueDate']
        amount = last['amount']
        cat = last['narration']
        bal = last['currentBalance']
        return_dict['Balance'] = bal

        date = datetime.strptime(date, "%Y-%m-%d").date()

        modelfile = '{}.sav'.format(data['body'][0]['fiObjects'][0]['Profile']['Holders']['Holder']['pan'])

        try:
            models = pickle.load(open(modelfile, 'rb'))
            model = tf.keras.models.load_model(
                '{}.h5'.format(data['body'][0]['fiObjects'][0]['Profile']['Holders']['Holder']['pan']), compile=False)
        except Exception as e:
            print(e)
            return None
        mon_scaler = models['month_scaler']
        day_scaler = models['day_scaler']
        amt_scaler = models['amount_scaler']
        categories = models['categories']

        return_dict['Predictions'] = []

        for d in range(1, period + 1):
            day = day_scaler.transform(np.array(date.strftime('%d')).reshape(1,-1))
            mon = mon_scaler.transform(np.array(date.strftime('%m')).reshape(1,-1))

            amount = amt_scaler.transform(np.array(amount).reshape(1,-1))

            cats = np.array([1 if x.split('_')[1] == cat else 0 for x in categories[:-1]]).reshape((len(categories[:-1]),1))

            date = date + timedelta(1)

            X = concatenate([cats, amount, mon, day], axis=0)

            X = X.reshape((X.shape[1], 1, X.shape[0]))

            yhat = model.predict(X)

            index = np.argmax(yhat[0], axis=1).astype(int)

            cate = categories[index[0]]

            type_t = 'CREDIT' if cate in creds else 'DEBIT'

            amount = amt_scaler.inverse_transform(yhat[1].reshape(1, -1)).astype(int)

            return_dict['Predictions'].append({'date': str(date), 'category': cate, 'amount': int(amount[0][0]), 'type': type_t})

        return json.dumps(return_dict)

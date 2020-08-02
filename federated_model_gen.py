import tensorflow as tf
import numpy as np
import time


def average_models(model_names):
    models = []

    for model_name in model_names:
        print('###Loading model no: {}###'.format(len(models)))
        time.sleep(5)
        model = tf.keras.models.load_model('Saved_models/{}'.format(model_name), compile=False)
        models.append(model)

    weights = [model.get_weights() for model in models]

    new_weights = list()

    for weights_list_tuple in zip(*weights):
        new_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        )

    model.set_weights(new_weights)

    return model


if __name__ == '__main__':

    name_list = []

    for _ in range(5):
        name_list.append('DSDIU3206D')

    avg_model = average_models(name_list)

    print(avg_model.summary())

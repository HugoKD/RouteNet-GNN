import os
import re
import sys

sys.path.insert(1, '../../')

from data_generator import input_fn
from model import RouteNet
import configparser
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def transformation(x, y, w, features):
    """Apply a transformation over all the samples included in the dataset.
           Args:
               x (dict): predictor variables.
               y (array): target variable.
           Returns:
               x,y: The modified predictor/target variables.
    """
    for k, v in features.items():
        x[k] = tf.math.divide(tf.math.subtract(x[k], v[0]), tf.math.subtract(v[1], v[0]))
    return x, tf.math.log(y), w  # tf.math.log(y)


def r2_score(y_true, y_pred):
    denorm_y_true = tf.math.exp(y_true)
    denorm_y_pred = tf.math.exp(y_pred)
    residual = tf.reduce_sum(tf.square(tf.subtract(denorm_y_true, denorm_y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(denorm_y_true, tf.reduce_mean(denorm_y_true))))
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    return r2


def denorm_MAPE(y_true, y_pred):
    denorm_y_true = tf.math.exp(y_true)
    denorm_y_pred = tf.math.exp(y_pred)
    return tf.abs((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100


min_max_scaling = {
    'traffic': [6677.713, 1.1091119e+09],
    'packets': [0.59622437, 99027.85],
    'packet_size': [[0.59622437, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [99027.85, 1400.0, 1400.0, 1400.0, 1400.0, 1400.0, 1400.0, 1.0]],
    'burst_size': [[0.0, 0.0, 0.0, 0.0], [493838.0, 493838.0, 493838.0, 18758042000.0]],
    'ipg': [[5.63e-07, 1.037e-06, 1.088e-06, 1.094e-06, 1.1e-06, 1.101e-06, 1.107e-06, 1.107e-06, 1.113e-06, 1.114e-06,
             1.12e-06, 1.12e-06, 1.12e-06, 1.12e-06, 1.126e-06, 1.126e-06, 1.126e-06, 1.126e-06, 1.126e-06, 1.126e-06,
             1.126e-06, 1.126e-06, 1.126e-06, 1.127e-06, 1.127e-06, 1.127e-06, 1.127e-06, 1.127e-06, 1.127e-06,
             1.132e-06, 1.133e-06, 1.133e-06, 1.133e-06, 1.133e-06, 1.139e-06, 1.14e-06, 1.145e-06, 1.146e-06,
             1.152e-06, 1.152e-06, 1.152e-06, 1.158e-06, 1.159e-06, 1.164e-06, 1.171e-06, 1.171e-06, 1.172e-06,
             1.177e-06, 1.177e-06, 1.178e-06, 1.178e-06, 1.184e-06, 1.184e-06, 1.197e-06, 1.204e-06, 1.223e-06,
             2.221e-06, 2.227e-06, 2.234e-06, 2.246e-06, 2.247e-06, 2.252e-06, 2.253e-06, 2.253e-06, 2.253e-06,
             2.253e-06, 2.253e-06, 2.259e-06, 2.26e-06, 2.272e-06, 2.272e-06, 2.278e-06, 2.284e-06, 2.285e-06,
             2.292e-06, 2.297e-06, 2.298e-06, 2.304e-06, 2.304e-06, 2.304e-06, 2.304e-06, 2.304e-06, 2.31e-06,
             2.311e-06, 2.317e-06, 2.33e-06, 2.336e-06, 3.372e-06, 3.398e-06, 3.418e-06, 3.43e-06, 3.431e-06, 3.443e-06,
             3.462e-06, 4.556e-06, 3.3824e-05, 3.4573e-05, 3.5667e-05, 3.7824e-05, 4.1472e-05, 5.9827e-05, 9.34716e-11,
             9.924217e-06],
            [1.694008, 1.6940098, 1.6940116, 1.6940132, 1.694015, 1.6940167, 1.6940185, 1.6940203, 1.6940219, 1.6940237,
             1.6940255, 1.6940272, 1.694029, 1.6940306, 1.6940324, 1.6940342, 1.6940359, 1.6940377, 1.6940395,
             1.6940411, 1.6940429, 1.6940446, 1.6940464, 1.6940482, 1.6940498, 1.6940516, 1.6940534, 1.6940551,
             1.6940569, 1.6940585, 1.6940603, 1.6940621, 1.6940638, 1.6940656, 1.6940672, 1.694069, 1.6940708,
             1.6940725, 1.6940743, 1.6940761, 1.6940777, 1.6940795, 1.6940812, 1.694083, 1.6940848, 1.6940864,
             1.6940882, 1.69409, 1.6940917, 1.6940935, 1.6940951, 1.6940969, 1.6940987, 1.6941004, 1.6941022, 1.694104,
             1.6941056, 1.6941074, 1.6941091, 1.6941109, 1.6941127, 1.6941143, 1.6941161, 1.6941178, 1.6941196,
             1.6941214, 1.694123, 1.6941248, 1.6941266, 1.6941283, 1.6941301, 1.6941317, 1.6941335, 1.6941353, 1.694137,
             1.6941388, 1.6941406, 1.6941422, 1.694144, 1.6941457, 1.6941475, 1.6941493, 1.6941509, 1.6941527,
             1.6941545, 1.6941562, 1.694158, 1.6941596, 1.6941614, 1.6941632, 1.6941649, 1.6941667, 1.6941683,
             1.6941701, 1.6941719, 1.6941736, 1.6941754, 1.6941772, 1.6941788, 1.6941806, 1.6941823, 2.3618642e-07,
             1.6940951]],
    'r_capacity': [1., 80.],
    's_capacity': [1., 80.],

}

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.0001,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model = RouteNet()

loss_object = tf.keras.losses.MeanAbsolutePercentageError()

model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False)

best = None
best_mre = float('inf')

ckpt_dir = f'./ckpt_dir'

for f in os.listdir(ckpt_dir):
    if os.path.isfile(os.path.join(ckpt_dir, f)):
        reg = re.findall("\d+\.\d+", f)
        if len(reg) > 0:
            mre = float(reg[0])
            if mre <= best_mre:
                best = f.replace('.index', '')
                best = best.replace('.data', '')
                best = best.replace('-00000-of-00001', '')
                best_mre = mre

best = "68-0.1259-11.0714--614.9732"
print("BEST CHECKOINT FOUND FOR: {}".format(best))

model.load_weights(os.path.join(ckpt_dir, best))

TEST_DATASET = f'../data/1CBR-1400B-2TG/test'

ds_test = input_fn(TEST_DATASET, shuffle=False)
ds_test = ds_test.map(lambda x, y, w: transformation(x, y, w, min_max_scaling))
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

predictions = model.predict(ds_test, verbose=1)
print(type(np.exp(predictions)))
predictions = np.squeeze(np.exp(predictions))

np.save(f'CBR-1400B-2TG-test', predictions)

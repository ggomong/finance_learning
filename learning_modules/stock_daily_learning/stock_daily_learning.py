from datetime import datetime
import sqlite3 as sql
import tensorflow as tf
import numpy as np
import random as rand

# for visual studio debug
#import ptvsd
#ptvsd.enable_attach(secret=None)
#ptvsd.wait_for_attach()

# configuration
#                               O * W + b -> 3 labels for each time series, O[? 7], W[7 3], B[3]
#                               ^ (O: output 7 vec from 7 vec input)
#                               |
#              +-+  +-+        +--+
#              |1|->|2|-> ...  |60| time_step_size = 60
#              +-+  +-+        +--+
#               ^    ^    ...   ^
#               |    |          |
# time series1:[7]  [7]   ...  [7]
# time series2:[7]  [7]   ...  [7]
# time series3:[7]  [7]   ...  [7]
# ...
# time series(250) or time series(750) (batch_size 250 or test_size 1000 - 250)
#      each input size = input_vec_size=lstm_size=7

# configuration variables
input_vec_size = lstm_size = 7
time_step_size = 60
label_size = 3
evaluate_size = 3

batch_size = 12500
test_size = 50000 - batch_size

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, W, B, lstm_size):
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size
    # XT shape: (time_step_size, batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)
    # XR shape: (time_step_size * batch_size, input_vec_size)
    X_split = tf.split(0, time_step_size, XR) # split them to time_step_size (60 arrays)
    # Each array shape: (batch_size, input_vec_size)

    # Make lstm with lstm_size (each input vector size)
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)

    # Get lstm cell output, time_step_size (60) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = tf.nn.rnn(lstm, X_split, dtype=tf.float32)

    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the stat

def read_random_data(conn, codes, dates, keys, size):
    X = list()
    Y = list()

    while (len(keys) < size):
        code = rand.choice(codes)[0]
        date = rand.choice(dates)[0]
        key = (code, date)
        if key in keys:
            continue
        keys.add(key)

        cursor = conn.cursor()
        cursor.execute("SELECT open, high, low, close, volume, hold_foreign, st_purchase_inst FROM stock_daily_series WHERE code = '{0}' AND date >= '{1}' ORDER BY date LIMIT {2}".format(code, date, time_step_size + evaluate_size))
        items = cursor.fetchall();

        X.append(np.array(items[:time_step_size]))

        price = items[-(evaluate_size + 1)][3]
        max = price;
        min = price;

        for item in items[-evaluate_size:]:
            if item[1] < max:
                max = item[1]
            if item[2] > min:
                min = item[2]

        if (min - price) / price < -0.02:
            Y.append((0, 0, 1))
        elif (max - price) / price > 0.04:
            Y.append((1, 0, 0))
        else:
            Y.append((0, 1, 0))
            
    return np.array(X), np.array(Y)

def read_data():
    conn = sql.connect("../../databases/finance_learning.db")
    with conn:
        rand.seed()

        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT code FROM stock_daily_series")
        codes = cursor.fetchall()

        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT date FROM stock_daily_series ORDER BY date")
        dates = cursor.fetchall()[:-(time_step_size + evaluate_size)]

        keys = set()
        trX, trY = read_random_data(conn, codes, dates, keys, batch_size)
        teX, teY = read_random_data(conn, codes, dates, keys, test_size)

    return trX, trY, teX, teY

trX, trY, teX, teY = read_data()

X = tf.placeholder("float", [None, time_step_size, input_vec_size])
Y = tf.placeholder("float", [None, label_size])

# get lstm_size and output 2 labels
W = init_weights([lstm_size, label_size])
B = init_weights([label_size])

py_x, state_size = model(X, W, B, lstm_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                            sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                            Y: teY[test_indices]})))
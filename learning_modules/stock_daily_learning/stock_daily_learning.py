from datetime import datetime
import sqlite3 as sql
import tensorflow as tf
import numpy as np

# configuration
#                                   O * W + b -> 3 labels for each time series, O[? 11], W[11 3], B[3]
#                                   ^ (O: output 11 vec from 11 vec input)
#                                   |
#               +-+   +-+         +--+
#               |1|-->|2|--> ...  |60| time_step_size = 60
#               +-+   +-+         +--+
#                ^     ^     ...    ^
#                |     |            |
# time series1:[11]  [11]    ...  [11]
# time series2:[11]  [11]    ...  [11]
# time series3:[11]  [11]    ...  [11]
#
# each input size = input_vec_size = lstm_size = 11
    
# configuration variables
input_vec_size = lstm_size = 11
time_step_size = 60
label_size = 3
evaluate_size = 3
lstm_depth = 4

epoch_size = 100
batch_size = 20000
train_rate = 25     # 25%


def model(X, W, B, lstm_size) :
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size
    # XT shape: (time_step_size, batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)
    # XR shape: (time_step_size * batch_size, input_vec_size) 
    X_split = tf.split(0, time_step_size, XR) # split them to time_step_size (60 arrays)
    # Each array shape: (batch_size, input_vec_size)

    # Make lstm with lstm_size (each input vector size)
    cell = tf.nn.rnn_cell.GRUCell(lstm_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell = cell, output_keep_prob = 0.5)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * lstm_depth, state_is_tuple = True)

    # Get lstm cell output, time_step_size (60) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = tf.nn.rnn(cell, X_split, dtype=tf.float32)

    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1], W) + B, cell.state_size # State size to initialize the stat


def get_code_dates() :
    conn = sql.connect("../../databases/finance_learning.db")
    with conn :
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT code FROM stock_daily_series")
        codes = cursor.fetchall()

        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT date FROM stock_daily_series ORDER BY date")
        dates = cursor.fetchall()[:-(time_step_size + evaluate_size)]

        code_dates = list()
        for date in dates :
            for code in codes :
                code_dates.append((code[0], date[0]))

        np.random.seed()
        np.random.shuffle(code_dates)

        return code_dates


def read_series_datas(conn, code_dates) :
    X = list()
    Y = list()

    for code_date in code_dates :
        cursor = conn.cursor()
        cursor.execute(
            "SELECT "
                "stock_daily_series.open, "
                "stock_daily_series.high, "
                "stock_daily_series.low, "
                "stock_daily_series.close, "
                "stock_daily_series.volume, "
                "stock_daily_series.hold_foreign, "
                "stock_daily_series.st_purchase_inst, "
                "exchange_daily_series.open, "
                "exchange_daily_series.high, "
                "exchange_daily_series.low, "
                "exchange_daily_series.close "
            "FROM stock_daily_series "
            "JOIN exchange_daily_series "
            "ON stock_daily_series.date = exchange_daily_series.date "
            "WHERE stock_daily_series.code = '{0}' "
                "AND stock_daily_series.date >= '{1}' "
                "AND exchange_daily_series.code = 'FX@KRW' "
            "ORDER BY stock_daily_series.date LIMIT {2}"
            .format(code_date[0], code_date[1], time_step_size + evaluate_size)
            )
        items = cursor.fetchall()

        X.append(np.array(items[:time_step_size]))

        price = items[-(evaluate_size + 1)][3]
        max = items[-evaluate_size][1]
        min = items[-evaluate_size][2]

        for item in items[-evaluate_size + 1:] :
            if max < item[1] :
                max = item[1]
            if item[2] < min :
                min = item[2]

        if (min - price) / price < -0.02 :
            Y.append((0., 0., 1.))
        elif (max - price) / price > 0.04 :
            Y.append((1., 0., 0.))
        else :
            Y.append((0., 1., 0.))

    arrX = np.array(X)
    norX = (arrX - np.mean(arrX, axis = 0)) / np.std(arrX, axis = 0)

    return norX, np.array(Y)


def read_datas(code_dates) :
    conn = sql.connect("../../databases/finance_learning.db")
    with conn :
        X, Y = read_series_datas(conn, code_dates)

    data_size = len(X)
    train_size = data_size * 25 // 100
    test_size = data_size - train_size

    return X[:train_size], Y[:train_size], X[:-test_size], Y[:-test_size]


X = tf.placeholder(tf.float32, [None, time_step_size, input_vec_size], name="input")
Y = tf.placeholder(tf.float32, [None, label_size], name="output")

# get lstm_size and output 3 labels
W = tf.Variable(tf.random_normal([lstm_size, label_size], stddev=0.1), name="weights")
B = tf.Variable(tf.random_normal([label_size], stddev=0.1), name="biases")

W_hist = tf.histogram_summary("weights", W)
B_hist = tf.histogram_summary("biases", B)
Y_hist = tf.histogram_summary("output", Y)

py_x, state_size = model(X, W, B, lstm_size)

loss = tf.nn.softmax_cross_entropy_with_logits(py_x, Y)
cost = tf.reduce_mean(loss)
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess :
    # you need to initialize all variables
    sess.run(tf.global_variables_initializer())

    code_dates = get_code_dates()

    for epoch in range(epoch_size) :
        for batch in range(len(code_dates) // batch_size) :
            trX, trY, teX, teY = read_datas(code_dates[batch_size * batch : batch_size * batch + batch_size])
            sess.run(train_op, feed_dict={X: trX, Y: trY})
            res = sess.run(predict_op, feed_dict={X: teX, Y: teY})
            print("epoch: {0}, batch: {1}, accuracy: {2}".format(epoch, batch, np.mean(np.argmax(teY, 1) == res)))
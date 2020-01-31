import os
from datetime import datetime
import numpy as np
import pandas as pd
import sqlite3 as sql
import multiprocessing
from multiprocessing import Pool
import tensorflow as tf
 
class DataEnvironment:
    input_size = 7
    sequence_len = 60
    evaluate_len = 5
    validation_rate = 0.20
    test_rate = 0.20
    rise_rate = 0.3
    fall_rate = -0.04
    output_size = 3

class ModelEnvironment:
    epoch = 100
    batch_size = 10
    lstm_size = 14
    lstm_depth = 5
    learning_rate = 0.002


def get_db_connection():
    db_conn = sql.connect(
        os.path.dirname(__file__) + "/../../databases/finance_learning.db"
    )
    cursor = db_conn.cursor()
    cursor.execute("PRAGMA synchronous = OFF")

    return db_conn


def get_label_index(row, df, env):
    index = df.index.get_loc(row.name)
    windows_size = env.sequence_len + env.evaluate_len
    window_df = df[index:index + windows_size]
    hold_price = window_df.iloc[0]['open']

    for index, row_df in window_df.iterrows():
        high = row_df['high']
        low = row_df['low']

        if float(low - hold_price) / hold_price < env.fall_rate:
            return 2
        elif float(high - hold_price) / hold_price > env.rise_rate:
            return 0

    return 1


def create_train_infos_by_codes(codes, train_end_date, env):
    db_conn = get_db_connection()

    results = pd.DataFrame(columns=('code', 'date', 'label_index'))

    for code in codes:
        df = pd.read_sql_query(
            "SELECT date, open, high, low"
            " FROM stock_daily_series"
            " WHERE code = '{}' AND date <= '{}'"
            " ORDER BY date"
            .format(code, train_end_date),
            db_conn
            )

        if len(df) == 0:
            continue

        df['label_index'] = df.apply(lambda row: get_label_index(row, df, env), axis=1)
        results = results.append(pd.DataFrame({
            'code': code,
            'date': df['date'],
            'label_index': df['label_index']
            }))

    db_conn.close()
    return results


def get_train_end_date(db_conn, env):
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(date) FROM stock_daily_series WHERE code = 'A005930'")
    max_series_len = cursor.fetchone()[0]
    cursor.close()

    window_size = env.sequence_len + env.evaluate_len
    max_train_len = round((max_series_len - window_size) * (1 - env.test_rate))
    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT date"
        " FROM stock_daily_series WHERE code = 'A005930'"
        " ORDER BY date"
        " LIMIT 1 OFFSET {}"
        .format(max_train_len)
        )
    train_end_date = cursor.fetchone()[0]
    cursor.close()
    return train_end_date


def min_max_scaler(data):
    # return np.nan_to_num((data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0)))
    return np.nan_to_num((data - np.mean(data, axis = 0)) / np.std(data, axis = 0))


def get_datas(db_conn, data_infos, env):
    results = []
    for index, data_info in data_infos.iterrows():
        df = pd.read_sql_query(
            "SELECT open, high, low, close, volume, hold_foreign, st_purchase_inst"
            " FROM stock_daily_series"
            " WHERE code = '{}' AND '{}' <= date"
            " ORDER BY date"
            " LIMIT {}"
            .format(data_info.code, data_info.date, env.sequence_len),
            db_conn
            )
        norm = min_max_scaler(df.values)
        results.append(norm)

    return np.array(results)


def get_labels(data_infos):
    labels = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
    results = []
    for index, data_info in data_infos.iterrows():
        results.append(labels[int(data_info.label_index)])
    
    return results


def main():
    db_conn = get_db_connection()

    data_env = DataEnvironment()
    model_env = ModelEnvironment()

    X = tf.placeholder(tf.float32, [None, data_env.sequence_len, data_env.input_size])

    # RNN Layer
    cell = tf.contrib.rnn.GRUCell(model_env.lstm_size)
    cell = tf.contrib.rnn.DropoutWrapper(cell = cell, output_keep_prob = 0.5)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * model_env.lstm_depth, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    # Softmax Layer
    # W = tf.Variable(tf.random_normal([model_env.lstm_size, data_env.output_size], stddev=0.1))
    # B = tf.Variable(tf.random_normal([data_env.output_size], stddev=0.1))
    W = tf.get_variable("W", [model_env.lstm_size, data_env.output_size])
    B = tf.get_variable("B", [data_env.output_size])
    logits = tf.matmul(tf.transpose(outputs, [1, 0, 2])[-1], W) + B
    predict_op = tf.nn.softmax(logits)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.dirname(__file__) + '/stock_daily_learning.pd')


        for i in range(model_env.epoch):
            for pos in range(0, train_infos.shape[0], model_env.batch_size):
                X_train = get_datas(db_conn, train_infos[pos:pos+model_env.batch_size], data_env)
                L_train = get_labels(train_infos[pos:pos+model_env.batch_size])
                L_validation_predict = sess.run(predict_op, feed_dict={X:X_validation})


if __name__ == '__main__':   
    main()

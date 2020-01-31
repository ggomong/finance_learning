import os
from datetime import datetime
import numpy as np
import pandas as pd
import sqlite3 as sql
import tensorflow as tf
 
class DataEnvironment:
    input_size = 7
    sequence_len = 60
    evaluate_len = 5
    validation_rate = 0.20
    test_rate = 0.20
    output_size = 2

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


def create_train_infos(db_conn, env):
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(date) FROM stock_daily_series WHERE code = 'A005930'")
    max_series_len = cursor.fetchone()[0]
    cursor.close()

    max_train_len = round((max_series_len - env.sequence_len + env.evaluate_len) * (1 - env.test_rate))
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

    cursor = db_conn.cursor()
    cursor.execute("SELECT DISTINCT code FROM stock_daily_series LIMIT 5")
    codes = np.array(cursor.fetchall())[:, 0]
    cursor.close()

    train_infos = pd.DataFrame()
    num_process = multiprocessing.cpu_count()
    pool = Pool(num_process)
    split_codes = np.array_split(codes, num_process)

    async_results = create_train_infos_by_codes(codes, train_end_date, env)
    # async_results = [
    #     pool.apply_async(create_train_infos_by_codes, (split_code, train_end_date, env))
    #     for split_code in split_codes
    #     ]

    for async_result in async_results:
        train_infos = train_infos.append(async_result.get())

    rise_train_info_indexes = train_infos.label_index == 0
    stay_train_info_indexes = train_infos.label_index == 1
    fall_train_info_indexes = train_infos.label_index == 2

    train_infos_len = min(
        rise_train_info_indexes.sum(),
        stay_train_info_indexes.sum(),
        fall_train_info_indexes.sum()
        )

    if train_infos_len == 0:
        return pd.DataFrame(columns=train_infos.columns), train_end_date

    result = pd.concat((
        train_infos[rise_train_info_indexes].sample(train_infos_len),
        train_infos[stay_train_info_indexes].sample(train_infos_len),
        train_infos[fall_train_info_indexes].sample(train_infos_len)
        )).sample(frac=1)

    train_size = int(train_infos_len * (1 - env.validation_rate - env.test_rate))
    return result[:train_size], result[train_size:], train_end_date


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
    train_infos, validation_infos, train_end_date = create_train_infos(db_conn, data_env)

    X = tf.placeholder(tf.float32, [None, data_env.sequence_len, data_env.input_size])
    L = tf.placeholder(tf.float32, [None, data_env.output_size])

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
    # logits = tf.matmul(outputs[-1], W) + B
    predict_op = tf.nn.softmax(logits)
    
    # Cost Function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=L))
    train_op = tf.train.AdamOptimizer(model_env.learning_rate).minimize(cost)

    X_validation = get_datas(db_conn, validation_infos, data_env)
    L_validation = get_labels(validation_infos)

    with tf.Session() as sess:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(model_env.epoch):
            for pos in range(0, train_infos.shape[0], model_env.batch_size):
                X_train = get_datas(db_conn, train_infos[pos:pos+model_env.batch_size], data_env)
                L_train = get_labels(train_infos[pos:pos+model_env.batch_size])
                sess.run(train_op, feed_dict={X:X_train, L:L_train})
                L_validation_predict, train_cost = sess.run([predict_op, cost], feed_dict={X:X_validation, L:L_validation})
                # print(L_validation_predict)
                print("epoch: {}, batch: {}, cost: {:.6f}, accuracy: {:.4f} %".format(
                    i,
                    pos // model_env.batch_size,
                    train_cost,
                    np.mean(np.argmax(L_validation, 1) == np.argmax(L_validation_predict, 1)) * 100)
                    )

        saver = tf.train.Saver()
        saver.save(sess, os.path.dirname(__file__) + '/stock_daily_learning.pd')


if __name__ == '__main__':   
    main()

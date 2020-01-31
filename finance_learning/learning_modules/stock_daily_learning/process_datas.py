import os
from datetime import datetime
import numpy as np
import pandas as pd
import sqlite3 as sql
import multiprocessing
import tensorflow as tf


def get_db_connection():
    db_conn = sql.connect(
        os.path.dirname(__file__) + "/../../databases/finance_learning.db"
    )
    cursor = db_conn.cursor()
    # cursor.execute("PRAGMA synchronous = OFF")

    return db_conn


def get_codes(db_conn):
    cursor = db_conn.cursor()
    cursor.execute("SELECT DISTINCT code FROM stock_daily_series LIMIT 5")
    codes = np.array(cursor.fetchall())[:, 0]
    cursor.close()


def save_label_index(db_conn, row, df, env):
    index = df.index.get_loc(row.name)
    evaluate_df = df[index + 1:index + env.evaluate_len + 1]

    if len(evaluate_df) < env.evaluate_len:
        return

    hold_price = evaluate_df.iloc[0]['open']
    label_index = 0
    for index, row_df in evaluate_df.iterrows():
        high = row_df['high']
        low = row_df['low']

        if float(low - hold_price) / hold_price < env.fall_rate:
            label_index = 2
            break
        elif float(high - hold_price) / hold_price > env.rise_rate:
            label_index = 0
            break

    sql_str = "INSERT INTO label_indexes(code, date, label_index) VALUES(?, ?, ?)"
    cursor = db_conn.cursor()
    cursor.execute(
        sql_str,
        (
            row_df['code'],
            df.iloc[index]['date'],
            label_index
        )
    )
    db_conn.commit()


def save_label_indexes(db_conn, env):
    db_conn.execute(
        "CREATE TABLE IF NOT EXISTS label_indexes("
            "code TEXT, "
            "date DATE, "
            "label_indexe REAL, "
            "PRIMARY KEY(code, date)"
        ")")

    for code in get_codes(db_conn):
        df = pd.read_sql_query(
            "SELECT code, date, open, high, low"
            " FROM stock_daily_series"
            " WHERE code = '{}'"
            " ORDER BY date"
            .format(code),
            db_conn
            )

        if len(df) <= env.evaluate_len:
            continue

        df.apply(lambda row: save_label_index(db_conn, row, df, env), axis=1)

# 일별 수익율 데이터(시가대비 고가 상승률) 저장
import os as os
from datetime import datetime, timedelta
import sqlite3 as sqlite3
import finance_learning.databases.database_module as db


# 테이블이 없으면 생성
def create_table(db_conn):
    db_conn.excute(
        "CREATE TABLE IF NOT EXISTS stock_daily_returns("
            "code TEXT, "
            "date DATE, "
            "return REAL, "
            "PRIMARY KEY(code, date)"
        ")"
    )


# 해당 종목의 마지막 데이터의 날짜를 얻음
def get_last_stock_return_date(db_conn, code):
    cursor = db_conn.cursor()
    cursor.execute("SELECT date FROM stock_daily_returns WHERE code = '{0}' ORDER BY date DESC LIMIT 1".format(code))
    d = cursor.fetchone()
    if d == None:
        return datetime(1900, 1, 1)     # 데이터가 없으면 1900-01-01

    return d


def save_stock_return_data(db_conn):
    cursor = db_conn.cursor()
    cursor.execute("SELECT DISTINCT code FROM stock_daily_series")

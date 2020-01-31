# 주가/환율 일간 시계열 데이터 저장(32bit)
from datetime import datetime, timedelta
import win32com.client as com
import finance_learning.databases.database_module as db

START_SERIES_DATE = 20040101    #주식 거래원 정보가 제공되기 시작한 날짜


# 테이블이 없으면 생성
def create_table(db_conn):
    db_conn.execute(
        "CREATE TABLE IF NOT EXISTS stock_daily_series("
            "code TEXT, "
            "date DATE, "
            "open INTEGER, "
            "high INTEGER, "
            "low INTEGER, "
            "close INTEGER, "
            "volume INTEGER, "
            "hold_foreign REAL, "
            "st_purchase_inst REAL, "
            "PRIMARY KEY(code, date)"
        ")"
    )
        
    db_conn.execute(
        "CREATE TABLE IF NOT EXISTS exchange_daily_series("
            "code TEXT, "
            "date DATE, "
            "open REAL, "
            "high REAL, "
            "low REAL, "
            "close REAL, "
            "PRIMARY KEY(code, date)"
        ")"
    )


# 주식 일간 시계열 테이블에 데이터를 저장
def save_stock_data(db_conn, code, stock_chart):
    sql_str = "INSERT INTO stock_daily_series(code, date, open, high, low, close, volume, hold_foreign, st_purchase_inst) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
    cursor = db_conn.cursor()

    for i in range(stock_chart.GetHeaderValue(3)):
        dt = stock_chart.GetDataValue(0, i)  # 날자
        cursor.execute(
            sql_str,
            (
                code,
                datetime(dt // 10000, dt // 100 % 100, dt % 100),
                stock_chart.GetDataValue(1, i),                     # 시가
                stock_chart.GetDataValue(2, i),                     # 고가
                stock_chart.GetDataValue(3, i),                     # 저가
                stock_chart.GetDataValue(4, i),                     # 종가
                stock_chart.GetDataValue(5, i),                     # 거래량
                float(stock_chart.GetDataValue(6, i)),              # 외국인 보유수량
                float(stock_chart.GetDataValue(7, i))               # 기관 누적 순매수
            )
        )

    db_conn.commit()


# 환율 일간 시계열 테이블에 데이터를 저장
def save_exchange_data(db_conn, code, exchange_chart):
    sql_str = "INSERT INTO exchange_daily_series(code, date, open, high, low, close) VALUES(?, ?, ?, ?, ?, ?)"
    cursor = db_conn.cursor()
    possDate = get_possible_exchange_store_date(db_conn, code)

    for i in range(exchange_chart.GetHeaderValue(3)):
        dt = exchange_chart.GetDataValue(0, i)  # 날자
        if dt < possDate:
            break

        cursor.execute(
            sql_str,
            (
                code,
                datetime(dt // 10000, dt // 100 % 100, dt % 100),
                round(exchange_chart.GetDataValue(1, i), 2),           # 시가
                round(exchange_chart.GetDataValue(2, i), 2),           # 고가
                round(exchange_chart.GetDataValue(3, i), 2),           # 저가
                round(exchange_chart.GetDataValue(4, i), 2),           # 종가
            )
        )

    db_conn.commit()


# 주식 일간 시계열 테이블에서 마지막 데이터의 다음날 즉 저장할 데이터의 날자를 얻음
def get_possible_stock_store_date(db_conn, code):
    cursor = db_conn.cursor()
    cursor.execute("SELECT date FROM stock_daily_series WHERE code = '{0}' ORDER BY date DESC LIMIT 1".format(code))
    d = cursor.fetchone()
    if d == None:
        return START_SERIES_DATE     # 데이터가 없으면 지정한 날까지 읽음
    
    dt = datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") + timedelta(days = 1) 
    return dt.year * 10000 + dt.month * 100 + dt.day


# 환율 일간 시계열 테이블에서 환율 데이터가 저장된 다음날 즉 저장할 데이터의 날자를 얻음
def get_possible_exchange_store_date(db_conn, code):
    cursor = db_conn.cursor()
    cursor.execute("SELECT date FROM exchange_daily_series WHERE code = '{0}' ORDER BY date DESC LIMIT 1".format(code))
    d = cursor.fetchone()
    if d == None:
        return START_SERIES_DATE     # 데이터가 없으면 지정한 날까지 읽음
    
    dt = datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") + timedelta(days = 1) 
    return dt.year * 10000 + dt.month * 100 + dt.day


def get_stcok_data(db_conn):
    stock_chart = com.Dispatch("CpSysDib.StockChart")
    stock_chart.SetInputValue(1, ord('1'))                      # 기간으로 요청
    stock_chart.SetInputValue(5, (0, 2, 3, 4, 5, 8, 16, 21))    # 요청필드(날짜, 시가, 고가, 저가, 종가, 거래량, 외국인 보유수량, 기관 누적 순매수
    stock_chart.SetInputValue(6, ord('D'))                      # 일간데이터
    stock_chart.SetInputValue(9, ord('1'))                      # 수정주가 요청

    stock_code = com.Dispatch("CpUtil.CpStockCode")
    for i in range(stock_code.GetCount()):
        code = stock_code.GetData(0, i)
        if (code[0] != 'A'):
            continue

        possDate = get_possible_stock_store_date(db_conn, code)
        stock_chart.SetInputValue(0, code)
        stock_chart.SetInputValue(3, possDate)                  # 종료일
    
        if stock_chart.BlockRequest() != 0 or stock_chart.GetDibStatus() != 0: # 오류시
            continue

        if stock_chart.GetHeaderValue(5) < possDate:            #  최종 영업일이 요청일 보다 이전인 경우 Skip
            continue

        save_stock_data(db_conn, code, stock_chart)

        while stock_chart.Continue:
            if stock_chart.BlockRequest() != 0 or stock_chart.GetDibStatus() != 0: # 오류시
                continue
            save_stock_data(db_conn, code, stock_chart)


def get_exchange_data(db_conn):
    code = "FX@KRW"

    exchange_chart = com.Dispatch("DSCBO1.CpSvr8300")
    exchange_chart.SetInputValue(0, code)                       # 코드
    exchange_chart.SetInputValue(1, ord('D'))                   # 일간데이터
    exchange_chart.SetInputValue(3, 9999)                       # 요청개수

    if exchange_chart.BlockRequest() != 0 or exchange_chart.GetDibStatus() != 0: # 오류시
        return

    save_exchange_data(db_conn, code, exchange_chart)


db_conn = db.get_connection()
create_table(db_conn)
get_exchange_data(db_conn)
get_stcok_data(db_conn)
db_conn.close()
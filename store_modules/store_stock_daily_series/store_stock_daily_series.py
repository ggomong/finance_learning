from datetime import datetime, timedelta
import sqlite3 as sql
import win32com.client as com

# 테이블이 없으면 생성
def create_table(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS stock_daily_series(code TEXT, date DATE, open INTEGER, high INTEGER, low INTEGER, close INTEGER, volume INTEGER, hold_foreign REAL, st_purchase_inst REAL, PRIMARY KEY(code, date))")

# 테이블에 데이터를 저장
def save_data(conn, code, stock_chart):
    sql_str = "INSERT INTO stock_daily_series(code, date, open, high, low, close, volume, hold_foreign, st_purchase_inst) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
    cursor = conn.cursor()

    for i in range(stock_chart.GetHeaderValue(3)):
        dt = stock_chart.GetDataValue(0, i)  # 날자
        cursor.execute(
            sql_str,
            (
                code,
                datetime(dt // 10000, dt // 100 % 100, dt % 100),
                stock_chart.GetDataValue(1, i),          # 시가
                stock_chart.GetDataValue(2, i),          # 고가
                stock_chart.GetDataValue(3, i),          # 저가
                stock_chart.GetDataValue(4, i),          # 종가
                stock_chart.GetDataValue(5, i),          # 거래량
                float(stock_chart.GetDataValue(6, i)),   # 외국인 보유수량
                float(stock_chart.GetDataValue(7, i))    # 기관 누적 순매수
            )
        )
    conn.commit()

# DB에 저장된 다음날 즉 저장할 데이터의 날자를 얻음
def get_possible_store_date(conn, code):
    cursor = conn.cursor()
    cursor.execute("SELECT date FROM stock_daily_series WHERE code = '{0}' ORDER BY date DESC LIMIT 1".format(code))
    d = cursor.fetchone()
    if d == None:
        return 20040101     # 데이터가 없으면 거래원 데이터 제공하는 날짜까지 읽음
    
    dt = datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") + timedelta(days = 1) 
    return dt.year * 10000 + dt.month * 100 + dt.day

conn = sql.connect("../../databases/finance_learning.db")
with conn:
    create_table(conn)

    stock_chart = com.Dispatch("CpSysDib.StockChart")
    stock_chart.SetInputValue(1, ord('1'))       # 기간으로 요청
    stock_chart.SetInputValue(5, (0, 2, 3, 4, 5, 8, 16, 21)) # 요청필드(날짜, 시가, 고가, 저가, 종가, 거래량, 외국인 보유수량, 기관 누적 순매수
    stock_chart.SetInputValue(6, ord('D'))       # 일간데이터
    stock_chart.SetInputValue(9, ord('1'))       # 수정주가 요청

    code_mgr = com.Dispatch("CpUtil.CpCodeMgr")
    for code in code_mgr.GetGroupCodeList(180):  # KOSPI 200
        possDate = get_possible_store_date(conn, code)
        stock_chart.SetInputValue(0, code)
        stock_chart.SetInputValue(3, possDate)   # 종료일
    
        if stock_chart.BlockRequest() != 0 or stock_chart.GetDibStatus() != 0: # 오류시
            continue

        if stock_chart.GetHeaderValue(5) < possDate: #  최종 영업일이 요청일 보다 이전인 경우 Skip
            continue

        save_data(conn, code, stock_chart)

        while stock_chart.Continue:
            if stock_chart.BlockRequest() != 0 or stock_chart.GetDibStatus() != 0: # 오류시
                continue

            save_data(conn, code, stock_chart)


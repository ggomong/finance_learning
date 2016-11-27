# Finance Learning

TensorFlow Framework를 이용한 금융 데이터의 딥러닝 학습

***Topics***
* [코스피 200종목의 일간데이터 저장](store_modules/store_stock_daily_series/store_stock_daily_series.pyproj)
    * 코스피 200종목의 일간데이터(시/고/저/종가, 거래량, 외국인 보유수량, 기관 누적 순매수)를 DB에 저장
    * SQLite DB(databases/finance_learning.db) stock_daily_series 테이블에 code와 date를 Primary키로 저장
    * COM을 이용하여 주가 데이터를 받아오기 때문에 32bit 환경에서 실행해야 함
    * CybosPlus 특성상 관리자 권한으로 실행해야 함
    * Windows 환경에서만 실행이 가능 
* [코스피 200종목의 일간데이터를 이용한 주가예측](learning_modules/stock_daily_learning/stock_daily_learning.py)
    * 예측하는 주가는 60일간의 데이터를 이용하여 향후 3일간 최고가가 4%초과(상승), 최저가가 -2%미만(하락), 그 이외(보합)로 구분하여 학습
    * 코스피 200종목의 일간데이터(시/고/저/종가, 거래량, 외국인 보유수량, 기관 누적 순매수)를 이용한 학습
    * SQLite DB(databases/finance_learning.db) stock_daily_series 테이블의 데이터를 이용

***Dependencies***
* TensorFlow
* Numpy
* SQLite3
* 대신증권 CybosPlus
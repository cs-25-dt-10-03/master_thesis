import sqlite3
import csv
import os
from datetime import datetime
from typing import List

DB_PATH = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(DB_PATH, "data.db")


def initializeDatabase():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS spotPrices(
                       datetime TEXT PRIMARY KEY UNIQUE,
                       priceArea TEXT NOT NULL,
                       price REAL NOT NULL
                       )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS mfrrPrices(
                       datetime TEXT PRIMARY KEY UNIQUE,
                       priceArea TEXT NOT NULL,
                       downPurchased REAL NOT NULL,
                       downPrice REAL NOT NULL,
                       upPurchased REAL NOT NULL,
                       upPrice REAL NOT NULL
                       )

                   ''')
    conn.commit()
    conn.close()
    print("Tables initialized.")


def insertSpotPriceData(file: str) -> None:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    with open(file, 'r') as fin:
        dr = csv.DictReader(fin)
        to_db = [(i['HourDK'], i['PriceArea'], i['SpotPriceEUR']) for i in dr]

    cursor.executemany('''
                       INSERT OR REPLACE INTO spotPrices
                       (datetime, priceArea, price)
                       VALUES (datetime(?), ?, ?);
                       ''', to_db)
    conn.commit()
    conn.close()


def insertMfrrPriceData(file: str) -> None:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    with open(file, 'r') as fin:
        dr = csv.DictReader(fin, delimiter=';')
        to_db = [(i['HourDK'], i['PriceArea'], i['mFRR_DownPurchased'], i['mFRR_DownPriceEUR'],
                  i['mFRR_UpPurchased'], i['mFRR_UpPriceEUR']) for i in dr]

    cursor.executemany('''
                       INSERT OR REPLACE INTO mfrrPrices
                       (datetime, priceArea, downPurchased,
                        downPrice, upPurchased, upPrice)
                       VALUES (datetime(?), ?, ?, ?, ?, ?);
                       ''', to_db)
    conn.commit()
    conn.close()


def fetchAllMfrrPrices() -> {}:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM mfrrPrices;")
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"retrieval error: {e}")
        return {}
    finally:
        conn.close()


def fetchAllSpotPrices() -> {}:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM spotPrices;")
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"retrieval error: {e}")
        return {}
    finally:
        conn.close()





def fetchSpotPricesByDate(time: int) -> float:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        dt = datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:00:00')
        cursor.execute("SELECT price FROM spotPrices WHERE datetime = ?;", (dt,))
        result = cursor.fetchall()
        return result[0][0]
    except sqlite3.Error as e:
        print(f"retrieval error: {e}")
        return {}
    finally:
        conn.close()


def fetchSpotPricesInRange(start_time: int, end_time: int) -> List[float]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        start_dt = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:00:00')
        end_dt = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:00:00')
        cursor.execute("SELECT datetime, price FROM spotPrices WHERE datetime BETWEEN ? AND ?;", (start_dt, end_dt))
        results = cursor.fetchall()

        spotprices = [row[0] for row in results] 
        
        return spotprices

    except sqlite3.Error as e:
        print(f"Database retrieval error: {e}")
        return {}
    finally:
        conn.close()
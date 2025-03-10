import sqlite3
import csv
import os

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
                       INSERT OR REPLACE INTO spotPrices (datetime, priceArea, price)
                       VALUES (datetime(?), ?, ?);
                       ''', to_db)
    conn.commit()
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

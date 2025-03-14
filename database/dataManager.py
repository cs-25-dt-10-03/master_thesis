import sqlite3
import pandas as pd
import os
from typing import List, Any

DB_PATH = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(DB_PATH, "data.db")


def initializeDatabase():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.executescript("""
                         CREATE TABLE IF NOT EXISTS ev_models(
                             model TEXT PRIMARY KEY UNIQUE,
                             batteri_capacity REAL NOT NULL,
                             milage REAL NOT NULL,
                             charging_power REAL NOT NULL
                             );

                         CREATE TABLE IF NOT EXISTS economy(
                             passed_hours REAL PRIMARY KEY UNIQUE,
                             timestamp TEXT NOT NULL,
                             year REAL NOT NULL,
                             dso_earning REAL NOT NULL,
                             chargeing_box_supplier_earnings REAL NOT NULL
                             );

                         CREATE TABLE IF NOT EXISTS electricity_prices(
                             passed_hours REAL PRIMARY KEY UNIQUE,
                             timestamp TEXT NOT NULL,
                             year REAL NOT NULL,
                             month REAL NOT NULL,
                             day REAL NOT NULL,
                             hour REAL NOT NULL,
                             minute REAL NOT NULL,
                             spotPrice REAL NOT NULL,
                             tso_tariff REAL NOT NULL,
                             dso_tariff REAL NOT NULL,
                             co2_emmision REAL NOT NULL
                             );

                         CREATE TABLE IF NOT EXISTS timestamps (
                             id INTEGER PRIMARY KEY AUTOINCREMENT,
                             timestamp DATETIME,
                             year REAL,
                             month REAL,
                             day REAL,
                             hour REAL,
                             minute REAL
                             );

                         CREATE TABLE IF NOT EXISTS households (
                             id INTEGER PRIMARY KEY AUTOINCREMENT,
                             household_name TEXT UNIQUE
                             );

                         CREATE TABLE IF NOT EXISTS ev_readings (
                             id INTEGER PRIMARY KEY AUTOINCREMENT,
                             timestamp_id INTEGER,
                             household_id INTEGER,
                             ev_data REAL,
                             FOREIGN KEY (timestamp_id) REFERENCES timestamps(id),
                             FOREIGN KEY (household_id) REFERENCES households(id)
                             );

                         CREATE TABLE IF NOT EXISTS measurements(
                             passed_hours REAL PRIMARY KEY UNIQUE,
                             timestamp TEXT NOT NULL,
                             year REAL NOT NULL,
                             month REAL NOT NULL,
                             day REAL NOT NULL,
                             hour REAL NOT NULL,
                             minute REAL NOT NULL,
                             Total_num_of_evs REAL NOT NULL,
                             number_of_charging_evs REAL NOT NULL,
                             number_of_driving_evs  REAL NOT NULL,
                             total_grid_load REAL NOT NULL,
                             aggregated_base_load REAL NOT NULL,
                             aggregated_charging_load REAL NOT NULL,
                             overload_duration REAL NOT NULL
                             );
                         """)
    conn.commit()
    conn.close()
    print("Tables initialized.")


def insertSimpleData(file: str, table: str) -> None:
    conn = sqlite3.connect(DB_NAME)

    df = pd.read_csv(file, skiprows=1)
    df.to_sql(table, conn, if_exists='replace', index=False)

    conn.commit()
    conn.close()


def insertHouseholdData(file: str) -> None:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    names = pd.read_csv(file, dtype="unicode").iloc[0]

    df = pd.read_csv(file, skiprows=1, dtype="unicode")

    time_cols = ['Timestamp', 'Year', 'Month', 'Day', 'Hour', 'Minute']

    time_df = df[time_cols].drop_duplicates()
    for _, row in time_df.iterrows():
        cursor.execute(
            "INSERT INTO timestamps (timestamp, year, month, day, hour, minute) VALUES (?, ?, ?, ?, ?, ?)",
            (row['Timestamp'], row['Year'], row['Month'], row['Day'], row['Hour'], row['Minute'])
        )

    for household in names:
        cursor.execute(
            "INSERT OR REPLACE INTO households (household_name) VALUES (?)",
            (household,)
        )
    data = []
    r = len(df)
    for i, row in df.iterrows():
        print(i, "/", r)
        timestamp = row['Timestamp']
        cursor.execute("SELECT id FROM timestamps WHERE timestamp = ?", (timestamp,))
        timestamp_id = cursor.fetchone()[0]

        for household in names:
            cursor.execute("SELECT id FROM households WHERE household_name = ?", (household,))
            household_id = cursor.fetchone()[0]
            data.append((timestamp_id, household_id, row[household]))

    cursor.executemany("INSERT INTO ev_readings (timestamp_id, household_id, ev_data) VALUES (?, ?, ?)", data)

    conn.commit()
    conn.close()


def fetchAllEvModels() -> List[Any]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM ev_models;")
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"retrieval error: {e}")
        return []
    finally:
        conn.close()


def fetchAllEconomy() -> List[Any]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM economy;")
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"retrieval error: {e}")
        return []
    finally:
        conn.close()


def fetchAllElectricityPrices() -> List[Any]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM electricity_prices;")
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"retrieval error: {e}")
        return []
    finally:
        conn.close()


def fetchAllMeasurements() -> List[Any]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM measurements;")
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"retrieval error: {e}")
        return []
    finally:
        conn.close()


def fetchAllHouseholdData():
    conn = sqlite3.connect(DB_NAME)
    query = """
                SELECT t.timestamp, h.household_name, e.ev_data
                FROM ev_readings e
                JOIN timestamps t ON e.timestamp_id = t.id
                JOIN households h ON e.household_id = h.id
                ORDER BY t.timestamp, h.household_name
                LIMIT 10;
            """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

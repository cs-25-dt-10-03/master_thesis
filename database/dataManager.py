import sqlite3
import os
import subprocess
import pandas as pd

DB_PATH = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(DB_PATH, "data.db")

def initializeDatabase():
    if not os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS averagePower (
                id INTEGER PRIMARY KEY,
                year INT NOT NULL,
                month INT NOT NULL,
                hour INT NOT NULL,
                dayType TEXT NOT NULL,
                power double NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"{DB_NAME} initialized with average power table. (More tables to come)")
    else:
        print(f"{DB_NAME} already exists.")


def fetchAveragePower():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM averagePower")
        data = cursor.fetchall()

        for elem in data:
            print(elem)
    except sqlite3.Error as e:
        print(f"Error fetching users: {e}")
    finally:
        conn.close()


def pullLatestChanges():
    try:
        subprocess.run(["git", "pull"], cwd=DB_PATH, check=True)
        print("Pulled latest changes from GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull changes: {e}")

def uploadChanges():
    try:
        subprocess.run(["git", "add", DB_NAME], cwd=DB_PATH, check=True)
        subprocess.run(["git", "commit", "-m", "Update database"], cwd=DB_PATH, check=True)
        subprocess.run(["git", "push"], cwd=DB_PATH, check=True)
        print("Database changes uploaded to GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to upload changes: {e}")

def insertData(filePath):

    df = pd.read_excel(filePath)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute('''
            INSERT INTO averagePower (Year, Month, Hour, DayType, power)
            VALUES (?, ?, ?, ?, ?)
        ''', (row['Year'], row['Month'], row['Hour'], row['DayType'], row['power']))

    conn.commit()
    conn.close()

import sqlite3
import os
from datetime import datetime

from classes.flexOffer import flexOffer

DB_PATH = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(DB_PATH, "data.db")

def initializeDatabase():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS spotPrices(
                       datetime INTEGER PRIMARY KEY,
                       price REAL NOT NULL
                       )
                   ''')

    # Ik tænk over det, det er bare så vi kan bruge energyprofile table på alle forskellige FO
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS entity(
                       id INTEGER PRIMARY KEY AUTOINCREMENT
                       )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS sfo(
                       id INTEGER PRIMARY KEY REFERENCES entity(id),
                       est INTEGER NOT NULL,
                       et INTEGER NOT NULL
                       )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS tec(
                       id INTEGER PRIMARY KEY REFERENCES entity(id),
                       est INTEGER NOT NULL,
                       et INTEGER NOT NULL,
                       min_energy REAL NOT NULL,
                       max_energy REAL NOT NULL,
                       total_energy_limit REAL NOT NULL
                       )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS energyProfile(
                       id INTEGER PRIMARY KEY,
                       flexoffer_id INTEGER,
                       min_energy REAL NOT NULL,
                       max_energy REAL NOT NULL,
                       FOREIGN KEY (flexoffer_id) REFERENCES entity(id)
                       )
                   ''')

    conn.commit()
    conn.close()
    print("Tables initialized.")


def insertData(fo):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute("PRAGMA foreign_keys = ON;")

        cursor.execute("INSERT INTO entity DEFAULT VALUES;")
        offer_id = cursor.lastrowid  

        est = int(fo.earliest_start.timestamp())
        et = int(fo.end_time.timestamp())

        if isinstance(fo, flexOffer) and fo.total_energy_limit is not None:
            cursor.execute('''
                INSERT INTO tec (id, est, et, min_energy, max_energy, total_energy_limit)
                VALUES (?, ?, ?, ?, ?, ?);
            ''', (offer_id, est, et, fo.min_energy, fo.max_energy, fo.total_energy_limit))

        elif isinstance(fo, flexOffer):
            cursor.execute('''
                INSERT INTO sfo (id, est, et)
                VALUES (?, ?, ?);
            ''', (offer_id, est, et))
        else:
            print("dfo er ikke implementeret enndu")
        
        
        for min_e, max_e in fo.energy_profile:
            cursor.execute('''
                INSERT INTO energyProfile (flexoffer_id, min_energy, max_energy)
                VALUES (?, ?, ?);
            ''', (offer_id, min_e, max_e))

        conn.commit()

    except sqlite3.Error as e:
        print(f"Insertion error: {e}")
        conn.rollback() 
        
    finally:
        conn.close()


def fetch_all_offers(): 
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    result = {"sfo": [], "tec": []}

    try:
        cursor.execute("SELECT * FROM sfo;")
        sfo_offers = cursor.fetchall()

        for sfo in sfo_offers:
            sfo_id, est, et= sfo

            est_dt = datetime.fromtimestamp(est)
            et_dt = datetime.fromtimestamp(et)

            cursor.execute("SELECT min_energy, max_energy FROM energyProfile WHERE flexoffer_id = ?;", (sfo_id,))
            energy_profiles = cursor.fetchall()

            result["sfo"].append({
                "id": sfo_id,
                "earliest_start": est_dt,
                "end_time": et_dt,
                "energy_profile": [{"min_energy": ep[0], "max_energy": ep[1]} for ep in energy_profiles]
            })

        cursor.execute("SELECT * FROM tec;")
        tec_offers = cursor.fetchall()

        for tec in tec_offers:
            tec_id, est, et, min_energy, max_energy, total_energy_limit = tec

            est_dt = datetime.fromtimestamp(est)
            et_dt = datetime.fromtimestamp(et)

            cursor.execute("SELECT min_energy, max_energy FROM energyProfile WHERE flexoffer_id = ?;", (tec_id,))
            energy_profiles = cursor.fetchall()

            result["tec"].append({
                "id": tec_id,
                "earliest_start": est_dt,
                "end_time": et_dt,
                "min_energy": min_energy,
                "max_energy": max_energy,
                "total_energy_limit": total_energy_limit,
                "energy_profile": [{"min_energy": ep[0], "max_energy": ep[1]} for ep in energy_profiles]
            })

        return result

    except sqlite3.Error as e:
        print(f"retrieval error: {e}")
        return {}

    finally:
        conn.close()


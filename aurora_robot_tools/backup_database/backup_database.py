"""Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Script to backup the chemspeedDB database to a folder with the base sample ID as the filename.

"""

import os
import shutil
import sqlite3
from datetime import datetime

DATABASE_FILEPATH = "C:\\Modules\\Database\\chemspeedDB.db"
BACKUP_FOLDER = "C:\\Modules\\Database\\Backup"

try:
    with sqlite3.connect(DATABASE_FILEPATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM Settings_Table WHERE key = 'Base Sample ID'")
        result = cursor.fetchone()
        value = result[0] if result is not None else ""
except sqlite3.Error as e:
    print("Database error: ", e)
    value = ""

if value == "":
    value = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("Base Sample ID not found in the database. Using current timestamp instead.")

# copy database file to backup folder with the base sample ID as the filename
os.makedirs(BACKUP_FOLDER, exist_ok=True)
backup_filepath = f"{BACKUP_FOLDER}\\{value}.db"
shutil.copy(DATABASE_FILEPATH, backup_filepath)
print(f"Database backed up to {backup_filepath}.")

import random
import sqlite3
import os
import load_data.get_path as get_path
import pandas as pd

def get_random(index):
    ran = int(random.random() *len(index))
    return index[ran]

def build_new_sqlite(user_name,password):
    db_name = os.path.join(get_path.sqlite_folder_path,f"{user_name}.sqlite")
    if os.path.exists(db_name):
        return False
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS UserInfo 
    (
        username TEXT NOT NULL PRIMARY KEY,
        password TEXT NOT NULL,

        index_id INTEGER
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS InterAction
    (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        video_id INTEGER,
        timestamp REAL NOT NULL,
        watch_ratio INTEGER
    )
    ''')

    user_feature = pd.read_csv(get_path.user_feature_path,index_col=0)
    ran = get_random(user_feature.index)


    cursor.execute('''INSERT INTO UserInfo VALUES (?,?,?)''' ,
                (user_name, password, ran.item()))

    conn.commit()
    conn.close()
    print("注册成功")
    return True

def login_sql(user_name,password):
    db_name = os.path.join(get_path.sqlite_folder_path,f"{user_name}.sqlite")
    if not os.path.exists(db_name):
        return False
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute('''SELECT username,password from UserInfo''')

    result = cursor.fetchall()

    for row in result:
        sq_uname,sq_ps = row
        if sq_uname == user_name and sq_ps == password:
            return True
    return False
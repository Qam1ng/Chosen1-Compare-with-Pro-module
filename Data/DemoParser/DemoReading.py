from demoparser2 import DemoParser
import sqlite3
import os


# 替换为你的demo文件路径
demo_path = "your_demo_file.dem"
# 替换为你的database文件路径
DB_path = "DateBase_file.db"

def setup_database():
    conn = sqlite3.connect(DB_path)
    cursor = conn.cursor()
    
    # 创建 matches 表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS matches (
        match_id INTEGER PRIMARY KEY AUTOINCREMENT,
        demo_filename TEXT NOT NULL UNIQUE,
        map_name TEXT,
        duration_seconds REAL,
        parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # 创建 players 表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS players (
        player_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        steam_id TEXT,
        name TEXT,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    )''')


    conn.commit()
    conn.close()

def parse_and_store_demo(demo_path):
    """解析demo并存入数据库"""
    if not os.path.exists(demo_path):
        print(f"Demo文件不存在: {demo_path}")
        return

    conn = sqlite3.connect(DB_path)
    cursor = conn.cursor()
    
    parser = DemoParser(demo_path)

    # 1. 插入比赛信息
    header = parser.parse_header()
    demo_filename = os.path.basename(demo_path)
    cursor.execute(
        "INSERT OR IGNORE INTO matches (demo_filename, map_name, duration_seconds) VALUES (?, ?, ?)",
        (demo_filename, header['map_name'], header['playback_time'])
    )
    # 获取刚刚插入的比赛的 match_id
    cursor.execute("SELECT match_id FROM matches WHERE demo_filename=?", (demo_filename,))
    match_id = cursor.fetchone()[0]

    # 2. 插入玩家信息
    result = parser.parse_result()
    for player in result['players']:
        cursor.execute(
            "INSERT INTO players (match_id, steam_id, name) VALUES (?, ?, ?)",
            (match_id, player['steam_id'], player['name'])
        )
import json
import os
import random
import uuid

# --- 1. 定义数据池 ---
MAP_POOL = ["de_mirage", "de_inferno"] #["de_dust2", "de_nuke", "de_overpass", "de_ancient"]出于MVP考虑暂时不加入
TEAM_POOL = ["CT", "T"]
ACTION_POOL = ["peek", "throw_grenade", "fire_weapon", "plant_bomb", "defuse_kit", "hold_angle"]
WEAPON_POOL = ["ak47", "m4a4", "awp", "usp-s", "glock", "grenade", "smoke", "flash", "molly"]
LOCATION_POOL = {
    "de_mirage": ["mid", "A_site", "palace", "connector", "B_apps", "catwalk"],
    "de_inferno": ["banana", "B_site", "A_site", "long", "short", "apartments"]
    # 为其他地图添加位置
}
OUTCOME_POOL = ["EnemySpoted", "Death", "EnemyDamaged", "FriendDamaged", "Assist"]
#Potential Impact IMPACT_POOL = ["LossControl", "MapInformation", "CT_Depletion", "T_Advantage", "ProjectileLoss"]

# --- 2. 随机生成函数 ---
def generate_fake_cs_event():
    selected_map = random.choice(MAP_POOL)
    
    # 生成10个玩家
    players_list = []
    player_names = [f"Player_{i+1}" for i in range(10)]
    for i in range(10):
        players_list.append({
            "player_id": f"76561198{random.randint(100000000, 999999999)}", # 模拟Steam ID，可以手动调控减小随机度，模拟特定玩家
            "name": player_names[i],
            "team": random.choice(TEAM_POOL)
        })

    # 随机选择玩家作为事件主角
    main_player = random.choice(players_list)
    
    # 随机选择目标玩家
    target_player = random.choice([p["name"] for p in players_list if p["name"] != main_player["name"]])

    # 创建核心事件
    trajectory_event = {
        "timestamp": round(random.uniform(5.0, 90.0), 1),
        "action": random.choice(ACTION_POOL),
        "location": random.choice(LOCATION_POOL.get(selected_map, ["unknown_location"])),
        "result": {
            "outcome": random.sample(OUTCOME_POOL, k=random.randint(1, 2)),
            #"impact": random.sample(IMPACT_POOL, k=random.randint(1, 2)),
            "targets": [target_player],
            "weapon": [random.choice(WEAPON_POOL)],
            "damage": [random.randint(0, 100)]
        }
    }
    
    # 组装JSON
    output_json = {
        "match_id": str(uuid.uuid4()), # 使用UUID生成唯一的ID
        "map": selected_map,
        "rounds": [
            {
                "round_number": random.randint(1, 30),
                "players": [
                    {
                        "player_id": main_player["player_id"],
                        "name": main_player["name"],
                        "team": main_player["team"],
                        "trajectory": [trajectory_event] # 简化为每个玩家只有一个事件
                    }
                ]
            }
        ],
        "metadata": {
            "source": "fake_data_generator",
            "version": 1.0,
            "created_at": "2025-08-18T14:00:00Z"
        }
    }
    
    return output_json

# --- 3. 生成并打印一个样本 ---
if __name__ == "__main__":
    fake_data = generate_fake_cs_event()
    
    # 定义文件保存路径
    output_dir = "output_json_files"
    os.makedirs(output_dir, exist_ok=True)
    ######################################################## To be Finished

    
    # 打印格式化的JSON
    print(json.dumps(fake_data, indent=2))
    
    ######################################################## To be implemented
    # 循环来生成大量文件
    # for i in range(1000):
    #     with open(f'sample_{i+1}.json', 'w') as f:
    #         json.dump(generate_fake_cs_event(), f, indent=2)

from demoparser2 import DemoParser
import pandas as pd
import os
import csv
from datetime import datetime

steamid64_player   = 76561198386265483   # <-- 玩家的64位steamid


# Path to the folder where CSVs should be saved
output_folder = r"C:\\Users\\14799\\Desktop\\scouting\\donk_dust2" # <-- 输出文件夹

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# demo
folder = r"C:\\Users\\14799\\Desktop\\demos\\donk_dust2" # <-- demo文件夹

log_file = os.path.join(output_folder, "log.csv")

if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["demo_name", "processed_time"])

# Loop over all files in the folder
for filename in os.listdir(folder):
    if filename.endswith(".dem"):
        demo_path = os.path.join(folder, filename)
        parser      = DemoParser(demo_path)

        player_count = 0
        ## skip knife round and game setup
        # Round start tick
        rs = parser.parse_event("round_start", other = ["total_rounds_played"])
        START_TICK = rs.iloc[rs[rs["total_rounds_played"] == 0].index.max(), 1]
        END_TICK = parser.parse_event("round_end")["tick"].max()
        temp = parser.parse_ticks(wanted_props = ["total_rounds_played"], ticks= [START_TICK])
        steamid64_list = temp["steamid"]
        def getPlayerInfo(info):
            return info.loc[info["steamid"] == steamid64]



        for steamid64 in steamid64_list:
            ########################################   Start round information ################################################

            rs_info = parser.parse_ticks(wanted_props = ["start_balance", "team_name", "total_rounds_played"], ticks= rs["tick"])

            rs_info_player = getPlayerInfo(rs_info)

            ## Calculate team_balance
            team_balance_df = (
                rs_info.groupby(["tick","team_name"])["start_balance"]
                .sum().reset_index()
                .rename(columns={"start_balance":"team_balance"})
            )

            rs_info = rs_info.merge(team_balance_df, on=["tick","team_name"], how="left")

            ## Get the player I want
            rs_info_player = rs_info.loc[rs_info["steamid"] == steamid64]


            start_info = []
            for _, row in rs_info_player.iterrows():
                if row["tick"] < START_TICK:
                    continue
                
                temp = [row["team_name"], row["start_balance"], row["team_balance"]]
                start_info.append(temp)


            ################################################################################################

            ########################################  Every 60 ticks information ################################################

            STEP = 60

            tick_list = []

            for tick in range(START_TICK, END_TICK, STEP):
                
                tick_list.append(tick)

            ## Without time_remaining and bomb_time
            player_props = [
                "is_alive",
                "health","armor_value","has_helmet",
                "inventory",
                "has_defuser",
                "is_bomb_planted",
                "team_name",
                "is_freeze_period"
            ]



            sr_info = parser.parse_ticks(wanted_props = player_props, ticks = tick_list)
            ## Filter out freeze time
            sr_info = sr_info.loc[(sr_info["is_freeze_period"] == False)]

            sr_info_player = getPlayerInfo(sr_info)
            ## Filter out dead time
            sr_info_player = sr_info_player.loc[sr_info_player["is_alive"] == True]


            ## The information required for every 60 seconds
            second_record = []

            def expand_inventory(df):
                # start with all zeros
                df = df.copy()
                num_smokes    = 0
                num_flashbang = 0
                num_he        = 0
                num_molotov   = 0
                num_bomb = 0
                bomb_carrier = False
                
                for items in df["inventory"]:
                    if(items == "Smoke Grenade"):
                        num_smokes   += 1
                    if(items == "Flashbang"):
                        num_flashbang   += 1
                    if(items == "High Explosive Grenade"):
                        num_he   += 1
                    if(items == "Incendiary Grenade" or items == "Molotov"):
                        num_molotov   += 1
                    if(items == "C4 Explosive"):
                        num_bomb    += 1
                    if num_bomb != 0:
                        bomb_carrier = True
                return num_smokes, num_flashbang, num_he, num_molotov, bomb_carrier


            for _, row in sr_info_player.iterrows():
                tick = row["tick"]

                player_hp = row["health"]
                player_armor = row["armor_value"]
                player_has_helmet = row["has_helmet"]

                num_smokes = 0
                num_flashbang = 0
                num_he = 0
                num_molotov  = 0

                has_defuse_kit = row["has_defuser"]
                bomb_carrier = False

                bomb_is_planted = row["is_bomb_planted"]
                
                num_smokes, num_flashbang, num_he, num_molotov, bomb_carrier = expand_inventory(row)
                
                temp = [tick, 
                        player_hp, player_armor, player_has_helmet,
                        num_smokes, num_flashbang, num_he, num_molotov,
                        has_defuse_kit,
                        bomb_carrier,
                        bomb_is_planted]
                second_record.append(temp)

            #print(second_record)

            ################################################################################################

            ########################################  information When player gets hurt ################################################


            TICKRATE   = 60                    # 60 if you want exactly 12 ticks ≈ 0.2s; set 64/128 if needed
            STEP       = 12                    # sample every 12 ticks  (~5Hz at 60 tick)
            WINDOW_S   = 3                     # collect ±3 seconds around each damage tick
            WINDOW_T   = int(WINDOW_S * TICKRATE)


            hurt_info = parser.parse_event("player_hurt")

            center_ticks = hurt_info["tick"].tolist()
            tick_set = set()
            for t in center_ticks:
                start = max(START_TICK, t - WINDOW_T)
                end   = min(END_TICK, t + WINDOW_T)
                tick_set.update(range(start, end + 1, STEP))
            hurt_tick_list = sorted(tick_set)


            player_props = [
                        "is_alive",
                        # positions & view
                        "X","Y","Z",
                        "pitch","yaw",  # common names; adjust below if your build differs
                        # weapon info
                        "item_def_idx", "active_weapon_ammo",
                        # states
                        "is_scoped","flash_duration","in_crouch",
                        # prepare for enmy and ally
                        "team_name"
                    ]
            hurt_info_all = parser.parse_ticks(wanted_props= player_props, ticks = hurt_tick_list)

            hurt_info_player = getPlayerInfo(hurt_info_all)

            hurt_info_player = hurt_info_player.loc[hurt_info_player["is_alive"] == True]

            ## 受伤时/造成伤害时收集
            info_when_hurt = []

            for _, row in hurt_info_player.iterrows():
                tick = row["tick"]
                player_postion = []
                player_view_angles = []
                active_weapon_id = 0
                active_weapon_ammo = 0
                shots_fired = 0

                player_postion = [row["X"], row["Y"], row["Z"]]
                player_view_angles = [row["pitch"], row["yaw"]]
                active_weapon_id = row["item_def_idx"]
                active_weapon_ammo = row["active_weapon_ammo"]

                player_is_scoped = row["is_scoped"]
                player_is_flashed = row["flash_duration"] > 0
                player_is_crouching = row["in_crouch"]

                ctick = row["tick"]
                temp = hurt_info_all.copy()
                temp = temp.loc[(temp["is_alive"] == True) & (temp["tick"] == ctick)] 
                ally = []
                enemy = []
                for _, i in temp.iterrows():
                    if i["team_name"] == row["team_name"]:
                        ally.append([i["X"], i["Y"], i["Z"]])
                    else:
                        enemy.append([i["X"], i["Y"], i["Z"]])
                info_when_hurt.append([ tick,
                                        # positions & view
                                        player_postion, player_view_angles,  
                                        # weapon info
                                        active_weapon_id,active_weapon_ammo,shots_fired, 
                                        # Player state
                                        player_is_scoped, player_is_flashed, player_is_crouching,
                                        # Remaning enemy
                                        enemy,
                                        # Ally
                                        ally
                                        ])

            if steamid64 == steamid64_player:
                csv_name1 = os.path.splitext(filename)[0] + "_info_when_hurt.csv"
                csv_path1 = os.path.join(output_folder, csv_name1)
                info_when_hurt = pd.DataFrame(info_when_hurt)
                info_when_hurt.to_csv(csv_path1, index=False)

                csv_name2 = os.path.splitext(filename)[0] + "_start_info.csv"
                csv_path2 = os.path.join(output_folder, csv_name2)
                start_info = pd.DataFrame(start_info)
                start_info.to_csv(csv_path2, index=False)

                csv_name3 = os.path.splitext(filename)[0] + "_second_record.csv"
                csv_path3 = os.path.join(output_folder, csv_name3)
                second_record = pd.DataFrame(second_record)
                second_record.to_csv(csv_path3, index=False)

                with open(log_file, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            else:
                csv_name1 = "not_" + os.path.splitext(filename)[0] +"_" + str(player_count) + "_info_when_hurt.csv"
                csv_path1 = os.path.join(output_folder, csv_name1)
                info_when_hurt = pd.DataFrame(info_when_hurt)
                info_when_hurt.to_csv(csv_path1, index=False)

                csv_name2 = "not_" + os.path.splitext(filename)[0] +"_" + str(player_count) + "_start_info.csv"
                csv_path2 = os.path.join(output_folder, csv_name2)
                start_info = pd.DataFrame(start_info)
                start_info.to_csv(csv_path2, index=False)

                csv_name3 = "not_" + os.path.splitext(filename)[0] +"_" + str(player_count) + "_second_record.csv"
                csv_path3 = os.path.join(output_folder, csv_name3)
                second_record = pd.DataFrame(second_record)
                second_record.to_csv(csv_path3, index=False)
                player_count += 1

                with open(log_file, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

print("process finished")        
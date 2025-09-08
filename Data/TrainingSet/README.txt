trainSet.py有基础的拉取demo信息的功能

output_folder = r"C:\\Users\\14799\\Desktop\\scouting\\m0nesy_dust2" # <-- demo文件夹

folder = r"C:\\Users\\14799\\Desktop\\demos\\m0nesy_dust2" # <-- 输出文件夹

steamid64   = 76561198074762801   # <-- 玩家的64位steamid

你只需要这三个信息就可以获得training set。


对于每一个demo，你都会得到3个csv文件，分别对应
start_info                                       ---------------->开局信息
second_record                             ---------------->每秒获取的信息
info_when_hurt                             ---------------->每当玩家受到伤害时， 收集的信息（前后3s， 每10tick收集一次）

其中的信息是按照以下格式



start_info = [team_name (TERRORIST/CT), start_balance (int), team_balance(int)]

second_record = [tick(int), player_hp(int 0-100), player_armor(int 0-100), player_has_helmet(bool),
                    num_smokes(int 0,1), num_flashbang(int 0,1), num_he(int 0,1), num_molotov(int 0,1),
                    has_defuse_kit(bool),
                    bomb_carrier(bool),
                    bomb_is_planted(bool)]

info_when_hurt = [ tick, 
				# positions & view
                                    player_postion(V[3] f), player_view_angles(V[2] f),,  
                                    # weapon info
                                    active_weapon_id(int),active_weapon_ammo(int),shots_fired(int), 
                                    # Player state
                                    player_is_scoped(bool), player_is_flashed(bool), player_is_crouching(bool),
                                    # Remaning enemy
                                    enemy(0-5个 V[3]),
                                    # Ally
                                    ally(0-5个 V[3])
				]		

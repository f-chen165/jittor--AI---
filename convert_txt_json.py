import json

seed_dict = dict()
with open(fr"C:\Users\fc\Desktop\tem\B_code\B_code_开源\B_weight\seed_img.txt", mode="r") as file_obj:
    for i_line in file_obj:
        thing_name = i_line.split(",")[0]
        style_num = int(i_line.split(",")[1].split(":")[0])
        seed_img = int(i_line.split(",")[1].split(":")[1])

        if style_num not in seed_dict.keys():
            seed_dict[style_num] = [(thing_name, seed_img)]
        else:
            seed_dict[style_num].append((thing_name, seed_img))
# 假设你有一个字典
my_dict = seed_dict

# 将字典保存为JSON文件
with open(fr'C:\Users\fc\Desktop\tem\B_code\B_code_开源\B_weight/my_dict.json', 'w', encoding='utf-8') as f:
    json.dump(my_dict, f, ensure_ascii=False)
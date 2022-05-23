# deprecated

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def notify(msg):
    import datetime
    channel="J0budaR2THarZw0OqS5O"
    notify_url=f"https://notify.run/{channel}"
    massage=f"{msg} | {str(datetime.datetime.now())}"
    os.system(f'curl {notify_url} -d "{massage}"')
    
names = [
    "sys_time", "license_number", "lng", "lat", "gps_time", "EMPTY1", "speed",
    "direction", "car_status", "alarm_status", "EMPTY2", "EMPTY3",
    "license_color", "recorder_speed", "mileage", "height", "EMPTY4"
]

# 这两个日期是需要处理的文件日期开头以及结尾+1
open_day = pd.to_datetime("2020-06-01 00:00:00")
close_day = pd.to_datetime("2020-07-01 00:00:00")

# dest_path是中间结果，暂存到一个文件夹中，之后读取再进行处理
dest_path = "../data/taxi_after_proc/intermediate/"
if not os.path.exists(dest_path):
    os.mkdir(dest_path)

# 获取taxi中所有文件，之后读取使用
root_path = os.walk("../data/taxi")
all_files = []
for root,ds,fs in root_path:
    days = []
    for f in fs:
        full_path = os.path.join(root,f)
        days.append(full_path)
    if len(days)==0:
        continue
    all_files.append(days)

# 将数据进行初步处理
all_files.sort()
for each_day in tqdm(all_files):
    each_day.sort()

    for each_time in tqdm(each_day):
        records = pd.read_csv(each_time, names=names)
        group = records.groupby("license_number")
        for each_lincense, each_record in group:
            posi = each_record[["lat", "lng", "gps_time", "speed"]]

            # 按gps时间排序，并去重
            posi["gps_time"] = pd.to_datetime(posi["gps_time"])
            posi.sort_values(by="gps_time", inplace=True, ascending=True)
            posi = posi.drop_duplicates(subset="gps_time", keep="first")

            posi = posi.loc[
                # 去除 GPS 0 值
                (posi["lat"] > 0) & (posi["lng"] > 0) &
                # 去除错误日期的记录
                (posi["gps_time"] > open_day) & (posi["gps_time"] < close_day) &
                # 保留速度 0~150 的记录
                (posi["speed"] > 0) & (posi["speed"] < 150)]

            # 去除数量过少的记录
            if len(posi) < 30:
                continue

            # 将结果输出到 dest_path 中
            posi.to_pickle(dest_path + each_time[-7:] + each_lincense + ".pkl")
            
notify("Finish intermediate.")

notify("Start merge.")

res_path="../data/taxi_after_proc/clean202006/"
if not os.path.exists(res_path):
    os.mkdir(res_path)

files = os.listdir(dest_path)
car_set = set()

for file in files:
    car = file[7:-4]
    car_set.add(car)

days = []
for i in range(1, 31):
    days.append(str(i).zfill(2))

for car in tqdm(car_set):
    for day in days:
        df_all = pd.DataFrame()
        for i in range(11):
            try:
                df_all = df_all.append(
                    pd.read_pickle(
                        "../data/taxi_after_proc/intermediate/06-{}_{}{}.pkl".
                        format(day, i, car)))
                df_all = df_all.drop_duplicates(subset="gps_time",
                                                keep="first")
                df_all = df_all[(df_all["gps_time"] > open_day)
                                & (df_all["gps_time"] < close_day)]
                df_all = df_all[(df_all["speed"] > 0)
                                & (df_all["speed"] < 150)]
#               print(df_all)
            except FileNotFoundError:
                continue
        df_all.to_pickle(
            "../data/taxi_after_proc/clean202006/06-{}_{}.pkl".format(day, car))
        
notify("Finish 202006.")

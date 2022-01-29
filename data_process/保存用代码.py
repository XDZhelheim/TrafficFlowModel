# %%
# 从这里往后不需要跑了
#
# # 接下来是处理，查看GPS track的
# path = "../data/taxi_after_proc/merged/"
#
# for file in os.listdir(path):
#     print(path+file)
#     tracks = pd.read_csv(path+file)
#     tracks
#     break
#
#
# %%

# test_p = "../data/taxi_after_proc/merged/12-11_粤BD99105.csv"
# tracks = pd.read_csv(test_p)
# tracks
#
# %%

# m = folium.Map(location=[22.565050, 114.045616], zoom_start=14)
# test_p = "../data/taxi_after_proc/merged/12-11_粤BD99105.csv"
# tracks = pd.read_csv(test_p)
# # 时间跨度不超过10分钟
# tracks['gps_time'] = pd.to_datetime(tracks['gps_time'])
# tmp = pd.Timedelta(minutes=10)
# cut = [0]
# for i in range(len(tracks) - 1):
#     if (tracks['gps_time'][i + 1] - tracks['gps_time'][i]) > tmp:
#         cut.append(i)
#
# # print(cut)
#
# for i in range(len(cut) - 1):
#     a = tracks.iloc[cut[i] + 1:cut[i + 1] + 1]
#     track_list = a[['lat', 'lng']].values.tolist()
#     if len(track_list) == 0:
#         continue
#     folium.PolyLine(track_list).add_to(m)
#     folium.Marker(track_list[0],popup='start',icon=folium.Icon(color='red')).add_to(m)
#     folium.Marker(track_list[-1],popup='end',icon=folium.Icon(color='green')).add_to(m)
# #     for n,posi in enumerate(track_list):
# #         if(n==0):
# #             continue
# #         if(n==len(track_list)-1):
# #             continue
# #         folium.Marker(posi).add_to(m)
#
#
# folium.PolyLine(track_list).add_to(m)
#
# m.save("1.html")
# m
#
# %%

# # 一天轨迹的时间信息，
# test_p = "../data/taxi_after_proc/merged/12-11_粤BD99105.csv"
# tracks = pd.read_csv(test_p)
# t = tracks["gps_time"].values.tolist()
# t
#
# %%

# # 获取所有原始数据
# root_path = os.walk("../data/taxi")
# all_files = []
# for root,ds,fs in root_path:
#     days = []
#     for f in fs:
#         full_path = os.path.join(root,f)
#         days.append(full_path)
#     if len(days)==0:
#         continue
#     all_files.append(days)
# all_files.sort()
# all_files
#
# %%

# path = ['../data/taxi/2019-12-11/2019-12-11_5',
#   '../data/taxi/2019-12-11/2019-12-11_0',
#   '../data/taxi/2019-12-11/2019-12-11_2',
#   '../data/taxi/2019-12-11/2019-12-11_3',
#   '../data/taxi/2019-12-11/2019-12-11_1',
#   '../data/taxi/2019-12-11/2019-12-11_7',
#   '../data/taxi/2019-12-11/2019-12-11_8',
#   '../data/taxi/2019-12-11/2019-12-11_6',
#   '../data/taxi/2019-12-11/2019-12-11_4']
# path.sort()
# names=["sys_time", "license_number", "lng", "lat", "gps_time", "EMPTY1", "speed", "direction", "car_status","alarm_status",
#        "EMPTY2", "EMPTY3", "license_color", "recorder_speed", "mileage", "height", "EMPTY4"]
#
#
# records = []
# for each_time in tqdm(path):
#     records.append(pd.read_csv(each_time, names=names))
# #     records = records[(records["license_number"]=="粤BD99105")]
#
# #     print(records)
# #     break
# #     group = records.groupby("license_number")
# #     for each_lincense, each_record in group:
# #         posi = each_record[['lat', 'lng', 'gps_time', 'speed', 'direction', 'car_status']]
# #         # 按gps时间排序，并去重
# #         posi['gps_time'] = pd.to_datetime(posi['gps_time'])
# #         posi.sort_values(by="gps_time", inplace=True, ascending=True)
# #         posi = posi.drop_duplicates()
# #         # 去除错误日期的记录
# #         open_day = '2019-12-01'
# #         close_day = '2019-12-14'
# #         posi = posi[(posi['gps_time'] > open_day) & (posi['gps_time'] < close_day)]
# #         # 去除速度为0的记录
# #         posi = posi.loc[(posi != 0).all(axis=1)]
# #         # 去除数量过少的记录
# #         if len(posi) < 30:
# #             continue
# #         posi.to_csv(dest_path + each_time[-7:] + each_lincense + ".csv", index=False)
#
#
# %%

# car_records = []
# for each_record in tqdm(records):
#     each_record = each_record.drop("sys_time",axis='columns')
#     car_record = each_record[(each_record["license_number"]=="粤BD99105")]
#     car_record['gps_time'] = pd.to_datetime(car_record['gps_time'])
#     car_record.sort_values(by="gps_time", inplace=True, ascending=True)
#     car_record = car_record.drop_duplicates()
#     car_record = car_record[(car_record["lat"]!=0)]
#
#     car_records.append(car_record)
#
#
# %%

# # m = folium.Map(location=[22.565050, 114.045616], zoom_start=14)
# # print(car_records[0])
# all_len = 0
# all_records = pd.DataFrame()
# for record in car_records:
#     print(record)
#     all_records = all_records.append(record)
# #     t = record["gps_time"].tolist()
# all_records
# all_records.to_csv("粤BD99105_notprocess.csv")
# #     all_len += len(t)

# print(all_len)
#     track_list = record[['lat', 'lng']].values.tolist()
#     if len(track_list) == 0:
#         continue
#     folium.PolyLine(track_list).add_to(m)


# tracks['gps_time'] = pd.to_datetime(tracks['gps_time'])
# tmp = pd.Timedelta(minutes=10)
# cut = [0]
# for i in range(len(tracks) - 1):
#     if (tracks['gps_time'][i + 1] - tracks['gps_time'][i]) > tmp:
#         cut.append(i)

# for i in range(len(cut) - 1):
#     a = tracks.iloc[cut[i] + 1:cut[i + 1] + 1]
#     track_list = a[['lat', 'lng']].values.tolist()
#     if len(track_list) == 0:
#         continue
#     folium.PolyLine(track_list).add_to(m)
#     folium.Marker(track_list[0],popup='start',icon=folium.Icon(color='red')).add_to(m)
#     folium.Marker(track_list[-1],popup='end',icon=folium.Icon(color='green')).add_to(m)

# folium.PolyLine(track_list).add_to(m)

# m.save("2.html")
# m

# %%

# names=["sys_time", "license_number", "lng", "lat", "gps_time", "EMPTY1", "speed", "direction", "car_status","alarm_status",
#        "EMPTY2", "EMPTY3", "license_color", "recorder_speed", "mileage", "height", "EMPTY4"]

# dest_path = "../data/taxi_after_proc/proc_segment/"
# all_files.sort()
# car_path = '../data/taxi/2019-12-02/2019-12-02_1'
# car_path2 = '../data/taxi/2019-12-02/2019-12-02_2'

# df = []
# records1 = pd.read_csv(car_path,names = names)
# records2 = pd.read_csv(car_path2,names = names)
# df.append(records1)
# df.append(records2)
# df


# %%


# posi = records2[['license_number','lat', 'lng', 'gps_time', 'speed', 'direction', 'car_status']]
# print(len(posi))
# # 按gps时间排序，并去重
# posi['gps_time'] = pd.to_datetime(posi['gps_time'])
# posi.sort_values(by="gps_time", inplace=True, ascending=True)
# posi = posi.drop_duplicates()
# print(len(posi))

# # 去除错误日期的记录
# open_day = '2019-12-01'
# close_day = '2019-12-03'
# posi = posi[(posi['gps_time'] > open_day) & (posi['gps_time'] < close_day)]
# print(len(posi))
# print(posi.head(30))

# # 去除速度为0的记录
# posi = posi.loc[(posi != 0).all(axis=1)]
# print(len(posi))

# # 去除数量过少的记录


# %%

# names=["sys_time", "license_number", "lng", "lat", "gps_time", "EMPTY1", "speed", "direction", "car_status","alarm_status",
#        "EMPTY2", "EMPTY3", "license_color", "recorder_speed", "mileage", "height", "EMPTY4"]

# car_path = [
# "../data/taxi/2019-12-02/2019-12-02_0",
# "../data/taxi/2019-12-02/2019-12-02_1",
# "../data/taxi/2019-12-02/2019-12-02_2",
# "../data/taxi/2019-12-02/2019-12-02_3",
# "../data/taxi/2019-12-02/2019-12-02_4",
# "../data/taxi/2019-12-02/2019-12-02_5",
# "../data/taxi/2019-12-02/2019-12-02_6",
# "../data/taxi/2019-12-02/2019-12-02_7"
# ]
# df = []
# records = []
# for p in car_path:
#     records.append(pd.read_csv(p,names = names,nrows=20000))


# %%

# for records1 in records:
#     posi = records1[['license_number',  'gps_time', 'speed', 'direction', 'car_status']]
#     # 按gps时间排序，并去重
#     posi['gps_time'] = pd.to_datetime(posi['gps_time'])
#     posi.sort_values(by="gps_time", inplace=True, ascending=True)
#     posi = posi.drop_duplicates()

#     # 去除错误日期的记录
#     open_day = '2019-12-01'
#     close_day = '2019-12-03'
#     posi = posi[(posi['gps_time'] > open_day) & (posi['gps_time'] < close_day)]

#     # 去除速度为0的记录
#     posi = posi.loc[(posi != 0).all(axis=1)]
# #     print(posi.head(3))

# #     print(len(posi))
#     speed_avg = posi['speed'].mean()
#     print(speed_avg)
# #     print()

#     # 去除数量过少的记录


# %%

# car = [
# "../data/taxi_after_proc/proc_segment/12-02_0粤BD01819.csv",
# "../data/taxi_after_proc/proc_segment/12-02_1粤BD01819.csv",
# "../data/taxi_after_proc/proc_segment/12-02_2粤BD01819.csv",
# "../data/taxi_after_proc/proc_segment/12-02_3粤BD01819.csv",
# "../data/taxi_after_proc/proc_segment/12-02_4粤BD01819.csv",
# "../data/taxi_after_proc/proc_segment/12-02_5粤BD01819.csv",
# "../data/taxi_after_proc/proc_segment/12-02_6粤BD01819.csv",
# "../data/taxi_after_proc/proc_segment/12-02_7粤BD01819.csv"
# ]
# for each_time in car:
#     car_tmp = pd.read_csv(each_time)
# #     print(len(car_tmp))
#     print(car_tmp.head(10))
#     print(car_tmp['speed'].mean())

# %%

# for records1 in records:
#     posi = records1[['license_number',  'gps_time', 'speed', 'direction', 'car_status']]
#     # 按gps时间排序，并去重
#     posi['gps_time'] = pd.to_datetime(posi['gps_time'])
#     posi.sort_values(by="gps_time", inplace=True, ascending=True)
#     posi = posi.drop_duplicates()

#     # 去除错误日期的记录
#     open_day = '2019-12-01'
#     close_day = '2019-12-03'
#     posi = posi[(posi['gps_time'] > open_day) & (posi['gps_time'] < close_day)]

#     # 去除速度为0的记录
# #     posi = posi.loc[(posi != 0).all(axis=1)]
# #     print(posi.head(3))
# #     print(len(posi))
#     speed_list = posi['speed'].values.tolist()
#     plt.figure()
#     freq, bins, _ = plt.hist(speed_list, rwidth=0.8) # 参数 ，下限大一些，
#     plt.title('Velocity statistical analysis')
#     plt.xlabel('speed (km/h)')
#     plt.ylabel('amount of different speed')
#     plt.show()


# %%

# for records1 in records:
#     posi = records1[['license_number',  'gps_time', 'speed', 'direction', 'car_status']]
#     # 按gps时间排序，并去重
#     posi['gps_time'] = pd.to_datetime(posi['gps_time'])
#     posi.sort_values(by="gps_time", inplace=True, ascending=True)
#     posi = posi.drop_duplicates()

#     # 去除错误日期的记录
#     open_day = '2019-12-01'
#     close_day = '2019-12-03'
#     posi = posi[(posi['gps_time'] > open_day) & (posi['gps_time'] < close_day)]

#     # 去除速度为0的记录
# #     posi = posi.loc[(posi != 0).all(axis=1)]
#     posi = posi[(posi['speed']>0)&(posi['speed']<150)]
# #     print(posi.head(3))
# #     print(len(posi))
#     speed_cnt = posi['speed'].value_counts()

#     print(type(speed_cnt))
#     d = speed_cnt.to_dict()
#     d_list = sorted(list(d.items()),key=lambda x:x[0])
#     print(d_list)
#     y_max = d_list[-1][0]+1
#     y = list(range(y_max))
#     x = []
#     print(d[2])
#     for i in range(y_max):
#         if i in d:
#             x.append(d[i])
#         else:
#             x.append(0)
#     print(y)
#     print(x)
# #     y_max =
#     plt.figure()
#     plt.plot(y,x)
#     plt.title('Velocity statistical analysis')
#     plt.xlabel('speed (km/h)')
#     plt.ylabel('amount of different speed')
#     plt.show()
#     break


# %%


# %%

# for records1 in records:
#     posi = records1[['license_number',  'gps_time', 'speed']]
#     # 按gps时间排序，并去重
#     posi['gps_time'] = pd.to_datetime(posi['gps_time'])
#     posi.sort_values(by="gps_time", inplace=True, ascending=True)
#     posi = posi.drop_duplicates()

#     # 去除错误日期的记录
#     open_day = '2019-12-01'
#     close_day = '2019-12-03'
#     posi = posi[(posi['gps_time'] > open_day) & (posi['gps_time'] < close_day)]

#     # 去除速度为0的记录
# #     posi = posi.loc[(posi != 0).all(axis=1)]
# #     posi = posi[(posi['speed']>0)&(posi['speed']<150)]

# #     speed_list = posi['speed'].values.tolist()
#     speed_cnt = posi['speed'].value_counts()

#     print(type(speed_cnt))
#     d = speed_cnt.to_dict()
#     d_list = sorted(list(d.items()),key=lambda x:x[0])
#     print(d_list)
#     y_max = d_list[-1][0]+1
#     y = list(range(y_max))
#     x = []
#     print(d[2])
#     for i in range(y_max):
#         if i in d:
#             x.append(d[i])
#         else:
#             x.append(0)
#     print(y)
#     print(x)
# #     y_max =
#     plt.figure()
#     plt.plot(y,x)
#     plt.title('Velocity statistical analysis')
#     plt.xlabel('speed (km/h)')
#     plt.ylabel('amount of different speed')
#     plt.show()

#     break


# %%

# import matplotlib.pyplot as plt
# import matplotlib.dates as dates

# car = [
# "../data/taxi_after_proc/proc_segment/12-02_0粤BD01819.csv",
# "../data/taxi_after_proc/proc_segment/12-02_1粤BD01819.csv",
# "../data/taxi_after_proc/proc_segment/12-02_2粤BD01819.csv",
# "../data/taxi_after_proc/proc_segment/12-02_3粤BD01819.csv",
# "../data/taxi_after_proc/proc_segment/12-02_4粤BD01819.csv",
# "../data/taxi_after_proc/proc_segment/12-02_5粤BD01819.csv",
# "../data/taxi_after_proc/proc_segment/12-02_6粤BD01819.csv",
# "../data/taxi_after_proc/proc_segment/12-02_7粤BD01819.csv"
# ]
# x = []
# y = []
# for each_time in car:
#     tracks = pd.read_csv(each_time)
# #     print(len(car_tmp))
#     # 时间跨度不超过10分钟
#     tracks['gps_time'] = pd.to_datetime(tracks['gps_time'])
#     x.append(tracks['gps_time'].values.tolist())
#     y.append(tracks['speed'].values.tolist())
#     break
#     # tracks['gps_time'] = tracks
#     # print(tracks)
#     # print(tracks.iloc[0:3])

# xfmt = mdates.DateFormatter('%y-%m-%d %H:%M')
# ax.xaxis.set_major_formatter(xfmt)

# plt.plot(x,y)
# plt.show()
# print(x)
# print(y)

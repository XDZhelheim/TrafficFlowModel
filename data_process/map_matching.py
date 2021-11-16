import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import os
import multiprocessing as mp
import ctypes

# global vars

df_roads=gpd.GeoDataFrame(pd.read_pickle("../data/df_main_roads.pkl"))

def to_shared_array(arr, ctype):
    shared_array = mp.Array(ctype, arr.size, lock=False)
    temp = np.frombuffer(shared_array, dtype=arr.dtype)
    temp[:] = arr.flatten(order='C')
    return shared_array

def to_numpy_array(shared_array, shape):
    '''Create a numpy array backed by a shared memory Array.'''
    arr = np.ctypeslib.as_array(shared_array)
    return arr.reshape(shape)

lock=mp.Lock()
np_mat=np.zeros((9, len(df_roads), len(df_roads)), dtype=np.int32)
shared_mat=to_shared_array(np_mat, ctypes.c_int32)
trans_matrix=to_numpy_array(shared_mat, np_mat.shape)

def get_distance(row, midpoint):
    row["distance"]=abs(row["center"][0]-midpoint[1])+abs(row["center"][1]-midpoint[0])

    return row

def get_seconds(row):
    row["time_seconds"]=row["gps_time"].hour*3600+row["gps_time"].minute*60+row["gps_time"].second
    return row

def resample_coords(df_taxi):
    index_drop_list=[]
    last_time=df_taxi.iloc[0]["time_seconds"]-30
    for index, row in df_taxi.iterrows():
        if row["time_seconds"]-last_time<30:
            index_drop_list.append(index)
            continue
        last_time=row["time_seconds"]
    df_taxi=df_taxi.drop(index_drop_list)
    coords=df_taxi[["lat", "lng"]].values
    time_list=df_taxi["time_seconds"].values

    return df_taxi, coords, time_list

def get_track(file_path):
    global df_roads

    df_taxi=pd.read_csv(file_path)
    df_taxi["gps_time"]=pd.to_datetime(df_taxi["gps_time"])
    df_taxi=df_taxi.apply(get_seconds, axis=1)

    df_taxi, coords, time_list=resample_coords(df_taxi)
    midpoint=coords[int(len(coords)/2)]

    df_roads_temp=df_roads.apply(get_distance, args=(midpoint,), axis=1)
    df_roads_temp=df_roads_temp.sort_values("distance")

    index_list=df_roads_temp.index.values
    geom_list=df_roads_temp["geometry"].values

    track=[]
    last_road=-1
    for i in range(len(coords)):
        coord=coords[i]
        find=False
        for j in range(len(geom_list)):
            if geom_list[j].contains(Point(coord[1], coord[0])):
                road=index_list[j]
                track.append(road)
                if last_road!=-1:
                    with lock:
                        trans_matrix[0, last_road, road]+=1
                        trans_matrix[int(time_list[i]/10800)+1, last_road, road]+=1
                        # print(last_road, road, trans_matrix[last_road, road])
                find=True
                last_road=road
                break
        if not find:
            track.append(None)

    df_taxi["track"]=track
    df_taxi.to_csv("../data/taxi_after_proc/tracks/{}".format(file_path.split("/")[-1]), index=False)
    
    print("../data/taxi_after_proc/tracks/{}".format(file_path.split("/")[-1]))

N=30

if __name__ == "__main__":
    pool=mp.Pool(32)

    for file in os.listdir("../data/taxi_after_proc/merged/"):
        file_path="../data/taxi_after_proc/merged/"+file
        date=int(file.split("-")[1].split("_")[0])
        if date>=2 and date<=7:
            res=pool.apply_async(get_track, (file_path,))

    pool.close()
    pool.join()

    np.save("../data/trans_matrix.npy", trans_matrix)

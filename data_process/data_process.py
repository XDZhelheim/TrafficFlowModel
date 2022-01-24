import numpy as np
import pandas as pd
import geopandas as gpd
import os
from tqdm import tqdm
import networkx as nx
import osmnx as ox
import json

DATA_PATH = "../data/"
TAXI_DATA_PATH = "../data/taxi_after_proc/merged"

MIN_LAT = 22.5310
MAX_LAT = 22.5397
MIN_LNG = 114.0442
MAX_LNG = 114.0633

START_DAY = 2
END_DAY = 6

DOWNSAMPLING_INTERVAL = 30
TRAJ_SPLIT_INTERVAL = 600
FLOW_AGG_INTERVAL_MINUTE = 15

N = -1  # num of roads


def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8"):
    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "nodes.shp")
    filepath_edges = os.path.join(filepath, "edges.shp")

    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)

    # We need an unique ID for each edge
    # gdf_edges["fid"] = gdf_edges.index.to_numpy()
    gdf_edges["fid"] = np.arange(
        gdf_edges.shape[0])  # https://github.com/cyang-kth/fmm/issues/166

    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)


def download_osm():
    # download by osmnx, print basic stats
    graph_drive = ox.graph_from_bbox(MIN_LAT,
                                     MAX_LAT,
                                     MIN_LNG,
                                     MAX_LNG,
                                     network_type="drive")
    ox.basic_stats(graph_drive)

    # convert to line graph
    line_graph_drive = nx.line_graph(graph_drive)
    global N
    N = len(line_graph_drive.nodes)

    # save .pkl
    nx.write_gpickle(graph_drive,
                     path=os.path.join(DATA_PATH, "graph_drive.pkl"))
    nx.write_gpickle(line_graph_drive,
                     path=os.path.join(DATA_PATH, "line_graph_drive.pkl"))

    # shapefiles for fmm
    save_graph_shapefile_directional(graph_drive,
                                     filepath=os.path.join(
                                         DATA_PATH, "fmm_data"))


def contains(lat, lng):
    return lat >= MIN_LAT and lat <= MAX_LAT and lng >= MIN_LNG and lng <= MAX_LNG


def gen_fmm_dataset():
    gps_file = open(os.path.join(DATA_PATH, "fmm_data", "gps.csv"), "w")
    gps_file.write("id;x;y;time\n")

    timedelta_downsampling = pd.Timedelta(seconds=DOWNSAMPLING_INTERVAL)
    timedelta_traj_split = pd.Timedelta(seconds=TRAJ_SPLIT_INTERVAL)

    traj_counter = 0
    for taxi_file in tqdm(sorted(os.listdir(TAXI_DATA_PATH))):
        date = int(taxi_file.split("_")[0].split("-")[1])
        if date < START_DAY or date > END_DAY:
            continue
        if os.path.getsize(os.path.join(TAXI_DATA_PATH, taxi_file)) < 100:
            continue
        df_taxi = pd.read_csv(os.path.join(TAXI_DATA_PATH, taxi_file),
                              parse_dates=["gps_time"])
        if df_taxi.empty:
            continue

        line_buffer = []
        last_time = df_taxi.iloc[0]["gps_time"] + pd.Timedelta(
            seconds=-TRAJ_SPLIT_INTERVAL)
        for row in df_taxi.itertuples():
            if not contains(row[1], row[2]):
                continue
            if row[3] - last_time < timedelta_downsampling:  # resample: drop <30s
                continue
            if row[3] - last_time > timedelta_traj_split:
                if len(line_buffer) > 1:  # only store length>1 traj
                    gps_file.write("".join(line_buffer))
                    traj_counter += 1
                line_buffer = []

            last_time = row[3]

            line_buffer.append(f"{traj_counter};{row[2]};{row[1]};{row[3]}\n")

    gps_file.close()


def run_fmm():
    os.system(
        "ubodt_gen --network {} --network_id fid --source u --target v --output {} --delta 0.03 --use_omp"
        .format(os.path.join(DATA_PATH, "fmm_data", "edges.shp"),
                os.path.join(DATA_PATH, "fmm_data", "ubodt.txt")))
    os.system(
        "fmm --ubodt {} --network {} --network_id fid --source u --target v --gps {} --gps_point -k 8 -r 0.003 -e 0.0005 --output {} --use_omp --output_fields id,opath,cpath,mgeom > {} 2>&1"
        .format(os.path.join(DATA_PATH, "fmm_data", "ubodt.txt"),
                os.path.join(DATA_PATH, "fmm_data", "edges.shp"),
                os.path.join(DATA_PATH, "fmm_data", "gps.csv"),
                os.path.join(DATA_PATH, "fmm_data", "mr.txt"),
                os.path.join(DATA_PATH, "fmm_data", "fmm.log")))


def fmm2geo():
    df_edges = gpd.GeoDataFrame.from_file(
        os.path.join(DATA_PATH, "fmm_data", "edges.shp"))

    df_geo = pd.DataFrame()

    df_geo["geo_id"] = df_edges["fid"]
    df_geo["type"] = "LineString"
    df_geo["coordinates"] = df_edges["geometry"].apply(
        lambda x: list(x.coords))

    df_geo.to_csv(os.path.join(DATA_PATH, "sz_taxi", "sz_taxi.geo"),
                  index=False)


def fmm2rel():
    df_edges = gpd.GeoDataFrame.from_file(
        os.path.join(DATA_PATH, "fmm_data", "edges.shp"))
    line_graph = nx.read_gpickle(
        os.path.join(DATA_PATH, "line_graph_drive.pkl"))

    df_right = df_edges[["u", "v", "key", "fid"]]

    unziped_node_list = list(zip(*list(line_graph.nodes)))
    df_left = pd.DataFrame()
    df_left["u"] = unziped_node_list[0]
    df_left["v"] = unziped_node_list[1]
    df_left["key"] = unziped_node_list[2]
    df_left["id"] = range(len(unziped_node_list[0]))

    df_join = pd.merge(df_left, df_right, on=["u", "v", "key"])

    adj = nx.convert_matrix.to_numpy_array(line_graph)

    rel = []
    rel_id_counter = 0

    for i in range(len(adj)):
        for j in range(len(adj[0])):
            if adj[i][j]:
                rel.append([
                    rel_id_counter, "geo", df_join.iloc[i]["fid"],
                    df_join.iloc[j]["fid"]
                ])
                rel_id_counter += 1

    df_rel = pd.DataFrame(
        rel, columns=["rel_id", "type", "origin_id", "destination_id"])

    df_rel.to_csv(os.path.join(DATA_PATH, "sz_taxi", "sz_taxi.rel"),
                  index=False)


def fmm2dyna():
    df_fmm_res = pd.read_csv(os.path.join(DATA_PATH, "fmm_data", "mr.txt"),
                             sep=";")
    df_fmm_data = pd.read_csv(os.path.join(DATA_PATH, "fmm_data", "gps.csv"),
                              sep=";",
                              parse_dates=["time"])

    dyna_file = open(os.path.join(DATA_PATH, "sz_taxi", "sz_taxi.dyna"), "w")
    dyna_file.write("dyna_id,type,time,entity_id,flow\n")

    assert (N != -1)
    flow_matrix = np.zeros((5, 24 * 60 // FLOW_AGG_INTERVAL_MINUTE, N),
                           dtype=np.int16)

    for traj_id in tqdm(df_fmm_res["id"].values):
        time_list = df_fmm_data.loc[df_fmm_data["id"] ==
                                    traj_id]["time"].values
        road_list = np.array(df_fmm_res.loc[df_fmm_res["id"] == traj_id]
                             ["opath"].values[0].split(","),
                             dtype=np.int16)

        assert (len(time_list) == len(road_list))

        for i in range(len(road_list)):
            time_i = pd.to_datetime(time_list[i])
            day = time_i.day
            mins = time_i.hour * 60 + time_i.minute

            flow_matrix[day - 2][mins // 15][road_list[i]] += 1

    dyna_id_counter = 0
    for day in range(flow_matrix.shape[0]):
        for interval in range(flow_matrix.shape[1]):
            for road in range(flow_matrix.shape[2]):
                dyna_file.write(
                    f"{dyna_id_counter},state,2019-12-0{day+2}T{str(interval*15//60).zfill(2)}:{str((interval%4)*15).zfill(2)}:00Z,{road},{flow_matrix[day][interval][road]}\n"
                )
                dyna_id_counter += 1

    dyna_file.close()


def gen_config():
    config = {}

    config["geo"] = {}
    config["geo"]["including_types"] = ["LineString"]
    config["geo"]["LineString"] = {}

    config["rel"] = {}
    config["rel"]["including_types"] = ["geo"]
    config["rel"]["geo"] = {}

    config["dyna"] = {}
    config["dyna"]["including_types"] = ["state"]
    config["dyna"]["state"] = {"entity_id": "geo_id", "flow": "num"}

    config["info"] = {}
    config["info"]["data_files"] = "sz_taxi"
    config["info"]["geo_file"] = "sz_taxi"
    config["info"]["rel_file"] = "sz_taxi"
    config["info"]["data_col"] = ["flow"]
    config["info"]["output_dim"] = 1
    config["info"]["time_intervals"] = 60 * FLOW_AGG_INTERVAL_MINUTE
    config["info"][
        "init_weight_inf_or_zero"] = "zero"  # adj matrix not connected: 0 (inf: infinity)
    config["info"][
        "set_weight_link_or_dist"] = "link"  # adj matrix 01 (dist: use weight)
    config["info"]["calculate_weight_adj"] = False
    # config["info"]["weight_adj_epsilon"]=0.1 # disabled when the above is false

    json.dump(config,
              open(os.path.join(DATA_PATH, "sz_taxi", "config.json"),
                   "w",
                   encoding="utf-8"),
              ensure_ascii=False)


def fmm2atom():
    fmm2geo()
    fmm2rel()
    fmm2dyna()
    gen_config()

if __name__ == "__main__":
    if os.path.split(os.path.dirname(os.path.realpath(__file__)))[-1] != "data_process":
        print("Wrong working dir")
        exit(1)
    
    download_osm()
    gen_fmm_dataset()
    run_fmm()
    fmm2atom()
    
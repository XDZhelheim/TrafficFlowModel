import numpy as np
import pandas as pd
import geopandas as gpd
import os
import networkx as nx
import osmnx as ox
import json
import time
import argparse

DATA_PATH = "../../data/"
TAXI_DATA_PATH = "../../data/taxi_after_proc/clean202006"
DATASET = "sz_taxi_202006"

MIN_LAT = 22.5311
MAX_LAT = 22.5517
MIN_LNG = 114.0439
MAX_LNG = 114.0633

DATE_PREFIX = "2020-06-"
START_DAY = 1
END_DAY = 30

DOWNSAMPLING_INTERVAL = 30
TRAJ_SPLIT_INTERVAL = 600
FLOW_AGG_INTERVAL_MINUTE = 5

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
                     path=os.path.join(DATA_PATH, DATASET, "graph_drive.pkl"))
    nx.write_gpickle(line_graph_drive,
                     path=os.path.join(DATA_PATH, DATASET,
                                       "line_graph_drive.pkl"))

    # shapefiles for fmm
    save_graph_shapefile_directional(graph_drive,
                                     filepath=os.path.join(
                                         DATA_PATH, DATASET, f"fmm_{DATASET}"))


def contains(lat, lng):
    return lat >= MIN_LAT and lat <= MAX_LAT and lng >= MIN_LNG and lng <= MAX_LNG


def gen_fmm_dataset():
    gps_file = open(
        os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}", "gps.csv"), "w")
    gps_file.write("id;x;y;time\n")

    timedelta_downsampling = pd.Timedelta(seconds=DOWNSAMPLING_INTERVAL)
    timedelta_traj_split = pd.Timedelta(seconds=TRAJ_SPLIT_INTERVAL)

    traj_counter = 0
    for taxi_file in sorted(os.listdir(TAXI_DATA_PATH)):
        date = int(taxi_file.split("_")[0].split("-")[1])
        if date < START_DAY or date > END_DAY:
            continue
        if os.path.getsize(os.path.join(TAXI_DATA_PATH, taxi_file)) < 100:
            continue
        # df_taxi = pd.read_csv(os.path.join(TAXI_DATA_PATH, taxi_file), parse_dates=["gps_time"])
        df_taxi = pd.read_pickle(os.path.join(TAXI_DATA_PATH, taxi_file))
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
        .format(
            os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}", "edges.shp"),
            os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}", "ubodt.txt")))
    os.system(
        "fmm --ubodt {} --network {} --network_id fid --source u --target v --gps {} --gps_point -k 8 -r 0.003 -e 0.0005 --output {} --use_omp --output_fields id,opath,cpath,mgeom > {} 2>&1"
        .format(
            os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}", "ubodt.txt"),
            os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}", "edges.shp"),
            os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}", "gps.csv"),
            os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}", "mr.txt"),
            os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}", "fmm.log")))


def fmm2geo():
    df_edges = gpd.GeoDataFrame.from_file(
        os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}", "edges.shp"))

    df_geo = pd.DataFrame()

    df_geo["geo_id"] = df_edges["fid"]
    df_geo["type"] = "LineString"
    df_geo["coordinates"] = df_edges["geometry"].apply(
        lambda x: list(x.coords))

    df_geo.to_csv(os.path.join(DATA_PATH, DATASET, f"{DATASET}.geo"),
                  index=False)


def fmm2rel():
    df_edges = gpd.GeoDataFrame.from_file(
        os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}", "edges.shp"))
    line_graph_drive = nx.read_gpickle(
        os.path.join(DATA_PATH, DATASET, "line_graph_drive.pkl"))

    df_right = df_edges[["u", "v", "key", "fid"]]

    unziped_node_list = list(zip(*list(line_graph_drive.nodes)))
    df_left = pd.DataFrame()
    df_left["u"] = unziped_node_list[0]
    df_left["v"] = unziped_node_list[1]
    df_left["key"] = unziped_node_list[2]
    df_left["id"] = range(len(unziped_node_list[0]))

    df_join = pd.merge(df_left, df_right, on=["u", "v", "key"])

    adj = nx.convert_matrix.to_numpy_array(line_graph_drive)

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

    df_rel.to_csv(os.path.join(DATA_PATH, DATASET, f"{DATASET}.rel"),
                  index=False)


def fmm2dyna():
    df_fmm_res = pd.read_csv(os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}",
                                          "mr.txt"),
                             sep=";")
    df_fmm_data = pd.read_csv(os.path.join(DATA_PATH, DATASET,
                                           f"fmm_{DATASET}", "gps.csv"),
                              sep=";",
                              parse_dates=["time"])

    dyna_file = open(
        os.path.join(DATA_PATH, DATASET,
                     f"{DATASET}_{FLOW_AGG_INTERVAL_MINUTE}min.dyna"), "w")
    dyna_file.write("dyna_id,type,time,entity_id,flow\n")

    assert (N != -1)
    flow_matrix = np.zeros(
        (END_DAY - START_DAY + 1, 24 * 60 // FLOW_AGG_INTERVAL_MINUTE, N),
        dtype=np.int16)

    for traj_id in df_fmm_res["id"].values:
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

            flow_matrix[day -
                        START_DAY][mins //
                                   FLOW_AGG_INTERVAL_MINUTE][road_list[i]] += 1

    dyna_id_counter = 0
    for day in range(flow_matrix.shape[0]):
        for interval in range(flow_matrix.shape[1]):
            for road in range(flow_matrix.shape[2]):
                dyna_file.write(
                    f"{dyna_id_counter}," + "state," +
                    f"{DATE_PREFIX}{str(day+START_DAY).zfill(2)}T{str(interval*FLOW_AGG_INTERVAL_MINUTE//60).zfill(2)}:{str((interval%(60//FLOW_AGG_INTERVAL_MINUTE))*FLOW_AGG_INTERVAL_MINUTE).zfill(2)}:00Z,"
                    + f"{road}," + f"{flow_matrix[day][interval][road]}\n")
                dyna_id_counter += 1

    dyna_file.close()

    np.save(
        os.path.join(DATA_PATH, DATASET,
                     f"{DATASET}_{FLOW_AGG_INTERVAL_MINUTE}min.npy"),
        flow_matrix)


def gen_time_seq(start_time, end_time, length):
    assert (end_time > start_time)

    start_time = start_time.view(np.int64)
    end_time = end_time.view(np.int64)

    seq = []
    step = (end_time - start_time) // (length + 1)
    for i in range(length):
        seq.append(start_time + step * (i + 1))

    return list(np.array(seq, dtype="datetime64[ns]"))


def convert_path(row):
    row["opath"] = np.array(row["opath"].split(","), dtype=np.int16)
    row["cpath"] = np.array(row["cpath"].split(","), dtype=np.int16)

    return row


def fmm2dyna_recovery():
    df_fmm_res = pd.read_csv(os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}",
                                          "mr.txt"),
                             sep=";").set_index("id").dropna()
    df_fmm_res = df_fmm_res.apply(convert_path, axis=1)
    df_fmm_data = pd.read_csv(os.path.join(DATA_PATH, DATASET,
                                           f"fmm_{DATASET}", "gps.csv"),
                              sep=";",
                              parse_dates=["time"])

    dyna_file = open(
        os.path.join(
            DATA_PATH, DATASET,
            f"{DATASET}_{FLOW_AGG_INTERVAL_MINUTE}min_recovered.dyna"), "w")
    dyna_file.write(
        "dyna_id,type,time,entity_id,flow\n")

    flow_matrix = np.zeros(
        (END_DAY - START_DAY + 1, 24 * 60 // FLOW_AGG_INTERVAL_MINUTE, N),
        dtype=np.int16)

    for traj_id in df_fmm_res.index:
        time_list = df_fmm_data.loc[df_fmm_data["id"] ==
                                    traj_id]["time"].values
        opath_list = df_fmm_res.loc[traj_id, "opath"]
        cpath_list = df_fmm_res.loc[traj_id, "cpath"]

        # drop duplicates
        del_indexes = []
        for i in range(len(time_list) - 1):
            if opath_list[i] == opath_list[i + 1]:
                del_indexes.append(i + 1)

        opath_list = np.delete(opath_list, del_indexes)
        time_list = np.delete(time_list, del_indexes)

        assert (len(opath_list) == len(time_list))
        assert (opath_list[-1] == cpath_list[-1])

        # generate recovered time
        i = 0
        j = 0
        recovered_time_list = []

        while True:
            if i > len(opath_list) - 1:
                break
            if opath_list[i] == cpath_list[j]:
                recovered_time_list.append(time_list[i])
                i += 1
                j += 1
            else:
                length = 0
                while opath_list[i] != cpath_list[j]:
                    j += 1
                    length += 1
                recovered_time_list.extend(
                    gen_time_seq(time_list[i - 1], time_list[i], length))
        cpath_list = cpath_list[:j] # fmm bug: trailing
        assert (len(recovered_time_list) == len(cpath_list))

        for i in range(len(cpath_list)):
            time_i = pd.to_datetime(recovered_time_list[i])
            day = time_i.day
            mins = time_i.hour * 60 + time_i.minute

            flow_matrix[day - START_DAY][mins // FLOW_AGG_INTERVAL_MINUTE][
                cpath_list[i]] += 1

    dyna_id_counter = 0
    for day in range(flow_matrix.shape[0]):
        if day == 18 or day == 19:
            continue
        for interval in range(flow_matrix.shape[1]):
            for road in range(flow_matrix.shape[2]):
                dyna_file.write(
                    f"{dyna_id_counter}," + "state," +
                    f"{DATE_PREFIX}{str(day+START_DAY).zfill(2)}T{str(interval*FLOW_AGG_INTERVAL_MINUTE//60).zfill(2)}:{str((interval%(60//FLOW_AGG_INTERVAL_MINUTE))*FLOW_AGG_INTERVAL_MINUTE).zfill(2)}:00Z,"
                    + f"{road}," + f"{flow_matrix[day][interval][road]}\n")
                dyna_id_counter += 1

    dyna_file.close()

    np.save(
        os.path.join(DATA_PATH, DATASET,
                     f"{DATASET}_{FLOW_AGG_INTERVAL_MINUTE}min_recovered.npy"),
        flow_matrix)


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
    config["info"][
        "data_files"] = f"{DATASET}_{FLOW_AGG_INTERVAL_MINUTE}min_recovered"
    config["info"]["geo_file"] = DATASET
    config["info"]["rel_file"] = DATASET
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
              open(os.path.join(DATA_PATH, DATASET, "config.json"),
                   "w",
                   encoding="utf-8"),
              ensure_ascii=False)


def fmm2atom():
    fmm2geo()
    fmm2rel()
    fmm2dyna()
    gen_config()


def notify(msg):
    import datetime
    channel = "J0budaR2THarZw0OqS5O"
    notify_url = f"https://notify.run/{channel}"
    massage = f"{msg} | {str(datetime.datetime.now())}"
    os.system(f'curl {notify_url} -d "{massage}"')


def convert_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("%02d:%02d:%02d" % (h, m, s))


def run1step(func, log_info):
    print(log_info)
    notify(log_info)

    start = time.time()
    func()
    end = time.time()
    print("Time cost:", convert_time(end - start))
    notify("Finish " + log_info)


if __name__ == "__main__":
    # if os.path.split(os.path.dirname(
    #         os.path.realpath(__file__)))[-1] != "data_process":
    #     print("Wrong working dir.")
    #     exit(1)

    if not os.path.exists(os.path.join(DATA_PATH, DATASET)):
        os.mkdir(os.path.join(DATA_PATH, DATASET))

    if not os.path.exists(os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}")):
        os.mkdir(os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--download_osm", action="store_true")
    parser.add_argument("--gen_fmm", action="store_true")
    parser.add_argument("--run_fmm", action="store_true")
    parser.add_argument("--gen_atom", action="store_true")
    parser.add_argument("--all", action="store_true")

    parser.add_argument("-N", type=int, required=False)
    args = parser.parse_args()

    if args.N:
        N = args.N

    if args.all:
        if not os.path.exists(
                os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}",
                             "edges.shp")):
            run1step(download_osm, "Step1: Downloading OSM Graph.")
        if not os.path.exists(
                os.path.join(DATA_PATH, DATASET, f"fmm_{DATASET}", "mr.txt")):
            run1step(gen_fmm_dataset, "Step2: Generating FMM Dataset.")
        if not os.path.exists(
                os.path.join(DATA_PATH, DATASET, f"{DATASET}.geo")):
            run1step(run_fmm, "Step3: Running FMM.")
        if not os.path.exists(os.path.join(DATA_PATH, DATASET, "config.json")):
            run1step(fmm2atom, "Step4: Converting to atom files.")

    else:
        if args.download_osm:
            run1step(download_osm, "Downloading OSM Graph.")
        if args.gen_fmm:
            run1step(gen_fmm_dataset, "Generating FMM Dataset.")
        if args.run_fmm:
            run1step(run_fmm, "Running FMM.")
        if args.gen_atom:
            run1step(fmm2atom, "Converting to atom files.")

    print("Done.")

# TrafficFlowModel

**TODO**
1. 调参：GPS 筛选的边界范围, trajectory split interval
2. 设阈值 < n 则清零
3. 原数据中 GPS 在变但是速度一直=0的异常值要不要算进 flow

**Data Processing Pipeline** [data_process/data_process.py](./data_process/data_process.py)

1. [data_process/gen_road_network_small.ipynb](./data_process/gen_road_network_small.ipynb)

    Input:
    * 4 bbox boundary values of graph (max, min for lat and lng)

    Output:
    * data/fmm_data/\<shapefiles for FMM\>
    * data/\<district\>_graph.pkl
    * data/\<district\>_line_graph.pkl

2. [data_process/fmm_dataset.ipynb](./data_process/fmm_dataset.ipynb)

    Input:
    * data/taxi_after_proc/merged/*
    * bbox boundaries of graph

    Output:
    * data/fmm_data/gps.csv
    * data/fmm_data/ubodt.txt
    * data/fmm_data/mr.txt
    * data/fmm_data/fmm.log

3. [data_process/fmm2atom.ipynb](./data_process/fmm2atom.ipynb)

    Input:
    * data/fmm_data/edges.shp -> .geo .rel
    * data/\<district\>_line_graph.pkl -> .rel
    * data/fmm_data/gps.csv -> .dyna
    * data/fmm_data/mr.txt -> .dyna

    Output
    * LibCity atom files: data/libcity_atom/*

**Baseline**

Suppose LibCity is at `~/Bigscity-LibCity`.

Suppose conda environment for LibCity is `torch1.7`.

```bash
model_list=(GRU LSTM FNN AutoEncoder \
Seq2Seq ASTGCN MSTGCN AGCRN CONVGCN STSGCN ToGCN ResLSTM DGCN DSAN STNN \
DCRNN STGCN GWNET MTGNN)
```

Run:
```bash
./scripts/run_baseline.sh <GPU ID> <Max Epoch> <Batch Size>
```

Logs are at `Bigscity-LibCity/log`.

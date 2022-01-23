#! /bin/bash

model_list=(GRU LSTM FNN AutoEncoder \
Seq2Seq ASTGCN MSTGCN AGCRN CONVGCN STSGCN ToGCN ResLSTM CRANN Multi-STGCnet DGCN DSAN STNN \
DCRNN STGCN GWNET MTGNN)

cd ~/Bigscity-LibCity

source $(conda info --base)/etc/profile.d/conda.sh
conda activate torch1.7

if [[ -z "$1" ]]; then
    echo "$(tput setaf 1)Must specify GPU ID.$(tput sgr0)"
    exit 1
else
    echo "$(tput setaf 6)Using GPU ID=$1.$(tput sgr0)"
fi

if [ ! -d "log" ]; then
    mkdir log
fi

for model in "${model_list[@]}"; do
    start=$(date +%s)

    echo "$(tput setaf 6)Start ${model}.$(tput sgr0)"
    python run_model.py --task traffic_state_pred --model ${model} --dataset sz_taxi --gpu_id $1 > ./log/flow_${model}.log 2>&1 &
    wait $!

    end=$(date +%s)
    time=$((end-start))

    echo "$(tput setaf 2)Finished ${model}, duration ${time}s.$(tput sgr0)"
done

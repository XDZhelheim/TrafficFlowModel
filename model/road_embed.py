import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import datetime
import sys
import argparse

DATA_PATH = "../data/sz_taxi_202006/"
SEQ_LEN = 5
NUM_ROADS = 492

GPU_ID = 2
DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")


def gen_xy(traj_list, seq_len):
    """
    Generate inputs and targets for traj next-hop prediction.
    
    Parameter
    ---
    traj_list: list of traj
    ```
    [
      [246, 0, 70, 316, 246, 0, 70],
      [265, 264, 261, 259, 255, 8, 60, 61, 111, 115, 79, 80, 81, 82, 164, 414],
      ...
    ]
    ```
    
    Returns
    ---
    x: (num_samples, seq_len)
    y: (num_samples,) 1-d vec for labels
    """

    x, y = [], []
    for traj in traj_list:
        for i in range(len(traj) - seq_len):
            x.append(traj[i : i + seq_len])
            y.append(traj[i + seq_len])

    return torch.LongTensor(x), torch.LongTensor(y)


def get_dataloaders(traj_list, seq_len, train_size=0.7, val_size=0.1, batch_size=256):
    """
    Parameters
    ---
    traj_list: list of traj
    """
    np.random.shuffle(traj_list)

    split1 = int(len(traj_list) * train_size)
    split2 = int(len(traj_list) * (train_size + val_size))

    train_data = traj_list[:split1]
    val_data = traj_list[split1:split2]
    test_data = traj_list[split2:]

    x_train, y_train = gen_xy(train_data, seq_len)
    x_val, y_val = gen_xy(val_data, seq_len)
    x_test, y_test = gen_xy(test_data, seq_len)

    print(f"Trainset:\tx-{x_train.size()}\ty-{y_train.size()}")
    print(f"Valset:  \tx-{x_val.size()}  \ty-{y_val.size()}")
    print(f"Testset:\tx-{x_test.size()}\ty-{y_test.size()}")

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    valset = torch.utils.data.TensorDataset(x_val, y_val)
    testset = torch.utils.data.TensorDataset(x_test, y_test)

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=True
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader


@torch.no_grad()
def onehot_decode(label):
    return torch.argmax(label, dim=1)


@torch.no_grad()
def accuracy(predictions, targets):
    pred_decode = onehot_decode(predictions)
    true_decode = targets

    assert len(pred_decode) == len(true_decode)

    acc = torch.mean((pred_decode == true_decode).float())

    return float(acc)


@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    batch_acc_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model.forward(x_batch)
        loss = criterion.forward(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        acc = accuracy(out_batch, y_batch)
        batch_acc_list.append(acc)

    return np.mean(batch_loss_list), np.mean(batch_acc_list)


def train_one_epoch(model, trainset_loader, optimizer, criterion):
    model.train()
    batch_loss_list = []
    batch_acc_list = []
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model.forward(x_batch)
        loss = criterion.forward(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        acc = accuracy(out_batch, y_batch)
        batch_acc_list.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(batch_loss_list), np.mean(batch_acc_list)


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    criterion,
    max_epochs=100,
    early_stop=10,
    verbose=1,
    plot=False,
    log="train.log",
):
    if log:
        log = open(log, "a")
        log.seek(0)
        log.truncate()

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(
            model, trainset_loader, optimizer, criterion
        )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        val_loss, val_acc = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        if (epoch + 1) % verbose == 0:
            print(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                "\tTrain Loss = %.5f" % train_loss,
                "Train acc = %.5f " % train_acc,
                "Val Loss = %.5f" % val_loss,
                "Val acc = %.5f " % val_acc,
            )

            if log:
                print(
                    datetime.datetime.now(),
                    "Epoch",
                    epoch + 1,
                    "\tTrain Loss = %.5f" % train_loss,
                    "Train acc = %.5f " % train_acc,
                    "Val Loss = %.5f" % val_loss,
                    "Val acc = %.5f " % val_acc,
                    file=log,
                )
                log.flush()

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = model.state_dict()
        else:
            wait += 1
            if wait >= early_stop:
                print(f"Early stopping at epoch: {epoch+1}")
                print(f"Best at epoch {best_epoch+1}:")
                print(
                    "Train Loss = %.5f" % train_loss_list[best_epoch],
                    "Train acc = %.5f " % train_acc_list[best_epoch],
                )
                print(
                    "Val Loss = %.5f" % val_loss_list[best_epoch],
                    "Val acc = %.5f " % val_acc_list[best_epoch],
                )

                if log:
                    print(f"Early stopping at epoch: {epoch+1}", file=log)
                    print(f"Best at epoch {best_epoch+1}:", file=log)
                    print(
                        "Train Loss = %.5f" % train_loss_list[best_epoch],
                        "Train acc = %.5f " % train_acc_list[best_epoch],
                        file=log,
                    )
                    print(
                        "Val Loss = %.5f" % val_loss_list[best_epoch],
                        "Val acc = %.5f " % val_acc_list[best_epoch],
                        file=log,
                    )
                    log.flush()
                break

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.plot(range(0, epoch + 1), train_acc_list, "-", label="Train Acc")
        plt.plot(range(0, epoch + 1), val_acc_list, "-", label="Val Acc")
        plt.title("Epoch-Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    if log:
        log.close()

    torch.save(best_state_dict, "./saved/best_state_dict.pkl")
    model.load_state_dict(best_state_dict)
    return model


class DontKnowWhat2EatNN(torch.nn.Module):
    def __init__(
        self,
        embed_dim=16,
        hidden_dim=64,
        dropout=0.0,
        net_type="lstm",
        num_layers=1,
        use_all_h=False,
        bidirectional=False,
    ):
        super(DontKnowWhat2EatNN, self).__init__()
        self.net_type = net_type
        self.use_all_h = use_all_h
        self.D = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(
            NUM_ROADS, embed_dim, padding_idx=-1
        )  # no padding here

        if net_type.lower() == "lstm":
            self.rnn = torch.nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.reset_lstm_params()
        elif net_type.lower() == "gru":
            self.rnn = torch.nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif net_type.lower() == "vanilla":
            self.rnn = torch.nn.RNN(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                nonlinearity="relu",
                bidirectional=bidirectional,
            )
        elif net_type.lower() == "attn":
            self.rnn = torch.nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=2, batch_first=True, dropout=dropout
            )
        else:
            print("Invalid type.")
            sys.exit(1)

        self.fc = torch.nn.Linear(
            in_features=hidden_dim * self.D, out_features=NUM_ROADS
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def reset_lstm_params(self):
        for name, p in self.rnn.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(p.data)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(p.data)
            elif "bias_ih" in name:
                p.data.fill_(0)
                # Set forget-gate bias to 1
                n = p.size(0)
                p.data[(n // 4) : (n // 2)].fill_(1)
            elif "bias_hh" in name:
                p.data.fill_(0)

    def forward(self, x):
        # x: (batch_size, seq_len)
        out = self.embedding(x)  # (batch_size, seq_len, embed_dim)

        if self.net_type == "attn":
            out, _ = self.rnn(out, out, out)  # (batch_size, seq_len, hidden_dim)
        else:
            out, _ = self.rnn(out)  # (batch_size, seq_len, hidden_dim)

        if self.use_all_h:
            out = out.contiguous().view(
                -1, out.shape[2]
            )  # (batch_size*seq_len, hidden_dim) use all step's output
            out = self.fc(out)  # (batch_size*seq_len, num_roads)
            out = out.view(-1, SEQ_LEN, NUM_ROADS)  # (batch_size, seq_len, num_roads)
            out = out[:, -1, :]  # (batch_size, num_roads)
        else:
            out = out[:, -1, :]  # (batch_size, hidden_dim) use last step's output
            out = self.fc(out)  # (batch_size, num_roads)

        out = self.softmax(out)  # (batch_size, num_roads) probabilities

        return out

    def get_embed_matrix(self):
        return self.embedding.weight.cpu().detach().numpy()  # N * D


def tune():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, required=True)
    parser.add_argument("-v", type=int, required=True)
    args = parser.parse_args()

    p = 0.05
    traj_list_all = np.load(
        f"../data/sz_taxi_202006/sz_taxi_202006_traj_list_bin_24_sampled_{p}_flatten_id.npy",
        allow_pickle=True,
    )

    # net_type_list = ["gru", "lstm"]
    # embed_dim_list = [16, 32, 64, 128]
    # hidden_dim_list = [32, 64, 128, 256]
    # batch_size_list = [64, 128, 256, 512]
    # lr_list = [1e-4, 1e-3]

    if args.t.lower() == "gru":
        net_type_list = ["gru"]
        batch_size_list = [64, 128, 256, 512]
        lr_list = [1e-4, 1e-3]
        if args.v == 1:
            embed_dim_list = [32, 64]
            hidden_dim_list = [128, 256]
        elif args.v == 2:
            embed_dim_list = [128]
            hidden_dim_list = [128, 256]
    elif args.t.lower() == "lstm":
        net_type_list = ["lstm"]
        batch_size_list = [64, 128, 256, 512]
        lr_list = [1e-4, 1e-3]
        if args.v == 1:
            embed_dim_list = [16]
            hidden_dim_list = [32, 64, 128, 256]
        if args.v == 2:
            embed_dim_list = [32]
            hidden_dim_list = [64, 128, 256]
        if args.v == 3:
            embed_dim_list = [64, 128]
            hidden_dim_list = [128, 256]

    for net_type in net_type_list:
        if net_type == "lstm":
            bi = True
        else:
            bi = False
        for embed_dim in embed_dim_list:
            for hidden_dim in hidden_dim_list:
                if hidden_dim <= embed_dim:
                    continue
                for batch_size in batch_size_list:
                    for lr in lr_list:
                        log_file = f"./log/{net_type}_ed{embed_dim}_hd{hidden_dim}_bs{batch_size}_lr{lr}.log"

                        train_loader, val_loader, test_loader = get_dataloaders(
                            traj_list_all, SEQ_LEN, batch_size=batch_size
                        )
                        model = DontKnowWhat2EatNN(
                            embed_dim=embed_dim,
                            hidden_dim=hidden_dim,
                            net_type=net_type,
                            num_layers=1,
                            use_all_h=False,
                            bidirectional=bi,
                        ).to(DEVICE)
                        criterion = torch.nn.CrossEntropyLoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                        train(
                            model,
                            train_loader,
                            val_loader,
                            optimizer,
                            criterion,
                            max_epochs=1000,
                            early_stop=10,
                            verbose=1,
                            plot=False,
                            log=log_file,
                        )


if __name__ == "__main__":
    net_type = "lstm"
    embed_dim = 64
    hidden_dim = 256
    batch_size = 256
    lr = 1e-4
    bi = True

    p = 0.8
    traj_list_all = np.load(
        f"../data/sz_taxi_202006/sz_taxi_202006_traj_list_bin_24_sampled_{p}_flatten_id.npy",
        allow_pickle=True,
    )

    log_file = "train.log"

    train_loader, val_loader, test_loader = get_dataloaders(
        traj_list_all, SEQ_LEN, batch_size=batch_size
    )
    model = DontKnowWhat2EatNN(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        net_type=net_type,
        num_layers=1,
        use_all_h=False,
        bidirectional=bi,
    ).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        max_epochs=1000,
        early_stop=10,
        verbose=1,
        plot=False,
        log=log_file,
    )

    test_loss, test_acc = eval_model(model, test_loader, criterion)
    with open(log_file, "a") as f:
        print("Test Loss = %.5f" % test_loss, "Test acc = %.5f " % test_acc, file=f)
    
    embedding_matrix=model.get_embed_matrix()
    np.save("./saved/embedding_matrix.npy", embedding_matrix, allow_pickle=True)

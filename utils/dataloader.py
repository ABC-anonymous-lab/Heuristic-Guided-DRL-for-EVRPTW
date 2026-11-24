import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


class EVRPDataset(Dataset):
    def __init__(
        self,
        csv_path,
        graph_fraction=1.0,
        max_graphs=None,
    ):
        if csv_path.endswith("csv"):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = pd.read_parquet(csv_path)

        graph_ids = self.df["graph_id"].drop_duplicates().to_list()

        if graph_fraction is not None and graph_fraction < 1.0:
            n_keep = max(1, int(len(graph_ids) * graph_fraction))
            graph_ids = graph_ids[:n_keep]

        if max_graphs is not None:
            graph_ids = graph_ids[:max_graphs]

        self.df = self.df[self.df["graph_id"].isin(graph_ids)]
        self.samples = []

        for gid, gdf in self.df.groupby("graph_id", sort=False):
            gdf = gdf.reset_index(drop=True)
            entry = {
                "graph_id": int(gid),
                "df": gdf,
                "num_nodes": len(gdf),
            }
            self.samples.append(entry)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return {
        "graph_ids": [item["graph_id"] for item in batch],
        "graphs": [item["df"] for item in batch],
        "num_nodes": [item["num_nodes"] for item in batch],
        "batch_size": len(batch),
    }


def build_dataloaders(config):
    g = torch.Generator()
    g.manual_seed(int(config["seed"]))

    dataset_cfg = config["dataset"]
    graph_fraction = dataset_cfg.get("graph_fraction", 1.0)
    max_graphs = dataset_cfg.get("max_graphs", None)

    def make_dataset(csv_key):
        return EVRPDataset(
            dataset_cfg[csv_key],
            graph_fraction=graph_fraction,
            max_graphs=max_graphs,
        )

    train_dataset = make_dataset("train_csv")
    val_dataset   = make_dataset("val_csv")
    test_dataset  = make_dataset("test_csv")

    batch_size = config["train"]["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, generator=g)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)

    return {"train": train_loader, "val": val_loader, "test": test_loader}
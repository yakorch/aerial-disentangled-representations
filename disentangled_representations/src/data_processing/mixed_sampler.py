import torch
from torch.utils.data import Dataset, Sampler



class MixedResamplingSampler(Sampler):
    def __init__(self, dataset_weights_mapping: dict[Dataset, float], shuffle: bool):
        super().__init__()

        self.datasets = list(dataset_weights_mapping.keys())
        self.weights  = list(dataset_weights_mapping.values())

        self.shuffle  = shuffle

        lengths = [0] + [len(ds) for ds in self.datasets]
        self.cum_sizes = torch.cumsum(torch.tensor(lengths), dim=0)

    def __iter__(self):
        all_indices = []
        for i, (ds, w) in enumerate(zip(self.datasets, self.weights)):
            N = len(ds)
            offset = int(self.cum_sizes[i].item())
            n_samples = int(w * N)

            if w > 1.0:
                # NOTE: oversample with replacement
                picks = torch.multinomial(
                    torch.ones(N), num_samples=n_samples, replacement=True
                ).tolist()
            else:
                # NOTE: undersample without replacement
                perm = torch.randperm(N)
                picks = perm[:n_samples].tolist()

            all_indices.extend(offset + p for p in picks)

        indices = torch.tensor(all_indices)
        if self.shuffle:
            perm = torch.randperm(indices.numel())
            indices = indices[perm]

        return iter(indices.tolist())

    def __len__(self):
        return sum(int(w * len(ds)) for ds, w in zip(self.datasets, self.weights))

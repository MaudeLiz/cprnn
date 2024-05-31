import torch


class PTBDataloader:
    def __init__(self, dataset_path: str, batch_size: int = 32, seq_len: int = 32, batch_first: bool = True):
        self.batch_first = batch_first
        self.arr = torch.load(dataset_path)
        self.n_seqs = batch_size
        self.n_steps = seq_len
        self.batch_size = self.n_seqs * self.n_steps
        self.n_batches = len(self.arr) // self.batch_size
        self.arr = self.arr[:self.n_batches * self.batch_size]
        self.arr = self.arr.reshape((self.n_seqs, -1))

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.arr.shape[1]:
            x = self.arr[:, self.n:self.n + self.n_steps]
            y = torch.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], self.arr[:, self.n + self.n_steps]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], self.arr[:, 0]
            self.n += self.n_steps

            if self.batch_first:
                return x, y
            else:
                return x.transpose(0, 1), y.transpose(0, 1)
        else:
            raise StopIteration

class PTBDataloaderOld2:
    def __init__(self, dataset_path: str, batch_size: int = 32, seq_len: int = 32, batch_first: bool = True):
        # Since dataset is small, we load it all into a single vector
        self.batch_first = batch_first
        self.dataset_ids = torch.load(dataset_path)  # entire dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = self.dataset_ids.size(0) // (self.batch_size*self.seq_len)
        self.dataset_ids = self.dataset_ids[:self.num_batches*self.batch_size*self.seq_len]

        # Split evenly into `batch_size` chunks [total//batch_size, batch_size]
        if self.batch_first:
            self.dataset_ids = self.dataset_ids.view(self.num_batches, self.batch_size,
                                                     self.seq_len).contiguous()
        else:
            self.dataset_ids = self.dataset_ids.view(self.num_batches, self.batch_size,
                                                     self.seq_len).permute(0, 2, 1).contiguous()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.num_batches:
            if self.batch_first:
                source = self.dataset_ids[self.n, :, :-1]  # [BS, L]
                target = self.dataset_ids[self.n, :, 1:]  # [BS, L]
            else:
                source = self.dataset_ids[self.n, :-1, :]  # [L, BS]
                target = self.dataset_ids[self.n, 1:, :]  # [L, BS]
            self.n += self.seq_len + 1
            return source, target
        else:
            raise StopIteration

class PTBDataloaderOld:
    def __init__(self, dataset_path: str, batch_size: int = 32, seq_len: int = 32):
        # Since dataset is small, we load it all into a single vector
        self.dataset_ids = torch.load(dataset_path)  # entire dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = self.dataset_ids.size(0) // self.batch_size

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        self.dataset_ids = torch.narrow(self.dataset_ids, 0, 0, self.num_batches * self.batch_size)

        # Split evenly into `batch_size` chunks [total//batch_size, batch_size]
        self.dataset_ids = self.dataset_ids.view(self.batch_size, self.num_batches).t().contiguous()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.num_batches:

            remaining = len(self.dataset_ids[self.n:])
            source = self.dataset_ids[self.n: self.n + min(self.seq_len, remaining - 1)]  # [L, BS]
            target = self.dataset_ids[self.n + 1: self.n + min(1 + self.seq_len, remaining)]  # [L, BS]
            self.n += self.seq_len + 1
            return source, target
        else:
            raise StopIteration


def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''

    batch_size = n_seqs * n_steps
    n_batches = len(arr) // batch_size

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]

    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):

        # The features
        x = arr[:, n:n + n_steps]

        # The targets, shifted by one
        y = np.zeros_like(x)

        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
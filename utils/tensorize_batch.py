import torch

def tensorize_batch(batch, device):
    batch_size= len(batch)
    sample = batch[0]

    res = torch.zeros(batch_size, *sample.shape)

    for i in range(batch_size):
        res[i] = batch[i].to(device)

    return res.to(device)
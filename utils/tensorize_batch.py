import torch

def tensorize_batch(batch):
    batch_size= len(batch)
    sample = batch[0]

    res = torch.zeros(batch_size, *sample.shape)

    for i in range(batch_size):
        res[i] = batch[i].clone().detach()

    return res
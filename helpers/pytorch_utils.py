import torch


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_numpy(tensor: torch.Tensor):
    out = tensor.detach().cpu().numpy()

    return out

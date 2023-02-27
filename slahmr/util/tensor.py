import torch
import numpy as np


def move_to(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [move_to(x, device) for x in obj]
    return obj  # otherwise do nothing


def detach_all(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    if isinstance(obj, dict):
        return {k: detach_all(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [detach_all(x) for x in obj]
    return obj  # otherwise do nothing


def to_torch(obj):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float()
    if isinstance(obj, dict):
        return {k: to_torch(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_torch(x) for x in obj]
    return obj


def to_np(obj):
    if isinstance(obj, torch.Tensor):
        return obj.numpy()
    if isinstance(obj, dict):
        return {k: to_np(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_np(x) for x in obj]
    return obj


def get_device(i=0):
    device = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def scatter_intervals(tensor, start, end, T=None):
    """
    Scatter the tensor contents into intervals from start to end
    output tensor indexed from 0 to end.max()
    :param tensor (B, S, *)
    :param start (B) start indices
    :param end (B) end indices
    :param T (int, optional) max length
    returns (B, T, *) scattered tensor
    """
    assert isinstance(tensor, torch.Tensor) and tensor.ndim >= 2
    if T is not None:
        end = torch.where(end < 0, T + 1 + end, end)
        assert torch.all(end <= T)
    else:
        T = end.max()

    B, S, *dims = tensor.shape
    start, end = start.long(), end.long()
    # get idcs that go past the last time step so we don't have repeat indices in scatter
    idcs = time_segment_idcs(start, end, clip=False)  # (B, S)
    # mask out the extra padding
    mask = idcs >= end[:, None]
    tensor[mask] = 0

    idcs = idcs.reshape(B, S, *(1,) * len(dims)).repeat(1, 1, *dims)
    output = torch.zeros(
        B, idcs.max() + 1, *dims, device=tensor.device, dtype=tensor.dtype
    )
    output.scatter_(1, idcs, tensor)
    # slice out the extra segments
    return output[:, :T]


def get_scatter_mask(start, end, T):
    """
    get the mask of selected intervals
    """
    B = start.shape[0]
    start, end = start.long(), end.long()
    assert torch.all(end <= T)
    idcs = time_segment_idcs(start, end, clip=True)
    mask = torch.zeros(B, T, device=start.device, dtype=torch.bool)
    mask.scatter_(1, idcs, 1)
    return mask


def select_intervals(series, start, end):
    """
    Select slices of a tensor from start to end
    will pad uneven sequences to all the max segment length
    :param series (B, T, *)
    :param start (B)
    :param end (B)
    returns (B, S, *) selected segments, S = max(end - start)
    """
    B, T, *dims = series.shape
    assert torch.all(end <= T)
    sel = time_segment_idcs(start, end, clip=True)
    S = sel.shape[1]
    sel = sel.reshape(B, S, *(1,) * len(dims)).repeat(1, 1, *dims)
    return torch.gather(series, 1, sel)


def get_select_mask(start, end):
    """
    get the mask of unpadded elementes for the selected time segments
    e.g. sel[mask] are the unpadded elements
    :param start (B)
    :param end (B)
    """
    idcs = time_segment_idcs(start, end, clip=False)
    return idcs < end[:, None]  # (B, S)


def time_segment_idcs(start, end, clip=True):
    """
    :param start (B)
    :param end (B)
    returns (B, S) long tensor of indices, where S = max(end - start)
    """
    start, end = start.long(), end.long()
    S = (end - start).max()
    seg = torch.arange(S, dtype=torch.int64, device=start.device)
    idcs = start[:, None] + seg[None, :]  # (B, S)
    if clip:
        # clip at the lengths of each track
        imax = end[:, None] - 1
        idcs = idcs.clamp(max=imax)
    return idcs

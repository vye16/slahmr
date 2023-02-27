import torch
import numpy as np


def get_seq_figure_skip(seq_name=None):
    if seq_name == "017437_mpii_test":
        return 15
    if seq_name == "012968_mpii_test":
        return 15
    if seq_name == "recording_20211002_S03_S18_04-all-100-200":
        return 11
    return 10


def get_seq_static_lookat_points(seq_name=None, bounds=None):
    if seq_name == "002276_mpii_test":
        top_source = torch.tensor([-0.5, -3.0, -2.0])
        top_target = torch.tensor([0.0, 0.0, 4.0])

        side_source = torch.tensor([6.0, -2.0, 2.0])
        side_target = torch.tensor([2.0, 0.0, 6.0])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "002374_mpii_test":
        top_source = torch.tensor([-4.0, 0.0, 0.0])
        top_target = torch.tensor([-2.0, 2.0, 3.0])

        side_source = torch.tensor([-7.0, 1.0, 7.0])
        side_target = torch.tensor([0.0, 1.0, 8.0])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "003742_mpii_test":
        top_source = torch.tensor([3.0, -2.0, -6])
        top_target = torch.tensor([3.0, 1.0, 2.0])

        side_source = torch.tensor([-5.0, -1.0, -2])
        side_target = torch.tensor([2.0, 0.0, 5])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "003747_mpii_test":
        top_source = torch.tensor([-5.0, -2.0, -5])
        top_target = torch.tensor([-3.0, 2.0, 1.0])

        side_source = torch.tensor([-9.0, -1.0, -4])
        side_target = torch.tensor([-3.0, 2.0, 3])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "003943_mpii_test":
        top_source = torch.tensor([0, -1, -2])
        top_target = torch.tensor([0, 1, 3])

        side_source = torch.tensor([6, 0, 1])
        side_target = torch.tensor([0, 0, 5])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "006537_mpii_test":
        top_source = torch.tensor([0, -4, -1])
        top_target = torch.tensor([1.0, 0.0, 5.0])

        side_source = torch.tensor([5, -2, 0])
        side_target = torch.tensor([1.0, 1.0, 7.0])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "007684_mpii_test":
        top_source = torch.tensor([-2, -4, -6])
        top_target = torch.tensor([0.5, 0.0, 4.0])

        side_source = torch.tensor([-8, -3, -2])
        side_target = torch.tensor([1.0, 0.0, 5.0])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "009039_mpii_test":
        top_source = torch.tensor([-2.0, -3.0, -1.5])
        top_target = torch.tensor([1.0, 0.0, 3.0])

        side_source = torch.tensor([6, -2, -1.5])
        side_target = torch.tensor([2.0, 0.0, 6.0])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "009607_mpii_test":
        top_source = torch.tensor([0, -3, -1])
        top_target = torch.tensor([0, 0, 5])

        side_source = torch.tensor([-6, -0.5, 2])
        side_target = torch.tensor([-1.0, 0.5, 7])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "012968_mpii_test":
        top_source = torch.tensor([-0.5, -3, -4])
        top_target = torch.tensor([0.0, 0.5, 2.0])

        side_source = torch.tensor([7, -4, 0])
        side_target = torch.tensor([0, 0.5, 2.0])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "014531_mpii_test":
        top_source = torch.tensor([0, -2, -2])
        top_target = torch.tensor([0, 0, 2])

        side_source = torch.tensor([-2, 0, -2])
        side_target = torch.tensor([0, 0, 2])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "015933_mpii_test":
        top_source = torch.tensor([0.0, 0.5, -5.0])
        top_target = torch.tensor([0.0, 2.0, -2.0])

        side_source = torch.tensor([5.0, 1.5, -3.0])
        side_target = torch.tensor([0.0, 2.5, 1.5])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "016195_mpii_test":
        top_source = torch.tensor([0, -1, -2])
        top_target = torch.tensor([0, 1, 1])

        side_source = torch.tensor([-3, 0, 1])
        side_target = torch.tensor([0, 1, 2])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "018061_mpii_test":
        top_source = torch.tensor([4, -1, 2])
        top_target = torch.tensor([1, 1, 3])

        side_source = torch.tensor([-3, 0, 4])
        side_target = torch.tensor([1, 1, 3])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "018713_mpii_test":
        top_source = torch.tensor([0, -1, -1])
        top_target = torch.tensor([0, 0.5, 2])

        side_source = torch.tensor([3, 0, 1])
        side_target = torch.tensor([0, 0, 3])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "022691_mpii_test":
        top_source = torch.tensor([0, -4, -4])
        top_target = torch.tensor([-2.0, 0.0, 1.0])

        side_source = torch.tensor([-5, -2, -5])
        side_target = torch.tensor([-3.5, 0.0, 1.0])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "023390_mpii_test":
        top_source = torch.tensor([0.0, -1.5, -1.0])
        top_target = torch.tensor([0.0, 0.0, 2.0])

        side_source = torch.tensor([4.0, -0.5, 0])
        side_target = torch.tensor([1.0, 0.5, 4])
        #         top_source = torch.tensor([0.0, -3.0, -2.0])
        #         top_target = torch.tensor([0.0, 0.5, 2.0])

        #         side_source = torch.tensor([5.0, -0.5, -3])
        #         side_target = torch.tensor([1.0, 0.5, 4])
        return (top_source, top_target), (side_source, side_target)

    if seq_name=="024165_mpii_test":
        top_source = torch.tensor([0, -3, -3])
        top_target = torch.tensor([0, 0, 3])

        side_source = torch.tensor([-4, -1, -2])
        side_target = torch.tensor([0, 0, 3])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "024154_mpii_test":
        top_source = torch.tensor([-1, -0.5, 1])
        top_target = torch.tensor([-1, 1, 8])

        side_source = torch.tensor([-4, 1, 3])
        side_target = torch.tensor([-1, 1, 8])
        return (top_source, top_target), (side_source, side_target)

#    if seq_name == "024159_mpii_test":
#        top_source = torch.tensor([-4.0, -2.0, -6.0])
#        top_target = torch.tensor([-4.0, 0.0, 0.0])

#        side_source = torch.tensor([-12, -1, -2])
#        side_target = torch.tensor([-10.0, 0.0, 0.0])
#        return (top_source, top_target), (side_source, side_target)

    if seq_name == "023962_mpii_test":
        top_source = torch.tensor([0, -0.5, -2])
        top_target = torch.tensor([0, 0.5, 2])

        side_source = torch.tensor([3, 0.5, 0])
        side_target = torch.tensor([0, 0, 2])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "017437_mpii_test":
        top_source = torch.tensor([0, -8, -6])
        top_target = torch.tensor([-2.0, -1.0, 2.0])

        side_source = torch.tensor([-5, -1, -7])
        side_target = torch.tensor([-3.0, -1.0, 1.0])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "recording_20211002_S03_S18_04-all-100-200":
        top_source = torch.tensor([1.0, -1, -4.5])
        top_target = torch.tensor([1.0, 0.0, 1])

        side_source = torch.tensor([1.0, 0.5, -4.5])
        side_target = torch.tensor([1.0, 0.5, 1])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "recording_20211002_S03_S18_04-all-100-500":
        top_source = torch.tensor([0.0, -0.5, -3])
        top_target = torch.tensor([0.0, 1.0, 1])

        side_source = torch.tensor([-2.0, 0.5, -3])
        side_target = torch.tensor([0.0, 1.0, 2])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "recording_20211002_S03_S18_01-all-700-862":
        top_source = torch.tensor([0.5, -2.0, -2])
        top_target = torch.tensor([0.5, -0.5, 2])

        side_source = torch.tensor([2.0, -0.5, -2])
        side_target = torch.tensor([0.5, -0.5, 1])
        return (top_source, top_target), (side_source, side_target)

    if seq_name == "recording_20220315_S21_S30_02-all-0-100":
        top_source = torch.tensor([0.0, -1, -2])
        top_target = torch.tensor([0.0, 0.5, 1])

        side_source = torch.tensor([-2.0, 0.0, -1])
        side_target = torch.tensor([0.0, 0.5, 1])
        return (top_source, top_target), (side_source, side_target)

    if bounds is not None:
        # (3), (3), (3)
        bb_min, bb_max, center = bounds
        print("SCENE BOUNDS", bb_min, bb_max, center)
        length = torch.abs(bb_max - bb_min).max()
        print(length)
        top_source = center + torch.tensor([0.0, -2.0, -0.9 * length])

        side_source = center + torch.tensor([0.5 * length, -0.5, -0.7 * length])
        return (top_source, center), (side_source, center)

    top_source = torch.tensor([0.0, -2.0, -3.0])
    top_target = torch.tensor([0.0, 0.0, 1.0])

    side_source = torch.tensor([3.0, -1.0, -1.0])
    side_target = torch.tensor([0.0, 0.0, 1.0])
    return (top_source, top_target), (side_source, side_target)

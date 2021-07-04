import numpy as np
import torch

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = "<blank>"
UNK_WORD = "<unk>"
BOS_WORD = "<s>"
EOS_WORD = "</s>"

def get_sinusoid_encoding_table(
    n_position: int,
    d_hid: int, 
    padding_idx=None
) -> torch.Tensor:
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

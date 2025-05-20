import numpy as np
import torch


def array_effort(filters_frq: torch.Tensor, bz_rirs: torch.Tensor):
    reference_rirs = bz_rirs[:, :, 0:1, :]  # [B, M, L, K]

    print(reference_rirs.size())

    m_b = reference_rirs.size()[1]
    reference_rirs_h = reference_rirs.transpose(1, 2).conj()  # (B, N, M, T)

    print(reference_rirs.size())

    # Now permute so you can batch matmul over (M, N)
    GH = reference_rirs_h.permute(0, 3, 1, 2)  # (B, T, N, M)
    G = reference_rirs.permute(0, 3, 1, 2)  # (B, T, M, N)

    abs_qr_sqr = m_b / (torch.matmul(GH, G))

    print(abs_qr_sqr.size())

    filters = filters_frq[:, :, None, :]

    filters_h = filters.transpose(1, 2).conj()

    print(filters.size())
    print(filters_h.size())

    qH = filters_h.permute(0, 3, 1, 2)
    q = filters.permute(0, 3, 1, 2)

    array_effort = torch.matmul(qH, q) / abs_qr_sqr

    print(array_effort[0, :, 0, 0])

    return array_effort

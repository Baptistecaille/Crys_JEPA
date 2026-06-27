"""Geometric augmentation helpers used by JEPA.

The functions here apply random translations and rotations so the JEPA model
sees multiple crystal views during contrastive training.
"""

import torch

def translate(frac, b, batch):
    """Translate each crystal in the batch by a random fractional offset."""
    device = frac.device
    vec_t = torch.rand(b, 3).to(device)
    frac_trans = (frac + vec_t[batch]) % 1.
    return frac_trans, vec_t

def _random_rotation_matrix_so3(batch=1):
    """Sample a random rotation matrix from SO(3) using quaternions."""
    u1 = torch.rand(batch)
    u2 = torch.rand(batch)
    u3 = torch.rand(batch)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1)     * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1)     * torch.cos(2 * torch.pi * u3)  # w

    # Quaternion to rotation matrix
    x, y, z, w = q1, q2, q3, q4
    R = torch.empty((batch, 3, 3))

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - z*w)
    R[:, 0, 2] = 2 * (x*z + y*w)

    R[:, 1, 0] = 2 * (x*y + z*w)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - x*w)

    R[:, 2, 0] = 2 * (x*z - y*w)
    R[:, 2, 1] = 2 * (y*z + x*w)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    return R[0] if batch == 1 else R, torch.stack([u1, u2, u3], -1)

def rotate(lattice, b):
    """Rotate each lattice in the batch with an independent random SO(3) matrix."""
    R, vec_r = _random_rotation_matrix_so3(b)
    Rt = R.transpose(-1, -2).to(lattice.device)
    lat_rot = lattice @ Rt
    return lat_rot, vec_r.to(lattice.device)

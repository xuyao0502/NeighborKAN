# ---------------------------
# functions
# ---------------------------

import numpy as np
import torch
import torch.nn.functional as F


def compute_image_gradients(img):
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    return grad_x, grad_y


def gradient_consistency_loss(I_ref_warp, I_def):
    gx_ref, gy_ref = compute_image_gradients(I_ref_warp)
    gx_def, gy_def = compute_image_gradients(I_def)

    grad_loss = torch.mean(torch.abs(gx_ref - gx_def) + torch.abs(gy_ref - gy_def))

    return grad_loss


def charbonnier_loss(Iw, I_t, eps=1e-3):
    diff = Iw - I_t
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def laplacian_smoothness_loss(uv_map):
    u = uv_map[:, 0:1, :, :]
    v = uv_map[:, 1:2, :, :]

    def laplace_loss(field):
        up    = field[:, :, :-2, 1:-1]
        down  = field[:, :, 2:, 1:-1]
        left  = field[:, :, 1:-1, :-2]
        right = field[:, :, 1:-1, 2:]
        center = field[:, :, 1:-1, 1:-1]
        neighbor_mean = (up + down + left + right) / 4.0
        return F.mse_loss(center, neighbor_mean, reduction='mean')

    return 0.5 * (laplace_loss(u) + laplace_loss(v))


def build_inputs_from_neighbor_buffer(neighbor_uv, H, W, device):

    if isinstance(neighbor_uv, np.ndarray):
        nb = neighbor_uv
    else:
        nb = neighbor_uv.cpu().numpy()

    # prepare coords normalized
    xs = np.linspace(-1.0, 1.0, W)
    ys = np.linspace(-1.0, 1.0, H)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')  # H x W
    x_flat = grid_x.reshape(-1).astype(np.float32)
    y_flat = grid_y.reshape(-1).astype(np.float32)

    # for neighbors: up,right,down,left
    u_up = np.zeros((H,W), dtype=np.float32)
    v_up = np.zeros((H,W), dtype=np.float32)
    u_right = np.zeros((H,W), dtype=np.float32)
    v_right = np.zeros((H,W), dtype=np.float32)
    u_down = np.zeros((H,W), dtype=np.float32)
    v_down = np.zeros((H,W), dtype=np.float32)
    u_left = np.zeros((H,W), dtype=np.float32)
    v_left = np.zeros((H,W), dtype=np.float32)

    # ---------------------------
    u_up[1:,:] = nb[:-1,:,0]
    v_up[1:,:] = nb[:-1,:,1]
    u_down[:-1,:] = nb[1:,:,0]
    v_down[:-1,:] = nb[1:,:,1]
    u_left[:,1:] = nb[:, :-1, 0]
    v_left[:,1:] = nb[:, :-1, 1]
    u_right[:,:-1] = nb[:, 1:, 0]
    v_right[:,:-1] = nb[:, 1:, 1]

    # ---------------------------
    u_up[0, :] = u_up[1, :]
    v_up[0, :] = v_up[1, :]
    u_down[-1, :] = u_down[-2, :]
    v_down[-1, :] = v_down[-2, :]
    u_left[:, 0] = u_left[:, 1]
    v_left[:, 0] = v_left[:, 1]
    u_right[:, -1] = u_right[:, -2]
    v_right[:, -1] = v_right[:, -2]

    feats = np.stack([
        x_flat, y_flat,
        u_up.reshape(-1), v_up.reshape(-1),
        u_right.reshape(-1), v_right.reshape(-1),
        u_down.reshape(-1), v_down.reshape(-1),
        u_left.reshape(-1), v_left.reshape(-1)
    ], axis=1)  # [N,10]

    return torch.from_numpy(feats).to(device)

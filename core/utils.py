import numpy as np
import torch
import pprint
import os
import imageio
import cv2
import io
import re

# Function to print attributes in a table-like format
def print_attributes(obj, level=0):
    # Get the dictionary of attributes and their values
    attributes = vars(obj)
    
    # Find the longest key name to align the table
    max_key_length = max(len(key) for key in attributes.keys()) if attributes else 0
    
    # Print the table header for the current level
    if level == 0:
        print(f"{'Attribute'.ljust(max_key_length)} | {'Value'}")
        print('-' * (max_key_length + 10))  # Print separator

    # Print each attribute and its value
    for key, value in attributes.items():
        if key.startswith('_'):
            continue
        indent = '    ' * level  # Indentation for nested attributes
        # If the value is an instance of a class, recursively print its attributes
        if hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool, np.ndarray)):
            print(f"{indent}{key.ljust(max_key_length)} |")
            print_attributes(value, level + 1)
            continue
        if isinstance(value, (np.ndarray, torch.Tensor)):
            value = value.shape
        print(f"{indent}{key.ljust(max_key_length)} | {pprint.pformat(value)}")


def load_with_cache(client, file_path, cache_dir, parse_text_to_float=True):
    if "s3://" in file_path:
        local_path = os.path.join(cache_dir, file_path.split('s3://')[-1])
        # Download from S3 if not in local cache
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f: 
                f.write(client.get(file_path))
    else:
        local_path = file_path

    # handle different file types
    if local_path.endswith('.txt'):
        data = []
        with open(local_path, 'r') as f:
            for line in f.readlines():
                if parse_text_to_float:
                    data.append([float(x) for x  in line.strip().split(' ')])
                else:
                    data.append(line.strip())
        return data
    elif local_path.endswith('.png'):
        img = cv2.imread(local_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        return img
    elif local_path.endswith('.npy'):
        return np.load(local_path)
    elif local_path.endswith('.mp4'):
        return imageio.get_reader(local_path, "ffmpeg")

def load_from_ceph(client, file_path, cache_dir, parse_text_to_float=True):
    is_ceph_bytes = False
    local_path = file_path
    # handle different file types
    if local_path.endswith('.txt'):
        if is_ceph_bytes:
            data = []
            for line in file_bytes.readlines():
                line = line.decode('utf-8')
                if parse_text_to_float:
                    data.append([float(x) for x  in line.strip().split(' ')])
                else:
                    data.append(line.strip())
            return data
        else:
            data = []
            with open(local_path, 'r') as f:
                for line in f.readlines():
                    if parse_text_to_float:
                        data.append([float(x) for x  in line.strip().split(' ')])
                    else:
                        data.append(line.strip())
            return data
    elif local_path.endswith('.png'):
        if is_ceph_bytes:
            image_bytes = np.frombuffer(file_bytes.read(), np.uint8)
            img = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        else:
            img = cv2.imread(local_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        return img
    elif local_path.endswith('.npy'):
        if is_ceph_bytes:
            return np.load(file_bytes)
        else:
            return np.load(local_path)
    elif local_path.endswith('.mp4'):
        if is_ceph_bytes:
            return imageio.v3.imread(file_bytes, format_hint='.mp4')
        else:
            return imageio.get_reader(local_path, "ffmpeg")
    elif os.path.isdir(local_path):
        return load_png_sequence_from_folder(local_path, is_ceph_bytes)

def load_png_sequence_from_folder(folder_path, is_ceph_bytes=False):
    images = []
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]
    png_files_sorted = sorted(png_files, key=lambda x: [
        int(part) if part.isdigit() else part 
        for part in re.findall(r'\d+|\D+', x)
    ])
    
    for png_file in png_files_sorted:
        img_path = os.path.join(folder_path, png_file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if img is not None:
            images.append(img)
    return images

def signed_expm1(x):
    sign = torch.sign(x)
    return sign * torch.expm1(torch.abs(x))

def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res
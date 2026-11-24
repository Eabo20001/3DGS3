import torch


COLOR_SPACE_NAME = "bt601_full_range_ycbcr"


def _check_three_channels(image: torch.Tensor, channel_dim: int) -> int:
    channel_dim = channel_dim % image.ndim
    if image.shape[channel_dim] != 3:
        raise ValueError(
            f"Expected 3 channels at dimension {channel_dim}, got shape {tuple(image.shape)}"
        )
    return channel_dim


def rgb_to_ycbcr(image: torch.Tensor, channel_dim: int = -3) -> torch.Tensor:
    """Convert RGB in [0, 1] to full-range BT.601 YCbCr in [0, 1]."""
    channel_dim = _check_three_channels(image, channel_dim)
    red, green, blue = image.unbind(dim=channel_dim)

    y = 0.299000 * red + 0.587000 * green + 0.114000 * blue
    cb = 0.500000 - 0.168736 * red - 0.331264 * green + 0.500000 * blue
    cr = 0.500000 + 0.500000 * red - 0.418688 * green - 0.081312 * blue
    return torch.stack((y, cb, cr), dim=channel_dim)


def ycbcr_to_rgb(image: torch.Tensor, channel_dim: int = -3) -> torch.Tensor:
    """Convert full-range BT.601 YCbCr to RGB without implicit clamping."""
    channel_dim = _check_three_channels(image, channel_dim)
    y, cb, cr = image.unbind(dim=channel_dim)
    cb = cb - 0.5
    cr = cr - 0.5

    red = y + 1.402000 * cr
    green = y - 0.344136 * cb - 0.714136 * cr
    blue = y + 1.772000 * cb
    return torch.stack((red, green, blue), dim=channel_dim)


def ycbcr_background(
    white_background: bool,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return encoded black/white background for BT.601 full-range YCbCr."""
    y = 1.0 if white_background else 0.0
    return torch.tensor([y, 0.5, 0.5], dtype=dtype, device=device)


def save_ycbcr444p8(image: torch.Tensor, path: str) -> None:
    """Save a CHW YCbCr tensor as planar 8-bit YCbCr 4:4:4 raw data."""
    _check_three_channels(image, 0)
    if image.ndim != 3:
        raise ValueError(f"Expected a CHW tensor, got shape {tuple(image.shape)}")
    planes = image.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().contiguous()
    with open(path, "wb") as output_file:
        output_file.write(planes.numpy().tobytes(order="C"))

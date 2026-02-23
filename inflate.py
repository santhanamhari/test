# onconet/models/inflate.py
"""
Inflate 2D CNN models to 3D with center weight initialization.
Based on: "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
https://arxiv.org/abs/1705.07750
"""

import torch
import torch.nn as nn
import copy


def inflate_model(model):
    """
    Inflate a 2D model to 3D, converting all layers and weights.
    
    Args:
        model: A 2D PyTorch model (e.g., CustomResnet)
    
    Returns:
        Inflated 3D model with properly inflated weights
    """
    # Deep copy to avoid modifying original
    model_3d = copy.deepcopy(model)
    
    # Recursively inflate all modules
    _inflate_module_recursive(model_3d)
    
    return model_3d


def _inflate_module_recursive(module):
    """
    Recursively inflate all 2D layers in a module to 3D.
    Processes children first (post-order traversal).
    """
    for name, child in list(module.named_children()):
        # Recursively process children first
        _inflate_module_recursive(child)
        
        # Replace 2D layers with 3D equivalents
        if isinstance(child, nn.Conv2d):
            inflated = _inflate_conv2d(child)
            setattr(module, name, inflated)
        
        elif isinstance(child, nn.BatchNorm2d):
            inflated = _inflate_batchnorm2d(child)
            setattr(module, name, inflated)
        
        elif isinstance(child, nn.MaxPool2d):
            inflated = _inflate_maxpool2d(child)
            setattr(module, name, inflated)
        
        elif isinstance(child, nn.AvgPool2d):
            inflated = _inflate_avgpool2d(child)
            setattr(module, name, inflated)
        
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            inflated = _inflate_adaptive_avgpool2d(child)
            setattr(module, name, inflated)
        
        elif isinstance(child, nn.AdaptiveMaxPool2d):
            inflated = _inflate_adaptive_maxpool2d(child)
            setattr(module, name, inflated)


def _inflate_conv2d(conv2d, time_kernel_size=3, time_stride=1, time_padding=1):
    """
    Inflate a Conv2d layer to Conv3d with center weight initialization.
    
    For center initialization, all the weight is placed in the center temporal
    frame, with zeros in other frames. This preserves the 2D behavior when
    applied to replicated frames.
    
    Args:
        conv2d: The 2D convolutional layer
        time_kernel_size: Temporal kernel size (default 3)
        time_stride: Temporal stride (default 1)  
        time_padding: Temporal padding (default 1)
    
    Returns:
        Inflated Conv3d layer with center-initialized weights
    """
    # Get 2D kernel size
    if isinstance(conv2d.kernel_size, tuple):
        kernel_h, kernel_w = conv2d.kernel_size
    else:
        kernel_h = kernel_w = conv2d.kernel_size
    
    # Handle 1x1 convs: use 1x1x1 (no temporal extent)
    if kernel_h == 1 and kernel_w == 1:
        time_kernel_size = 1
        time_padding = 0
    
    # Handle 7x7 conv (first layer): use 3x7x7
    if kernel_h == 7 and kernel_w == 7:
        time_kernel_size = 3
        time_padding = 1
    
    # Get 2D stride
    if isinstance(conv2d.stride, tuple):
        stride_h, stride_w = conv2d.stride
    else:
        stride_h = stride_w = conv2d.stride
    
    # Get 2D padding
    if isinstance(conv2d.padding, tuple):
        pad_h, pad_w = conv2d.padding
    else:
        pad_h = pad_w = conv2d.padding
    
    # Get 2D dilation
    if isinstance(conv2d.dilation, tuple):
        dil_h, dil_w = conv2d.dilation
    else:
        dil_h = dil_w = conv2d.dilation
    
    # Create 3D conv
    conv3d = nn.Conv3d(
        in_channels=conv2d.in_channels,
        out_channels=conv2d.out_channels,
        kernel_size=(time_kernel_size, kernel_h, kernel_w),
        stride=(time_stride, stride_h, stride_w),
        padding=(time_padding, pad_h, pad_w),
        dilation=(1, dil_h, dil_w),
        groups=conv2d.groups,
        bias=(conv2d.bias is not None)
    )
    
    # Inflate weights with center initialization
    weight_2d = conv2d.weight.data  # Shape: (out_ch, in_ch/groups, H, W)
    out_channels = weight_2d.shape[0]
    in_channels_per_group = weight_2d.shape[1]
    
    # Initialize 3D weights with zeros
    weight_3d = torch.zeros(
        out_channels, 
        in_channels_per_group,
        time_kernel_size, 
        kernel_h, 
        kernel_w,
        dtype=weight_2d.dtype,
        device=weight_2d.device
    )
    
    # Place all weight in center temporal frame
    center_idx = time_kernel_size // 2
    weight_3d[:, :, center_idx, :, :] = weight_2d
    
    conv3d.weight.data = weight_3d
    
    # Copy bias if present
    if conv2d.bias is not None:
        conv3d.bias.data = conv2d.bias.data.clone()
    
    return conv3d


def _inflate_batchnorm2d(bn2d):
    """
    Convert BatchNorm2d to BatchNorm3d.
    BatchNorm parameters are directly transferable (channel-wise stats).
    """
    bn3d = nn.BatchNorm3d(
        num_features=bn2d.num_features,
        eps=bn2d.eps,
        momentum=bn2d.momentum,
        affine=bn2d.affine,
        track_running_stats=bn2d.track_running_stats
    )
    
    # Copy learned parameters (gamma and beta)
    if bn2d.affine:
        bn3d.weight.data = bn2d.weight.data.clone()
        bn3d.bias.data = bn2d.bias.data.clone()
    
    # Copy running statistics
    if bn2d.track_running_stats and bn2d.running_mean is not None:
        bn3d.running_mean.data = bn2d.running_mean.data.clone()
        bn3d.running_var.data = bn2d.running_var.data.clone()
        if bn2d.num_batches_tracked is not None:
            bn3d.num_batches_tracked.data = bn2d.num_batches_tracked.data.clone()
    
    return bn3d


def _inflate_maxpool2d(pool2d):
    """
    Convert MaxPool2d to MaxPool3d.
    Temporal dimension uses kernel=1, stride=1 (no temporal pooling).
    """
    # Get kernel size
    if isinstance(pool2d.kernel_size, tuple):
        k_h, k_w = pool2d.kernel_size
    else:
        k_h = k_w = pool2d.kernel_size
    
    # Get stride
    if isinstance(pool2d.stride, tuple):
        s_h, s_w = pool2d.stride
    else:
        s_h = s_w = pool2d.stride if pool2d.stride is not None else pool2d.kernel_size
    
    # Get padding
    if isinstance(pool2d.padding, tuple):
        p_h, p_w = pool2d.padding
    else:
        p_h = p_w = pool2d.padding
    
    return nn.MaxPool3d(
        kernel_size=(1, k_h, k_w),
        stride=(1, s_h, s_w),
        padding=(0, p_h, p_w),
        dilation=pool2d.dilation,
        ceil_mode=pool2d.ceil_mode
    )


def _inflate_avgpool2d(pool2d):
    """
    Convert AvgPool2d to AvgPool3d.
    Temporal dimension uses kernel=1, stride=1 (no temporal pooling).
    """
    # Get kernel size
    if isinstance(pool2d.kernel_size, tuple):
        k_h, k_w = pool2d.kernel_size
    else:
        k_h = k_w = pool2d.kernel_size
    
    # Get stride
    if isinstance(pool2d.stride, tuple):
        s_h, s_w = pool2d.stride
    else:
        s_h = s_w = pool2d.stride if pool2d.stride is not None else pool2d.kernel_size
    
    # Get padding
    if isinstance(pool2d.padding, tuple):
        p_h, p_w = pool2d.padding
    else:
        p_h = p_w = pool2d.padding
    
    return nn.AvgPool3d(
        kernel_size=(1, k_h, k_w),
        stride=(1, s_h, s_w),
        padding=(0, p_h, p_w),
        ceil_mode=pool2d.ceil_mode,
        count_include_pad=pool2d.count_include_pad
    )


def _inflate_adaptive_avgpool2d(pool2d):
    """
    Convert AdaptiveAvgPool2d to AdaptiveAvgPool3d.
    Preserves temporal dimension (output_size=1 for T).
    """
    output_size = pool2d.output_size
    
    if output_size is None:
        return nn.AdaptiveAvgPool3d((1, None, None))
    
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    # Pool temporal to 1, spatial as specified
    return nn.AdaptiveAvgPool3d((1, output_size[0], output_size[1]))


def _inflate_adaptive_maxpool2d(pool2d):
    """
    Convert AdaptiveMaxPool2d to AdaptiveMaxPool3d.
    """
    output_size = pool2d.output_size
    
    if output_size is None:
        return nn.AdaptiveMaxPool3d((1, None, None))
    
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    return nn.AdaptiveMaxPool3d((1, output_size[0], output_size[1]))


# ============== Utility Functions ==============

def inflate_state_dict(state_dict_2d, temporal_kernel_size=3):
    """
    Inflate a 2D state dict to 3D (standalone utility).
    Useful for inflating pretrained weights without the model.
    
    Args:
        state_dict_2d: State dict from 2D model
        temporal_kernel_size: Temporal kernel size for conv layers
    
    Returns:
        Inflated state dict for 3D model
    """
    inflated = {}
    
    for key, value in state_dict_2d.items():
        if len(value.shape) == 4:  # Conv weights: (out, in, H, W)
            out_ch, in_ch, h, w = value.shape
            
            # Determine temporal kernel size
            if h == 1 and w == 1:
                t_size = 1  # 1x1 conv -> 1x1x1
            else:
                t_size = temporal_kernel_size
            
            # Center initialization
            weight_3d = torch.zeros(
                out_ch, in_ch, t_size, h, w,
                dtype=value.dtype
            )
            center_idx = t_size // 2
            weight_3d[:, :, center_idx, :, :] = value
            
            inflated[key] = weight_3d
        else:
            # BatchNorm, bias, FC - direct copy
            inflated[key] = value.clone()
    
    return inflated


def check_inflation(model):
    """
    Debug utility to verify inflation was successful.
    Prints all Conv and BatchNorm layers with their dimensions.
    """
    print("\n" + "="*60)
    print("MODEL INFLATION CHECK")
    print("="*60)
    
    conv2d_count = 0
    conv3d_count = 0
    bn2d_count = 0
    bn3d_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv2d_count += 1
            print(f"[2D] Conv2d: {name} - {module.weight.shape}")
        elif isinstance(module, nn.Conv3d):
            conv3d_count += 1
            print(f"[3D] Conv3d: {name} - {module.weight.shape}")
        elif isinstance(module, nn.BatchNorm2d):
            bn2d_count += 1
        elif isinstance(module, nn.BatchNorm3d):
            bn3d_count += 1
    
    print("="*60)
    print(f"Conv2d layers: {conv2d_count}")
    print(f"Conv3d layers: {conv3d_count}")
    print(f"BatchNorm2d layers: {bn2d_count}")
    print(f"BatchNorm3d layers: {bn3d_count}")
    print("="*60 + "\n")
    
    if conv2d_count > 0:
        print("WARNING: Some Conv2d layers were not inflated!")
    if bn2d_count > 0:
        print("WARNING: Some BatchNorm2d layers were not inflated!")
    
    return conv2d_count == 0 and bn2d_count == 0

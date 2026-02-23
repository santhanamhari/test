# onconet/models/pools/global_avg_pool.py

import torch
import torch.nn as nn
from onconet.models.pools.factory import RegisterPool


@RegisterPool("GlobalAvgPool")
class GlobalAvgPool(nn.Module):
    """
    Global Average Pooling that works for both 2D and 3D inputs.
    """
    
    def __init__(self, args, num_chan):
        super(GlobalAvgPool, self).__init__()
        self.args = args
        self.num_chan = num_chan
        
    def forward(self, x, risk_factors=None):
        """
        Args:
            x: Input tensor
               - 2D: (B, C, H, W)
               - 3D: (B, C, D, H, W)
        
        Returns:
            logit: Placeholder (zeros)
            hidden: Pooled features (B, C)
        """
        # Adaptive pooling works for both 2D and 3D
        if len(x.shape) == 5:  # 3D input: (B, C, D, H, W)
            hidden = x.mean(dim=[2, 3, 4])  # Pool over D, H, W
        elif len(x.shape) == 4:  # 2D input: (B, C, H, W)
            hidden = x.mean(dim=[2, 3])  # Pool over H, W
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Return placeholder logit and hidden
        logit = torch.zeros(hidden.shape[0], self.args.num_classes).to(hidden.device)
        
        return logit, hidden
    
    def replaces_fc(self):
        return False


@RegisterPool("GlobalMaxPool")
class GlobalMaxPool(nn.Module):
    """
    Global Max Pooling that works for both 2D and 3D inputs.
    """
    
    def __init__(self, args, num_chan):
        super(GlobalMaxPool, self).__init__()
        self.args = args
        self.num_chan = num_chan
        
    def forward(self, x, risk_factors=None):
        if len(x.shape) == 5:  # 3D input
            hidden = x.amax(dim=[2, 3, 4])
        elif len(x.shape) == 4:  # 2D input
            hidden = x.amax(dim=[2, 3])
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        logit = torch.zeros(hidden.shape[0], self.args.num_classes).to(hidden.device)
        
        return logit, hidden
    
    def replaces_fc(self):
        return False

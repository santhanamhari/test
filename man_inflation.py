from onconet.models.custom_resnet import CustomResnet
from onconet.models.inflate import inflate_model, check_inflation

# Create 2D model with pretrained weights
args.pretrained_on_imagenet = True
model_2d = CustomResnet(args)

# Inflate to 3D
model_3d = inflate_model(model_2d)

# Verify
check_inflation(model_3d)

# Use with 3D input
x = torch.randn(2, 1, 4, 256, 256)  # (B, C, D, H, W)
output = model_3d(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class M2DFramework(nn.Module):
    """
    Core M2D Framework Architecture.
    Includes an EfficientNet-B3 encoder, a UPerHead decoder, and a Domain Adapter Layer (DAL).
    """
    def __init__(self, num_classes=3):
        super(M2DFramework, self).__init__()
        
        # Encoder: Pre-trained EfficientNet-B3
        # Output feature channels from B3's final block is 1536
        self.encoder = models.efficientnet_b3(pretrained=False)
        
        # Decoder: UPerHead formulation (Skeleton structure for peer review)
        # Taking the 1536-channel feature map and projecting it to segmentation masks
        self.decoder = nn.Sequential(
            nn.Conv2d(1536, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Domain Adapter Layer (DAL) for cross-device feature alignment
        # Maps global pooled features to a domain-invariant space
        self.domain_adapter = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        # 1. Feature Extraction
        # EfficientNet features output shape: [Batch, 1536, H/32, W/32]
        features = self.encoder.features(x)
        
        # 2. Segmentation Branch
        seg_features = self.decoder(features)
        # Upsample the segmentation map back to original input resolution (512x512)
        seg_out = F.interpolate(seg_features, size=(x.shape[2], x.shape[3]), 
                                mode='bilinear', align_corners=False)
        
        # 3. Domain Adaptation Branch (Used for Global Alignment Loss)
        # Global Average Pooling (B, 1536, H', W' -> B, 1536)
        global_features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        domain_out = self.domain_adapter(global_features)
        
        return seg_out, domain_out

if __name__ == "__main__":
    # Sanity check for peer review
    print("Initializing M2D Framework...")
    model = M2DFramework()
    
    # Simulate a standardized OCT B-scan input: [Batch_size, Channels, Height, Width]
    dummy_input = torch.randn(1, 3, 512, 512)
    
    # Forward pass
    seg_output, domain_output = model(dummy_input)
    
    print("Forward pass successful!")
    print(f"-> Input shape: {dummy_input.shape}")
    print(f"-> Segmentation output shape: {seg_output.shape} (Matches input resolution)")
    print(f"-> Domain Adapter output shape: {domain_output.shape} (For global alignment loss)")

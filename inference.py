import torch
from torchvision import transforms
from PIL import Image
import argparse
from models.network import M2DFramework

def set_seed(seed=42):
    """Fix random seeds for deterministic reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_preprocessing_pipeline():
    """
    Strict preprocessing as defined in Section 2.3.2:
    1. Resize B-scans to 512x512 pixels.
    2. Normalize intensity to [0, 1] range.
    """
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(), # Automatically scales PIL image [0, 255] to [0.0, 1.0]
    ])

def main(args):
    # Enforce reproducibility
    set_seed(42)
    
    # Initialize hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")

    # Load Model
    model = M2DFramework(num_classes=3).to(device)
    
    # Placeholder for loading weights
    # model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print("M2D architecture loaded successfully.")
    
    # Optional: run a dummy inference to verify pipeline
    with torch.no_grad():
        dummy_scan = torch.randn(1, 3, 512, 512).to(device)
        seg_mask, domain_vector = model(dummy_scan)
        print(f"Inference verified. Mask shape generated: {seg_mask.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="M2D Framework Inference Script")
    parser.add_argument('--input', type=str, default='sample_oct.png', help='Path to input OCT scan')
    parser.add_argument('--weights', type=str, default='weights/m2d_pretrained.pth', help='Path to model weights')
    args = parser.parse_args()
    
    main(args)

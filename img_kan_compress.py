import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim
import torch.hub

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
MODEL_PATH = './compressed_model.pth'

class FourierFeatures(nn.Module):
    def __init__(self, in_features=1, mapping_size=64, scale=10.0):
        super(FourierFeatures, self).__init__()
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B  # (N, mapping_size)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(device).view(1,3,1,1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(device).view(1,3,1,1)
    def forward(self, input, target):
        if input.size(2) < 224 or input.size(3) < 224:
            input = F.interpolate(input, size=(224,224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224,224), mode='bilinear', align_corners=False)
        input_3 = input.repeat(1,3,1,1)
        target_3 = target.repeat(1,3,1,1)
        input_3 = (input_3 - self.mean) / self.std
        target_3 = (target_3 - self.mean) / self.std
        feat_input = self.vgg(input_3)
        feat_target = self.vgg(target_3)
        return F.mse_loss(feat_input, feat_target)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return F.relu(out + residual)

class PsiNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, output_dim=256, use_fourier=False, fourier_mapping_size=64, fourier_scale=10.0):
        super(PsiNetwork, self).__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = FourierFeatures(in_features=input_dim, mapping_size=fourier_mapping_size, scale=fourier_scale)
            in_dim = 2 * fourier_mapping_size
        else:
            in_dim = input_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        if self.use_fourier:
            x = self.fourier(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.fc2(x)

class PhiNetwork(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=1):
        super(PhiNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.res = ResidualBlock(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.res(x)
        return self.fc2(x)

class KolmogorovBlock(nn.Module):
    def __init__(self, hidden_dim=256, use_fourier=False, fourier_mapping_size=64, fourier_scale=10.0):
        super(KolmogorovBlock, self).__init__()
        self.psi_x = PsiNetwork(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim, 
                                  use_fourier=use_fourier, fourier_mapping_size=fourier_mapping_size, fourier_scale=fourier_scale)
        self.psi_y = PsiNetwork(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim,
                                  use_fourier=use_fourier, fourier_mapping_size=fourier_mapping_size, fourier_scale=fourier_scale)
        self.phi = PhiNetwork(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
    def forward(self, x, y):
        psi_x_out = self.psi_x(x)
        psi_y_out = self.psi_y(y)
        z = psi_x_out + psi_y_out
        return self.phi(z)

class KolmogorovImageCompressor(nn.Module):
    def __init__(self, num_blocks=20, hidden_dim=256, use_fourier=False, fourier_mapping_size=64, fourier_scale=10.0):
        super(KolmogorovImageCompressor, self).__init__()
        self.blocks = nn.ModuleList([
            KolmogorovBlock(hidden_dim, use_fourier, fourier_mapping_size, fourier_scale)
            for _ in range(num_blocks)
        ])
    def forward(self, coords):
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        return sum(block(x, y) for block in self.blocks)

class ImageDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super(ImageDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*2, features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*4, features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(features*8*4*4, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def load_image(path, size=(64, 64)):
    img = Image.open(path).convert("L")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(img)  # (1, H, W)

def create_coordinate_grid(H, W):
    xs = torch.linspace(-1, 1, steps=W)
    ys = torch.linspace(-1, 1, steps=H)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([grid_x, grid_y], dim=-1)
    return coords.view(-1, 2)

def save_compressed_model(model, path="compressed_model.pth"):
    torch.save(model.state_dict(), path)

def load_compressed_model(path="compressed_model.pth", num_blocks=20, hidden_dim=256, use_fourier=False, fourier_mapping_size=64, fourier_scale=10.0):
    model = KolmogorovImageCompressor(num_blocks=num_blocks, hidden_dim=hidden_dim,
                                      use_fourier=use_fourier, fourier_mapping_size=fourier_mapping_size, fourier_scale=fourier_scale).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def detect_roi_with_yolo(image_path, model_name='yolov5s', conf_threshold=0.5):
    """
    Detect regions of interest using YOLO and return a weight map
    """
    try:        
        print(f"Loading YOLO model: {model_name}")
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        model.to(device)
        model.eval()
        
        img = Image.open(image_path).convert("RGB")
        results = model(img)
        detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]
        
        # Create a weight map (same size as image, initialized to 1)
        weight_map = np.ones((img.size[1], img.size[0]))
        
        # Class-specific weights (higher weight = more important)
        class_weights = {
            'person': 5.0,  # Highest priority
            'face': 5.0,
            'car': 3.0,
            'truck': 3.0,
            'bus': 3.0,
            'motorcycle': 3.0,
            'bicycle': 3.0,
            'dog': 3.0,
            'cat': 3.0,
            'bird': 2.5,
            'horse': 2.5,
            'sheep': 2.5,
            'cow': 2.5,
            'elephant': 2.5,
            'bear': 2.5,
            'zebra': 2.5,
            'giraffe': 2.5,
            'backpack': 2.0,
            'umbrella': 2.0,
            'handbag': 2.0,
            'tie': 2.0,
            'suitcase': 2.0,
            'frisbee': 2.0,
            'skis': 2.0,
            'snowboard': 2.0,
            'sports ball': 2.0,
            'kite': 2.0,
            'baseball bat': 2.0,
            'baseball glove': 2.0,
            'skateboard': 2.0,
            'surfboard': 2.0,
            'tennis racket': 2.0,
            'bottle': 2.0,
            'wine glass': 2.0,
            'cup': 2.0,
            'fork': 2.0,
            'knife': 2.0,
            'spoon': 2.0,
            'bowl': 2.0,
            'banana': 2.0,
            'apple': 2.0,
            'sandwich': 2.0,
            'orange': 2.0,
            'broccoli': 2.0,
            'carrot': 2.0,
            'hot dog': 2.0,
            'pizza': 2.0,
            'donut': 2.0,
            'cake': 2.0,
            'chair': 2.0,
            'couch': 2.0,
            'potted plant': 2.0,
            'bed': 2.0,
            'dining table': 2.0,
            'toilet': 2.0,
            'tv': 2.0,
            'laptop': 2.0,
            'mouse': 2.0,
            'remote': 2.0,
            'keyboard': 2.0,
            'cell phone': 2.0,
            'microwave': 2.0,
            'oven': 2.0,
            'toaster': 2.0,
            'sink': 2.0,
            'refrigerator': 2.0,
            'book': 2.0,
            'clock': 2.0,
            'vase': 2.0,
            'scissors': 2.0,
            'teddy bear': 2.0,
            'hair drier': 2.0,
            'toothbrush': 2.0
        }
        
        # For each detection above confidence threshold
        roi_count = 0
        for det in detections:
            if det[4] >= conf_threshold:
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                class_idx = int(det[5])
                class_name = model.names[class_idx]
                
                # Get weight for this class
                weight = class_weights.get(class_name, 2.0)  # Default weight if class not specified
                
                # Create a Gaussian-weighted mask for smooth transitions
                center_y = (y1 + y2) // 2
                center_x = (x1 + x2) // 2
                radius_y = (y2 - y1) // 2 + 1
                radius_x = (x2 - x1) // 2 + 1
                
                y_grid, x_grid = np.ogrid[:weight_map.shape[0], :weight_map.shape[1]]
                mask = np.exp(-((x_grid - center_x)**2 / (2 * radius_x**2) + 
                               (y_grid - center_y)**2 / (2 * radius_y**2)))
                
                # Apply the weight with Gaussian smoothing
                weight_map = np.maximum(weight_map, mask * weight)
                roi_count += 1
                
        print(f"Detected {roi_count} regions of interest")
        return weight_map
        
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        # Return uniform weights if YOLO fails
        img = Image.open(image_path)
        return np.ones((img.size[1], img.size[0]))

def compress_image(image_path,
                   model_save_path="compressed_model.pth",
                   image_size=(64, 64),
                   num_blocks=20,
                   hidden_dim=256,
                   use_fourier=True,
                   fourier_mapping_size=64,
                   fourier_scale=10.0,
                   epochs=15000,
                   lr=1e-3,
                   mse_weight=1.0,
                   l1_weight=0.5,
                   perceptual_weight=0.8,
                   adv_weight=0.005,
                   use_adversarial=True,
                   use_roi=True):
    """
    Train generator on image with combined loss function and ROI prioritization.
    If model already exists, load it instead of training from scratch.
    """
    if os.path.exists(model_save_path):
        print(f"Model {model_save_path} already exists. Loading existing model.")
        return load_compressed_model(model_save_path, num_blocks=num_blocks, hidden_dim=hidden_dim,
                                     use_fourier=use_fourier, fourier_mapping_size=fourier_mapping_size, fourier_scale=fourier_scale)
    
    print("Loading image...")
    img_tensor = load_image(image_path, size=image_size)
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    coords = create_coordinate_grid(H, W).to(device)
    target = img_tensor.view(-1, 1).to(device)
    
    # Get ROI weight map if enabled
    if use_roi:
        print("Detecting regions of interest with YOLO...")
        weight_map = detect_roi_with_yolo(image_path)
        # Resize weight map to match image_size
        weight_map = cv2.resize(weight_map, image_size, interpolation=cv2.INTER_LINEAR)
        weight_tensor = torch.from_numpy(weight_map).float().view(-1, 1).to(device)
        
        # Visualize the weight map
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img_tensor.squeeze().numpy(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(weight_map, cmap='hot')
        plt.title('ROI Weight Map')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(img_tensor.squeeze().numpy(), cmap='gray')
        plt.imshow(weight_map, cmap='hot', alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        weight_tensor = torch.ones_like(target)
    
    generator = KolmogorovImageCompressor(num_blocks=num_blocks, hidden_dim=hidden_dim,
                                          use_fourier=use_fourier, fourier_mapping_size=fourier_mapping_size, fourier_scale=fourier_scale).to(device)
    optimizer_G = optim.AdamW(generator.parameters(), lr=lr)
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs)
    
    # Define weighted loss functions
    def weighted_mse_loss(input, target, weights):
        return (weights * (input - target) ** 2).mean()
    
    def weighted_l1_loss(input, target, weights):
        return (weights * torch.abs(input - target)).mean()
    
    perceptual_loss_fn = PerceptualLoss(device)
    
    if use_adversarial:
        discriminator = ImageDiscriminator(in_channels=1, features=64).to(device)
        optimizer_D = optim.AdamW(discriminator.parameters(), lr=lr)
        bce_loss_fn = nn.BCELoss()
    
    print("Starting model training...")
    for epoch in range(epochs):
        generator.train()
        optimizer_G.zero_grad()
        
        output = generator(coords)
        fake_img = output.view(1, 1, H, W)
        real_img = target.view(1, 1, H, W)
        
        # Use weighted losses
        loss_mse = weighted_mse_loss(output, target, weight_tensor)
        loss_l1 = weighted_l1_loss(output, target, weight_tensor)
        loss_perc = perceptual_loss_fn(fake_img, real_img)
        
        loss_adv = 0.0
        if use_adversarial:
            pred_fake = discriminator(fake_img)
            valid = torch.ones_like(pred_fake, device=device)
            loss_adv = bce_loss_fn(pred_fake, valid)
        
        loss_G = mse_weight * loss_mse + l1_weight * loss_l1 + perceptual_weight * loss_perc + adv_weight * loss_adv
        loss_G.backward()
        optimizer_G.step()
        scheduler_G.step()
        
        if use_adversarial:
            optimizer_D.zero_grad()
            pred_real = discriminator(real_img)
            valid = torch.ones_like(pred_real, device=device)
            loss_real = bce_loss_fn(pred_real, valid)
            pred_fake = discriminator(fake_img.detach())
            fake = torch.zeros_like(pred_fake, device=device)
            loss_fake = bce_loss_fn(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()
        
        if epoch % 1000 == 0:
            if use_adversarial:
                print(f"Epoch {epoch}: Loss_G = {loss_G.item():.6f} (MSE={loss_mse.item():.6f}, L1={loss_l1.item():.6f}, Perc={loss_perc.item():.6f}, Adv={loss_adv.item():.6f}), Loss_D = {loss_D.item():.6f}")
            else:
                print(f"Epoch {epoch}: Loss_G = {loss_G.item():.6f} (MSE={loss_mse.item():.6f}, L1={loss_l1.item():.6f}, Perc={loss_perc.item():.6f})")
    print("Training completed. Saving model...")
    save_compressed_model(generator, model_save_path)
    print(f"Model saved to {model_save_path}")
    return generator

def decompress_image(model_path,
                     image_size=(64, 64),
                     num_blocks=20,
                     hidden_dim=256,
                     use_fourier=True,
                     fourier_mapping_size=64,
                     fourier_scale=10.0):
    H, W = image_size
    coords = create_coordinate_grid(H, W).to(device)
    generator = load_compressed_model(model_path, num_blocks=num_blocks, hidden_dim=hidden_dim,
                                      use_fourier=use_fourier, fourier_mapping_size=fourier_mapping_size, fourier_scale=fourier_scale)
    generator.eval()
    with torch.no_grad():
        reconstructed = generator(coords).view(H, W).cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed, cmap="gray")
    plt.axis("off")
    plt.show()
    return reconstructed

def compress_to_jpeg(image_path, quality=10, output_path="compressed.jpg"):
    img = cv2.imread(image_path)
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print(f"Image compressed to JPEG and saved as {output_path}")

def evaluate_compression(original_path, decompressed_path):
    orig = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    decompressed = cv2.imread(decompressed_path, cv2.IMREAD_GRAYSCALE)
    psnr_value = cv2.PSNR(orig, decompressed)
    ssim_value = ssim(orig, decompressed)
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")

def postprocess_image(reconstructed, upscale_factor=2):
    img_uint8 = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)
    upscaled = cv2.resize(img_uint8, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(upscaled, None, h=10, templateWindowSize=7, searchWindowSize=21)
    equalized = cv2.equalizeHist(denoised)
    return equalized

def fine_tune_image(image_path,
                    model_path="compressed_model.pth",
                    fine_tune_epochs=5000,
                    lr=1e-4,
                    mse_weight=1.0,
                    l1_weight=0.5,
                    perceptual_weight=0.5,
                    adv_weight=0.005,
                    use_adversarial=True,
                    use_roi=True,
                    num_blocks=20,
                    hidden_dim=256,
                    use_fourier=True,
                    fourier_mapping_size=64,
                    fourier_scale=10.0,
                    image_size=(64, 64)):

    print("Loading image for fine-tuning...")
    img_tensor = load_image(image_path, size=image_size)
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    coords = create_coordinate_grid(H, W).to(device)
    target = img_tensor.view(-1, 1).to(device)

    # Get ROI weight map if enabled
    if use_roi:
        print("Detecting regions of interest with YOLO for fine-tuning...")
        weight_map = detect_roi_with_yolo(image_path)
        weight_map = cv2.resize(weight_map, image_size, interpolation=cv2.INTER_LINEAR)
        weight_tensor = torch.from_numpy(weight_map).float().view(-1, 1).to(device)
    else:
        weight_tensor = torch.ones_like(target)

    generator = load_compressed_model(model_path, num_blocks=num_blocks, hidden_dim=hidden_dim,
                                      use_fourier=use_fourier, fourier_mapping_size=fourier_mapping_size, fourier_scale=fourier_scale)
    generator.train()

    optimizer_G = optim.AdamW(generator.parameters(), lr=lr)
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=fine_tune_epochs)
    
    # Define weighted loss functions
    def weighted_mse_loss(input, target, weights):
        return (weights * (input - target) ** 2).mean()
    
    def weighted_l1_loss(input, target, weights):
        return (weights * torch.abs(input - target)).mean()
    
    perceptual_loss_fn = PerceptualLoss(device)

    if use_adversarial:
        discriminator = ImageDiscriminator(in_channels=1, features=64).to(device)
        optimizer_D = optim.AdamW(discriminator.parameters(), lr=lr)
        bce_loss_fn = nn.BCELoss()
    
    print("Starting fine-tuning...")
    for epoch in range(fine_tune_epochs):
        generator.train()
        optimizer_G.zero_grad()
        
        output = generator(coords)
        fake_img = output.view(1, 1, H, W)
        real_img = target.view(1, 1, H, W)
        
        # Use weighted losses
        loss_mse = weighted_mse_loss(output, target, weight_tensor)
        loss_l1 = weighted_l1_loss(output, target, weight_tensor)
        loss_perc = perceptual_loss_fn(fake_img, real_img)
        
        loss_adv = 0.0
        if use_adversarial:
            pred_fake = discriminator(fake_img)
            valid = torch.ones_like(pred_fake, device=device)
            loss_adv = bce_loss_fn(pred_fake, valid)
        
        loss_G = mse_weight * loss_mse + l1_weight * loss_l1 + perceptual_weight * loss_perc + adv_weight * loss_adv
        loss_G.backward()
        optimizer_G.step()
        scheduler_G.step()
        
        if use_adversarial:
            optimizer_D.zero_grad()
            pred_real = discriminator(real_img)
            valid = torch.ones_like(pred_real, device=device)
            loss_real = bce_loss_fn(pred_real, valid)
            pred_fake = discriminator(fake_img.detach())
            fake = torch.zeros_like(pred_fake, device=device)
            loss_fake = bce_loss_fn(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()
        
        if epoch % 200 == 0:
            if use_adversarial:
                print(f"Epoch {epoch}: Loss_G = {loss_G.item():.6f} (MSE={loss_mse.item():.6f}, L1={loss_l1.item():.6f}, Perc={loss_perc.item():.6f}, Adv={loss_adv.item():.6f}), Loss_D = {loss_D.item():.6f}")
            else:
                print(f"Epoch {epoch}: Loss_G = {loss_G.item():.6f} (MSE={loss_mse.item():.6f}, L1={loss_l1.item():.6f}, Perc={loss_perc.item():.6f})")
    
    print("Fine-tuning completed. Saving updated model...")
    save_compressed_model(generator, model_path)
    print(f"Updated model saved to {model_path}")


if __name__ == "__main__":
    image_path = input('Enter path to image: ')
    model_save_path = "compressed_model.pth"
    jpeg_output_path = "compressed.jpg"
    image_size = (64, 64)
    
    if not os.path.exists(image_path):
        print(f"File {image_path} not found. Place the image in the working folder.")
        exit(1)
    
    # Ask if user wants to use ROI detection
    use_roi = input("Use YOLO for region-of-interest detection? (y/n): ").strip().lower() == 'y'
    
    # Compress the image with or without ROI
    compress_image(image_path,
                   model_save_path=model_save_path,
                   image_size=image_size,
                   num_blocks=20,
                   hidden_dim=256,
                   use_fourier=True,
                   fourier_mapping_size=64,
                   fourier_scale=10.0,
                   epochs=15000,
                   lr=1e-3,
                   mse_weight=1.0,
                   l1_weight=0.5,
                   perceptual_weight=0.8,
                   adv_weight=0.005,
                   use_adversarial=True,
                   use_roi=use_roi)
    
    # Ask if user wants to fine-tune
    fine_tune = input("Perform model fine-tuning? (y/n): ").strip().lower()
    if fine_tune == 'y':
        fine_tune_image(image_path,
                        model_path=model_save_path,
                        fine_tune_epochs=5000,
                        lr=1e-4,
                        mse_weight=1.0,
                        l1_weight=0.5,
                        perceptual_weight=0.5,
                        adv_weight=0.005,
                        use_adversarial=True,
                        use_roi=use_roi,
                        num_blocks=20,
                        hidden_dim=256,
                        use_fourier=True,
                        fourier_mapping_size=64,
                        fourier_scale=10.0,
                        image_size=image_size)
    
    # Decompress and show results
    reconstructed = decompress_image(model_save_path,
                     image_size=image_size,
                     num_blocks=20,
                     hidden_dim=256,
                     use_fourier=True,
                     fourier_mapping_size=64,
                     fourier_scale=10.0)
    
    postprocessed = postprocess_image(reconstructed, upscale_factor=2)
    plt.figure(figsize=(6,6))
    plt.title("Post-processed Image")
    plt.imshow(postprocessed, cmap="gray")
    plt.axis("off")
    plt.show()
    
    # Compare with JPEG compression
    compress_to_jpeg(image_path, quality=10, output_path=jpeg_output_path)
    evaluate_compression(image_path, jpeg_output_path)

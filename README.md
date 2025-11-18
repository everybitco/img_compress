This project implements an image compression algorithm inspired by the Kolmogorov–Arnold representation theorem.

#### High-level overview (what the system does)

```
1. Load an input image and convert it to a grayscale tensor.
2. Optionally detect regions of interest (ROIs) using a YOLO model and build a smooth weight map that prioritizes important regions.
3. Create a coordinate grid of normalized (x, y) pairs in [-1, 1] for every pixel.
4. Train a coordinate-based implicit generator (KolmogorovImageCompressor) to map coordinates -> pixel values using a combined loss:
   - Weighted MSE and weighted L1 (pixel-wise, modulated by the ROI map)
   - Perceptual loss computed on duplicated grayscale -> 3-channel inputs using VGG16 features
   - Optional adversarial loss using a convolutional discriminator
5. Save the trained model state_dict as the compressed representation.
6. Decompress by loading the model, evaluating it on the coordinate grid, and reconstructing the image.
7. Optionally apply post-processing (upscale, denoise, equalize) and evaluate JPEG baseline (PSNR, SSIM).

```
### 1) Image loading and preprocessing
- The script opens the image with PIL and converts it to grayscale (`convert("L")`).
- The image is resized to `image_size` and transformed to a tensor with shape (1, H, W).
- A flattened target vector is created (N × 1) where N = H × W.

### 2) Coordinate grid construction
- A regular grid of coordinates is created with torch.linspace and torch.meshgrid, normalized to [-1, 1].
- Coordinates are flattened to shape (N, 2) and passed to the generator during training and inference.

### 3) ROI detection and weight map creation (optional)
- If enabled, `detect_roi_with_yolo(image_path)` runs:
  - Loads YOLO via `torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)`. Note: first call downloads repo/weights and requires internet.
  - Runs inference on the original RGB image and collects detections (x1, y1, x2, y2, confidence, class_idx).
  - Initializes a weight map of ones with same height/width as the input image.
  - For each detection whose confidence ≥ `conf_threshold`:
    - Class lookup maps object name -> a configured importance weight (e.g., 'person' -> 5.0).
    - Generates a smooth Gaussian mask centered on the detection box with sigma proportional to box size.
    - Combines the mask into the global weight map by taking element-wise maximum with current values: weight_map = max(weight_map, mask * class_weight).
  - Returns a floating weight map (H_orig × W_orig). If any error occurs (no model, offline), the code catches the exception and returns a uniform map of ones.
- After detection, the weight map is resized to the model `image_size` using OpenCV (`cv2.resize`). Important: cv2.resize expects size=(width, height); the code must pass (W, H) to avoid dimension swap.

### 4) Weighted losses
- Weighted MSE: mean over all pixels of weight * (pred - target)^2.
- Weighted L1: mean over all pixels of weight * |pred - target|.
- Perceptual loss:
  - Convert single-channel images to 3-channel by repeating channel dimension.
  - Normalize using VGG mean/std, resize to 224×224, and pass the result through the first layers of pretrained VGG16.
  - Compute MSE between VGG features of real and fake images.
- Adversarial loss (optional):
  - `ImageDiscriminator` is a small conv-net that down-samples the image (stride=2 convs), flattens and outputs a single sigmoid score.
  - For stability, the generator is trained to maximize `log(D(fake))` via BCE with `valid` labels; the discriminator is trained with `real`=1 and `fake`=0.

### 5) Model architecture (generator)
- KolmogorovImageCompressor:
  - A list of `num_blocks` KolmogorovBlock instances.
  - Each KolmogorovBlock computes psi_x(x) and psi_y(y) with identical Psi networks and then feeds psi_x + psi_y into a Phi network to produce a scalar output for each coordinate.
  - The generator sums outputs of all blocks to produce the final intensity for each coordinate.
- Psi networks can use Fourier features:
  - Coordinates are projected by a fixed random matrix B, then sin/cos applied and concatenated. This expands the input to a higher-dimensional sinusoidal embedding allowing the MLPs to learn high-frequency detail easier.
- Residual blocks and BatchNorm1d are used within MLPs for stability and performance.

### 6) Training loop (per epoch)
- Compute generator output on the full coordinate grid (single large batch N).
- Reshape output to (1, 1, H, W) for perceptual and discriminator inputs.
- Compute weighted MSE and weighted L1 using the ROI weight tensor (shape N×1).
- Compute perceptual loss on 3-channel rescaled images.
- If adversarial enabled:
  - Compute generator adversarial loss via discriminator(fake).
  - Update discriminator: BCE on real images (label 1) and fake images (label 0).
- Total generator loss = mse_weight * weighted_mse + l1_weight * weighted_l1 + perceptual_weight * perc_loss + adv_weight * adv_loss.
- Backpropagate the generator loss, optimizer step, scheduler step.
- Periodically print training statistics.

### 7) Saving the compressed representation
- The compressed artifact is the generator's state_dict saved with `torch.save(model.state_dict(), path)`.
- Recommendation: save a small JSON alongside containing hyperparameters (num_blocks, hidden_dim, use_fourier, image_size) so decompression reconstructs the model identically.

### 8) Decompression (reconstruction)
- Load the model with the same hyperparameters, load state_dict, set model.eval().
- Recompute the coordinate grid for the target image_size, feed through model to produce N outputs, reshape to H×W, and convert to numpy for display/saving.

### 9) Fine-tuning
- Loads the saved state_dict, optionally recomputes ROI weights, and runs a smaller training loop (fine-tune epochs and reduced LR).
- Uses the same combined, ROI-weighted losses and optionally adversarial updates.

### 10) Post-processing and evaluation
- Post-processing pipeline:
  - Multiply reconstructed float [0..1] by 255 -> uint8.
  - Upscale by factor (bicubic).
  - Denoise with OpenCV fastNlMeansDenoising.
  - Histogram equalization to enhance contrast.
- JPEG baseline:
  - Save original via OpenCV with specified quality and compute PSNR/SSIM between original and JPEG using cv2.PSNR and skimage.metrics.ssim.

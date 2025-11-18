## Implementation notes, assumptions and caveats
- cv2.resize size parameter: OpenCV requires (width, height). The script must pass (W, H) when resizing weight maps or images. Keep consistent conventions: elsewhere `image_size` is usually (H, W), so convert when calling cv2.resize.
- Discriminator flatten shape: the final linear expects a specific flattened size (features*8*4*4). This matches 64×64 input with four 2× downsampling layers. If you change image resolution, either adapt the linear layer size dynamically or compute tensor shape at runtime.
- VGG pretrained: torchvision versions may deprecate `pretrained=True` in favor of `weights=...`. Expect deprecation warnings on newer torchvision.
- YOLO dependency: `torch.hub.load('ultralytics/yolov5', ...)` downloads code and weights on first run. If the environment is offline, detection will fail and the script falls back to uniform weights.
- I want to use moondream for ROI but maybe some other time.
- BatchNorm with single large batch training: BatchNorm1d uses batch statistics over the N coordinate points. Because training uses the full image as a single batch, BatchNorm will still see many samples, but behavior differs if batch sizes become small. Consider GroupNorm if behavior is unstable.
- Mixed precision: using `torch.cuda.amp` can speed up GPU training and reduce memory usage.
- the Fourier random matrix B is stored as a non-learnable Parameter in the model. If you want identical compressed files across runs, ensure deterministic seeds are used before model initialization or save the B matrix/seed in model metadata.

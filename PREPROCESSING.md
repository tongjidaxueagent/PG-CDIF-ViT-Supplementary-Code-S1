# Image preprocessing and augmentation specification

This document addresses image-value preprocessing (S01) and training augmentation (S02). Coordinate normalization for the attention-distance analysis is described separately in `METHODS_AND_DATA.md`.

## S01 image-value normalization

Images are converted to RGB. Configured tilt correction, when nonzero, precedes ROI cropping; resize then gives 224 × 224 pixels. The inspected tilt correction uses an expanded canvas, bicubic resampling and an RGB fill of (128,128,128). The subsequent torchvision resize uses bilinear interpolation. Full per-group ROI parameters and raw images are outside this archive; the settings here are not a complete raw-image reproduction package.

For channel c, the pixel calculation is:

```text
z[c] = (I_uint8[c] / 255 - mean[c]) / std[c]
```

| RGB channel | mean | std |
|---|---:|---:|
| R | 0.485 | 0.229 |
| G | 0.456 | 0.224 |
| B | 0.406 | 0.225 |

These are fixed ImageNet constants, not estimates from this study's training images. The constants are shared by side and top views and by training, validation and test sets. Pixel values lie in [0,1] after scaling and before standardization; the standardized values are not restricted to that interval.

An illustrative RGB pixel (0,128,255) becomes approximately (-2.11790393, 0.20518207, 2.64000000) in double-precision arithmetic. Original float32 tensor arithmetic differs in its last digits. This artificial pixel is only an arithmetic example; it is not presented as a study measurement.

## S02 training augmentation

After deterministic RGB/tilt/ROI processing, the training composition is:

```text
Resize -> HorizontalFlip -> RandAugment -> ToTensor -> Normalize -> RandomErasing
```

| Operation | Effective configuration |
|---|---|
| Horizontal flip | Probability 0.5 |
| RandAugment | Two operations, magnitude index 9 on 31 bins; the transform is invoked for every training image |
| Independent ColorJitter step | Disabled |
| Random erasing | Attempt probability 0.3; area fraction [0.02,0.15]; aspect ratio [0.3,3.3] |
| Separate additive Gaussian-noise step | Absent |

The RandAugment pool contains Identity, ShearX, ShearY, TranslateX, TranslateY, Rotate, Brightness, Color, Contrast, Sharpness, Posterize, Solarize, AutoContrast and Equalize. Two operations are selected uniformly with replacement and applied sequentially; an operation can occur twice. Geometric operations can change spatial structure, so no invariant-interface or invariant-shape claim is made.

Each named operation has selection probability 1/14 per draw. Brightness and Contrast each appear at least once with probability `1-(13/14)^2 = 27/196`, approximately 13.78%. They are not independent Bernoulli switches. The probability that both occur in the two slots is `2/196`, approximately 1.02%. These are theoretical selection probabilities implied by the implementation, not empirical frequencies recovered from training logs. Each selected Brightness/Contrast operation uses an approximate factor of 0.73 or 1.27, with equal sign probabilities.

Random erasing samples area uniformly and aspect ratio log-uniformly. A maximum of ten rectangle proposals are tried; if none is valid, the image is retained. The selected rectangle is replaced by independent standard-normal values in the already standardized tensor space. This is not full-image additive Gaussian noise.

The same composition is called separately for side and top images, drawing new random parameters for each view. Parameters are not synchronized. Validation/test use deterministic RGB/tilt/ROI processing, resize, ToTensor and the same normalization, without the random augmentation steps.

## Evidence and limits

The audited environment contained PyTorch 2.9.1+cu128 and torchvision 0.24.1+cu128. These are evidence for interpretation of the archived implementation, not dependencies of the supplied verification scripts. The original data-loader factory forwarded resize settings but relied on transform-function defaults for normalization and advanced augmentation; the specification describes effective behavior rather than assuming all YAML settings were wired through.

Historical per-image random choices were not logged and cannot be replayed from this archive. No training rerun or evaluation of augmentation efficacy is claimed. The machine-readable values are in `config/preprocessing_spec.json`; the runner checks one pixel example and the operation-selection probability only.

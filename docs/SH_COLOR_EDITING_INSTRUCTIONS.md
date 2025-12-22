# Spherical Harmonics Color Editing in gsmod

## Overview

This document describes the mathematically correct approach to editing colors in 3D Gaussian Splatting data that uses spherical harmonics (SH) for view-dependent color representation.

## Table of Contents

1. [Background: Spherical Harmonics in 3DGS](#background-spherical-harmonics-in-3dgs)
2. [The Problem with Current Implementation](#the-problem-with-current-implementation)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Classification of Color Operations](#classification-of-color-operations)
5. [Implementation Guide](#implementation-guide)
6. [API Reference](#api-reference)
7. [Testing Requirements](#testing-requirements)

---

## Background: Spherical Harmonics in 3DGS

### What is Stored

In 3D Gaussian Splatting, each Gaussian stores color as spherical harmonics coefficients:

| Field | Shape | Description |
|-------|-------|-------------|
| `sh0` (f_dc) | `[N, 3]` | DC component (degree 0) - base color |
| `shN` (f_rest) | `[N, K, 3]` | Higher-order coefficients (degrees 1-3) |

Where `K` depends on SH degree:

- Degree 1: K = 3 coefficients
- Degree 2: K = 8 coefficients
- Degree 3: K = 15 coefficients

### How Color is Computed During Rendering

For a viewing direction **d**, the final color is:

```
color(d) = C₀ · sh0 + Σ(Cₗ · shN[l] · Yₗₘ(d))
```

Where:

- `C₀ = 0.28209479177387814` is the DC normalization constant
- `Cₗ` are band normalization constants
- `Yₗₘ(d)` are spherical harmonic basis functions

### Key Insight: What Each Band Represents

- **Band 0 (DC/sh0)**: The **average color** across all viewing directions
- **Band 1**: Linear directional variation (3 coefficients)
- **Band 2**: Quadratic directional variation (5 coefficients)
- **Band 3**: Cubic directional variation (7 coefficients)

The DC term represents the integral of color over the sphere:

```
sh0 = (1/4π) ∫ color(d) dΩ
```

---

## The Problem with Current Implementation

### Current Behavior (Broken)

```python
# Current adjust_saturation - WRONG!
def adjust_saturation(self, factor, inplace=True):
    if not self.is_sh0_rgb:
        self.to_rgb(inplace=True)  # Converts sh0 only!

    gray = compute_luminance(self.sh0)
    self.sh0 = gray + (self.sh0 - gray) * factor
    self.sh0.clamp_(0, 1)
    # shN is left unchanged in SH space!
```

### What Goes Wrong

1. **sh0 is converted to RGB** (values clamped to [0,1])
2. **shN remains in SH space** (values can be any real number)
3. **Inconsistent state**: Renderer expects both in same space
4. **Data corruption**: View-dependent effects are destroyed

### Test That Demonstrates the Bug

```python
t = GSTensorPro(sh0=torch.randn(10, 3) * 2, shN=torch.randn(10, 15, 3))
print(f"Before: sh0 range [{t.sh0.min():.2f}, {t.sh0.max():.2f}]")
print(f"Before: is_sh0_rgb = {t.is_sh0_rgb}")

t.adjust_saturation(1.3, inplace=True)

print(f"After: sh0 range [{t.sh0.min():.2f}, {t.sh0.max():.2f}]")  # [0, 1]
print(f"After: is_sh0_rgb = {t.is_sh0_rgb}")  # True
print(f"After: shN unchanged = {(t.shN == original_shN).all()}")  # True - BUG!
```

---

## Mathematical Foundation

### Key Revelation: Saturation is a Linear Operation

The saturation formula appears nonlinear:

```
c' = L + s·(c - L)
```

Where `L = 0.299·R + 0.587·G + 0.114·B` is luminance.

But expanding for each channel:

```
R' = s·R + (1-s)·(0.299·R + 0.587·G + 0.114·B)
   = R·(s + (1-s)·0.299) + G·(1-s)·0.587 + B·(1-s)·0.114
```

This is a **3×3 matrix multiplication**:

```
[R']   [s+(1-s)·w_r  (1-s)·w_g  (1-s)·w_b] [R]
[G'] = [(1-s)·w_r  s+(1-s)·w_g  (1-s)·w_b] [G]
[B']   [(1-s)·w_r  (1-s)·w_g  s+(1-s)·w_b] [B]
```

Where `w = [0.299, 0.587, 0.114]` are luminance weights.

### Why Linear Operations Work on SH

Because SH is a linear representation:

```
color(d) = Σ coefficients · basis_functions(d)
```

Any linear operation `f(x) = M·x + b` can be applied to the coefficients:

```
f(color(d)) = M · color(d) + b
            = M · Σ(c_i · Y_i(d)) + b
            = Σ(M·c_i · Y_i(d)) + b
```

For the offset `b`:

- **DC term**: Add `b/C₀` to sh0
- **Higher-order terms**: Add nothing (they represent variation, not absolute value)

---

## Classification of Color Operations

### Category 1: Multiplicative Scaling

Operations of the form `c' = k · c`

| Operation | DC (sh0) | Higher-Order (shN) |
|-----------|----------|-------------------|
| Brightness | `sh0 *= k` | `shN *= k` |
| Exposure | `sh0 *= 2^ev` | `shN *= 2^ev` |

**Implementation**:

```python
def adjust_brightness(self, factor):
    self.sh0 *= factor
    if self.shN is not None:
        self.shN *= factor
    # Only clamp if RGB format
    if self.is_sh0_rgb:
        self.sh0.clamp_(0, 1)
```

### Category 2: Additive Offset

Operations of the form `c' = c + b`

| Operation | DC (sh0) | Higher-Order (shN) |
|-----------|----------|-------------------|
| Black Point | `sh0 += b/C₀` (SH) or `sh0 += b` (RGB) | No change |
| Fade | `sh0 += fade/C₀` (SH) or `sh0 += fade` (RGB) | No change |

**Why shN is unchanged**: The offset only affects the average color, not the directional variation.

**Implementation**:

```python
def adjust_fade(self, amount):
    if self.is_sh0_rgb:
        self.sh0 += amount
        self.sh0.clamp_(0, 1)
    else:
        self.sh0 += amount / SH_C0
    # shN unchanged
```

### Category 3: Linear Matrix (3×3 Transform)

Operations of the form `c' = M · c`

| Operation | DC (sh0) | Higher-Order (shN) |
|-----------|----------|-------------------|
| Saturation | `sh0 = sh0 @ M.T` | `shN = shN @ M.T` |
| Temperature | `sh0 = sh0 @ M.T` | `shN = shN @ M.T` |
| Tint | `sh0 = sh0 @ M.T` | `shN = shN @ M.T` |
| Hue Shift | `sh0 = sh0 @ M.T` | `shN = shN @ M.T` |

**Implementation**:

```python
def adjust_saturation(self, factor):
    M = build_saturation_matrix(factor)
    self.sh0 = self.sh0 @ M.T  # [N, 3] @ [3, 3] = [N, 3]
    if self.shN is not None:
        self.shN = self.shN @ M.T  # [N, K, 3] @ [3, 3] = [N, K, 3]
    if self.is_sh0_rgb:
        self.sh0.clamp_(0, 1)
```

### Category 4: Combined (Scale + Offset)

Operations of the form `c' = k · c + b`

| Operation | DC (sh0) | Higher-Order (shN) |
|-----------|----------|-------------------|
| Contrast | `sh0 = sh0 * k + offset` | `shN *= k` (no offset) |

**Implementation**:

```python
def adjust_contrast(self, factor):
    if self.is_sh0_rgb:
        self.sh0 = (self.sh0 - 0.5) * factor + 0.5
        self.sh0.clamp_(0, 1)
    else:
        offset = 0.5 * (1 - factor) / SH_C0
        self.sh0 = self.sh0 * factor + offset

    if self.shN is not None:
        self.shN *= factor  # Scale only, no offset
```

### Category 5: Nonlinear Operations

These cannot be exactly applied in SH space.

| Operation | Approach |
|-----------|----------|
| Gamma | Approximate: apply to DC, scale shN proportionally |
| Vibrance | Approximate: use uniform saturation with damping |
| Shadows/Highlights | Approximate: per-Gaussian scale based on luminance |

**Implementation Strategy**:

```python
def adjust_gamma(self, gamma):
    if not self.has_high_order_sh:
        # No shN: apply directly
        self.to_rgb(inplace=True)
        self.sh0 = self.sh0.pow(gamma).clamp(0, 1)
    else:
        # Has shN: approximate by scaling
        sh0_rgb = sh_to_rgb(self.sh0)
        lum_before = compute_luminance(sh0_rgb)

        sh0_rgb_gamma = sh0_rgb.clamp(1e-6, 1).pow(gamma)
        lum_after = compute_luminance(sh0_rgb_gamma)

        scale = lum_after / (lum_before + 1e-8)

        self.sh0 = rgb_to_sh(sh0_rgb_gamma)
        self.shN = self.shN * scale.unsqueeze(1)
```

---

## Implementation Guide

### Constants

```python
# SH DC normalization constant
SH_C0 = 0.28209479177387814

# Luminance weights (Rec. 709)
LUMA_R = 0.299
LUMA_G = 0.587
LUMA_B = 0.114
LUMA_WEIGHTS = torch.tensor([LUMA_R, LUMA_G, LUMA_B])
```

### Conversion Functions

```python
def sh_to_rgb(sh: Tensor) -> Tensor:
    """Convert SH DC to RGB."""
    return sh * SH_C0 + 0.5

def rgb_to_sh(rgb: Tensor) -> Tensor:
    """Convert RGB to SH DC."""
    return (rgb - 0.5) / SH_C0
```

### Matrix Builders

#### Saturation Matrix

```python
def build_saturation_matrix(factor: float, device) -> Tensor:
    """
    Build 3x3 saturation matrix.

    M = (1-s) * outer([1,1,1], [w_r, w_g, w_b]) + s * I

    Example for s=1.3:
    [[1.2103, -0.1761, -0.0342],
     [-0.0897, 1.1239, -0.0342],
     [-0.0897, -0.1761, 1.2658]]
    """
    s = factor
    w = torch.tensor([0.299, 0.587, 0.114], device=device)
    M = (1 - s) * torch.ones(3, 1, device=device) @ w.unsqueeze(0)
    M = M + s * torch.eye(3, device=device)
    return M
```

#### Temperature Matrix

```python
def build_temperature_matrix(temp: float, device) -> Tensor:
    """
    Build temperature adjustment matrix.

    Positive = warmer (boost red, reduce blue)
    Negative = cooler (reduce red, boost blue)
    """
    return torch.diag(torch.tensor([
        1.0 + temp * 0.1,
        1.0,
        1.0 - temp * 0.1
    ], device=device))
```

#### Tint Matrix

```python
def build_tint_matrix(tint: float, device) -> Tensor:
    """
    Build tint adjustment matrix.

    Positive = magenta (reduce green)
    Negative = green (boost green)
    """
    return torch.diag(torch.tensor([
        1.0,
        1.0 - tint * 0.1,
        1.0
    ], device=device))
```

#### Hue Rotation Matrix

```python
def build_hue_matrix(degrees: float, device) -> Tensor:
    """
    Build luminance-preserving hue rotation matrix.
    """
    theta = math.radians(degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    return torch.tensor([
        [0.213 + 0.787*cos_t - 0.213*sin_t,
         0.715 - 0.715*cos_t - 0.715*sin_t,
         0.072 - 0.072*cos_t + 0.928*sin_t],
        [0.213 - 0.213*cos_t + 0.143*sin_t,
         0.715 + 0.285*cos_t + 0.140*sin_t,
         0.072 - 0.072*cos_t - 0.283*sin_t],
        [0.213 - 0.213*cos_t - 0.787*sin_t,
         0.715 - 0.715*cos_t + 0.715*sin_t,
         0.072 + 0.928*cos_t + 0.072*sin_t]
    ], device=device)
```

### Order of Operations

Apply color adjustments in this order for predictable results:

1. **White Balance** (Temperature, Tint)
2. **Exposure/Brightness**
3. **Contrast**
4. **Shadows/Highlights**
5. **Saturation/Vibrance**
6. **Hue Shift**
7. **Fade**
8. **Gamma** (last, as it's nonlinear)

---

## API Reference

### Properties

```python
@property
def is_sh0_rgb(self) -> bool:
    """True if sh0 is in RGB format [0,1], False if in SH format."""

@property
def is_sh0_sh(self) -> bool:
    """True if sh0 is in SH format (unbounded), False if in RGB format."""

@property
def has_high_order_sh(self) -> bool:
    """True if shN exists and has data."""

@property
def sh_degree(self) -> int:
    """Get SH degree (0, 1, 2, or 3) based on shN shape."""
```

### Linear Operations (Apply to All Bands)

```python
def adjust_brightness(self, factor: float, inplace: bool = True) -> Self:
    """Scale all SH coefficients by factor."""

def adjust_contrast(self, factor: float, inplace: bool = True) -> Self:
    """Adjust contrast. Offset only affects DC."""

def adjust_saturation(self, factor: float, inplace: bool = True) -> Self:
    """Apply saturation matrix to all bands."""

def adjust_temperature(self, temp: float, inplace: bool = True) -> Self:
    """Apply temperature matrix to all bands."""

def adjust_tint(self, tint: float, inplace: bool = True) -> Self:
    """Apply tint matrix to all bands."""

def adjust_hue_shift(self, degrees: float, inplace: bool = True) -> Self:
    """Apply hue rotation matrix to all bands."""

def adjust_fade(self, amount: float, inplace: bool = True) -> Self:
    """Add offset to DC only."""
```

### Nonlinear Operations (Approximated)

```python
def adjust_gamma(self, gamma: float, inplace: bool = True) -> Self:
    """Apply gamma. For SH data, scales shN proportionally."""

def adjust_vibrance(self, amount: float, inplace: bool = True) -> Self:
    """Adaptive saturation. Approximated for SH data."""

def adjust_shadows(self, amount: float, inplace: bool = True) -> Self:
    """Adjust shadows. Per-Gaussian scaling based on luminance."""

def adjust_highlights(self, amount: float, inplace: bool = True) -> Self:
    """Adjust highlights. Per-Gaussian scaling based on luminance."""
```

### Combined

```python
def color(self, values: ColorValues, inplace: bool = True) -> Self:
    """Apply all color adjustments from ColorValues object."""
```

---

## Testing Requirements

### Unit Tests

#### 1. Format Preservation

```python
def test_brightness_preserves_sh_format():
    t = create_sh_tensor()  # sh0 in SH format
    assert t.is_sh0_sh
    t.adjust_brightness(1.2, inplace=True)
    assert t.is_sh0_sh  # Should still be SH format

def test_saturation_preserves_sh_format():
    t = create_sh_tensor()
    assert t.is_sh0_sh
    t.adjust_saturation(1.3, inplace=True)
    assert t.is_sh0_sh  # Should NOT convert to RGB
```

#### 2. All Bands Modified

```python
def test_saturation_modifies_all_bands():
    t = create_sh_tensor()
    sh0_before = t.sh0.clone()
    shN_before = t.shN.clone()

    t.adjust_saturation(1.3, inplace=True)

    assert not torch.allclose(t.sh0, sh0_before)
    assert not torch.allclose(t.shN, shN_before)

def test_brightness_modifies_all_bands():
    t = create_sh_tensor()
    shN_before = t.shN.clone()

    t.adjust_brightness(1.5, inplace=True)

    assert torch.allclose(t.shN, shN_before * 1.5)
```

#### 3. Offset Only Affects DC

```python
def test_contrast_offset_only_affects_dc():
    t = create_sh_tensor()
    # Set shN to zeros
    t.shN = torch.zeros_like(t.shN)

    t.adjust_contrast(0.5, inplace=True)

    # shN should still be zeros (offset doesn't affect it)
    assert torch.allclose(t.shN, torch.zeros_like(t.shN))

def test_fade_only_affects_dc():
    t = create_sh_tensor()
    shN_before = t.shN.clone()

    t.adjust_fade(0.1, inplace=True)

    # shN should be unchanged
    assert torch.allclose(t.shN, shN_before)
```

#### 4. No Clamping for SH Format

```python
def test_no_clamping_for_sh_format():
    t = create_sh_tensor()
    t.sh0 = torch.randn(10, 3) * 5  # Values outside [0,1]

    t.adjust_saturation(1.3, inplace=True)

    # Should not be clamped
    assert t.sh0.min() < 0 or t.sh0.max() > 1
```

#### 5. Clamping for RGB Format

```python
def test_clamping_for_rgb_format():
    t = create_sh_tensor()
    t.to_rgb(inplace=True)
    t.sh0 = torch.rand(10, 3) * 0.5 + 0.25  # Valid RGB

    t.adjust_brightness(3.0, inplace=True)  # Would exceed 1

    assert t.sh0.max() <= 1.0
    assert t.sh0.min() >= 0.0
```

### Integration Tests

#### Roundtrip: Edit → Save → Load

```python
def test_edit_save_load_roundtrip():
    # Load original
    t1 = load_ply("test_scene.ply")
    sh0_orig = t1.sh0.clone()
    shN_orig = t1.shN.clone()

    # Apply edits
    t1.adjust_saturation(1.3, inplace=True)
    t1.adjust_brightness(0.9, inplace=True)

    # Save
    save_ply(t1, "test_edited.ply")

    # Reload
    t2 = load_ply("test_edited.ply")

    # Verify edits persisted
    assert not torch.allclose(t2.sh0, sh0_orig)
    assert not torch.allclose(t2.shN, shN_orig)
    assert torch.allclose(t1.sh0, t2.sh0, atol=1e-5)
    assert torch.allclose(t1.shN, t2.shN, atol=1e-5)
```

#### Render Comparison

```python
def test_render_before_after():
    t = load_ply("test_scene.ply")

    # Render before
    img_before = render(t, camera)

    # Apply saturation boost
    t.adjust_saturation(1.5, inplace=True)

    # Render after
    img_after = render(t, camera)

    # Images should differ (edit had effect)
    assert not torch.allclose(img_before, img_after)

    # Average saturation should increase
    sat_before = compute_image_saturation(img_before)
    sat_after = compute_image_saturation(img_after)
    assert sat_after > sat_before
```

---

## Reference Implementation: SuperSplat

SuperSplat (PlayCanvas) implements this correctly in `splat-serialize.ts`:

```typescript
const applyTransform = (c, s, offset) => {
    // Scale
    c.r = offset + c.r * s.r;
    c.g = offset + c.g * s.g;
    c.b = offset + c.b * s.b;

    // Saturation (linear matrix)
    const grey = c.r * 0.299 + c.g * 0.587 + c.b * 0.114;
    c.r = grey + (c.r - grey) * saturation;
    c.g = grey + (c.g - grey) * saturation;
    c.b = grey + (c.b - grey) * saturation;
};

// Apply to DC with offset
applyTransform(dc_color, scale, offset);

// Apply to higher-order SH with offset=0
for (let d = 0; d < shCoeffs; ++d) {
    applyTransform(sh_color, scale, 0);  // No offset!
}
```

Key observations:

1. Same linear transform applied to all bands
2. Offset only applied to DC
3. No conversion to RGB
4. No clamping of SH values

---

## Summary: Golden Rules

1. **Never convert to RGB** for color operations if you have higher-order SH
2. **Never clamp** SH format values - only clamp RGB format
3. **Apply linear matrices to ALL bands** (DC and higher-order)
4. **Additive offsets only affect DC** - shN represents variation, not absolute color
5. **Approximate nonlinear operations** by scaling shN proportionally to DC luminance change
6. **Track format explicitly** - always know if sh0 is in SH or RGB space
7. **Preserve shN** - never discard or zero out higher-order coefficients

Following these rules ensures:

- View-dependent effects are preserved
- Edited data can be saved and reloaded correctly
- Rendering produces expected results
- Mathematical consistency is maintained

"""
Adjust Engine — Pure image processing library for the Adjust tab.

No GUI dependencies. All functions operate on numpy arrays (float32 [0,1] BGR
or uint8 [0,255] BGR as noted). The engine is usable from CLI, tests, or GUI.

Pipeline order (from plan):
  1. RAW Development (Phase C)
  2. Noise Reduction
  3. White Balance
  4. Exposure
  5. Shadows / Highlights
  6. Contrast / Levels / Curves
  7. Colour (saturation, vibrance, HSL, channel mixer, split toning)
  8. Colour correction (chart / reference matching)
  9. Sharpening
  10. Corrections (vignette, CA)
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# State dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AdjustmentState:
    """All tone/colour/detail slider values. Defaults = no change."""

    # Tone
    exposure: float = 0.0        # EV stops, -10 to +10
    contrast: float = 0.0        # -100 to +100
    highlights: float = 0.0      # -100 to +100
    shadows: float = 0.0         # -100 to +100
    whites: float = 0.0          # -100 to +100
    blacks: float = 0.0          # -100 to +100
    gamma: float = 1.0           # 0.1 to 5.0 (1.0 = no change)

    # Colour
    temperature: float = 0.0     # Kelvin offset from neutral (0 = no change)
    tint: float = 0.0            # -100 to +100 (green/magenta)
    saturation: float = 0.0      # -100 to +100
    vibrance: float = 0.0        # -100 to +100

    # Detail
    sharpen_amount: float = 0.0  # 0 to 100
    sharpen_radius: float = 1.0  # 0.5 to 5.0
    sharpen_threshold: float = 0.0  # 0 to 100
    denoise_strength: float = 0.0   # 0 to 100
    denoise_method: str = "bilateral"  # bilateral, nlmeans, wavelet, ai_nafnet, ai_scunet

    # Corrections
    vignette_strength: float = 0.0  # 0 to 2.0 (0 = none)
    clahe_clip: float = 0.0         # 0 = disabled, >0 = clip limit


@dataclass
class RawDevelopState:
    """RAW development parameters (Phase C — rawpy integration)."""

    demosaic_algorithm: str = "AHD"
    output_color: str = "sRGB"
    output_bps: int = 16
    use_camera_wb: bool = True
    use_auto_wb: bool = False
    exp_shift: float = 1.0       # 0.25 to 8.0
    highlight_mode: int = 0
    no_auto_bright: bool = True
    gamma: tuple = (2.222, 4.5)  # BT.709 default


@dataclass
class AdjustDocument:
    """Complete state for undo/presets. Nests adjustments + RAW params."""

    adjustments: AdjustmentState = field(default_factory=AdjustmentState)
    raw: RawDevelopState = field(default_factory=RawDevelopState)


# ══════════════════════════════════════════════════════════════════════════════
# Undo stack
# ══════════════════════════════════════════════════════════════════════════════

class UndoStack:
    """Stores full AdjustDocument snapshots — both adjust sliders and RAW params are undoable."""

    def __init__(self, max_depth: int = 50):
        self._stack: list[AdjustDocument] = []
        self._redo: list[AdjustDocument] = []
        self._max = max_depth

    def push(self, state: AdjustDocument):
        self._stack.append(copy.deepcopy(state))
        if len(self._stack) > self._max:
            self._stack.pop(0)
        self._redo.clear()

    def undo(self) -> Optional[AdjustDocument]:
        if len(self._stack) < 2:
            return None
        self._redo.append(self._stack.pop())
        return copy.deepcopy(self._stack[-1])

    def redo(self) -> Optional[AdjustDocument]:
        if not self._redo:
            return None
        state = self._redo.pop()
        self._stack.append(state)
        return copy.deepcopy(state)

    @property
    def can_undo(self) -> bool:
        return len(self._stack) >= 2

    @property
    def can_redo(self) -> bool:
        return len(self._redo) > 0


# ══════════════════════════════════════════════════════════════════════════════
# RAW development
# ══════════════════════════════════════════════════════════════════════════════

# Gated import — rawpy is optional
try:
    import rawpy as _rawpy
    HAS_RAWPY = True
except ImportError:
    _rawpy = None
    HAS_RAWPY = False

# Map friendly names → rawpy enum values (resolved at call time to avoid
# import-time failure when rawpy is missing)
_DEMOSAIC_MAP = {
    "AHD": 3, "VNG": 1, "PPG": 2, "DCB": 4, "LMMSE": 9,
    "AMAZE": 10, "DHT": 11, "AAHD": 12,
}

_COLORSPACE_MAP = {
    "sRGB": 1, "Adobe": 2, "Wide": 3, "ProPhoto": 4,
    "XYZ": 5, "ACES": 6, "P3D65": 7, "Rec2020": 8,
}


def load_raw(path: str, state: RawDevelopState) -> np.ndarray:
    """Develop a RAW file into a float32 BGR [0,1] image.

    Args:
        path: filesystem path to .nef/.raf/.dng/.cr2/.cr3/.arw file
        state: RAW development parameters

    Returns:
        float32 BGR image [0, 1]

    Raises:
        ImportError: if rawpy is not installed
        RuntimeError: if the file cannot be decoded
    """
    if not HAS_RAWPY:
        raise ImportError(
            "rawpy is not installed. Install with: pip install rawpy"
        )

    demosaic_val = _DEMOSAIC_MAP.get(state.demosaic_algorithm, 3)
    colorspace_val = _COLORSPACE_MAP.get(state.output_color, 1)

    with _rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            demosaic_algorithm=_rawpy.DemosaicAlgorithm(demosaic_val),
            output_color=_rawpy.ColorSpace(colorspace_val),
            output_bps=state.output_bps,
            use_camera_wb=state.use_camera_wb,
            use_auto_wb=state.use_auto_wb,
            exp_shift=state.exp_shift,
            highlight_mode=state.highlight_mode,
            no_auto_bright=state.no_auto_bright,
            gamma=state.gamma,
        )

    # Convert to float32 [0, 1]
    max_val = 65535.0 if state.output_bps == 16 else 255.0
    img_float = rgb.astype(np.float32) / max_val

    # rawpy outputs RGB; convert to BGR for OpenCV pipeline consistency
    img_bgr = img_float[:, :, ::-1].copy()
    return img_bgr


# ══════════════════════════════════════════════════════════════════════════════
# Denoise — bilateral, NL-means, AI (gated)
# ══════════════════════════════════════════════════════════════════════════════

# Gated imports for AI denoise
try:
    from nafnetlib import DenoiseProcessor as _NafnetProcessor
    HAS_NAFNET = True
except ImportError:
    _NafnetProcessor = None
    HAS_NAFNET = False

try:
    import spandrel as _spandrel
    HAS_SPANDREL = True
except ImportError:
    _spandrel = None
    HAS_SPANDREL = False

# Available denoise methods — used by UI to build dropdown
DENOISE_METHODS = ["bilateral", "nlmeans"]
if HAS_NAFNET:
    DENOISE_METHODS.append("ai_nafnet")
if HAS_SPANDREL:
    DENOISE_METHODS.append("ai_scunet")


def denoise_bilateral(img: np.ndarray, strength: float) -> np.ndarray:
    """Edge-preserving bilateral filter.

    Args:
        img: float32 BGR [0, 1]
        strength: 0 to 100
    """
    if strength <= 0:
        return img

    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    s = strength / 100.0
    d = max(3, int(s * 9))
    sigma_color = 20 + s * 55
    sigma_space = 20 + s * 55
    filtered = cv2.bilateralFilter(img_u8, d, sigma_color, sigma_space)
    return filtered.astype(np.float32) / 255.0


def denoise_nlmeans(img: np.ndarray, strength: float) -> np.ndarray:
    """Non-local means denoising.

    Args:
        img: float32 BGR [0, 1]
        strength: 0 to 100
    """
    if strength <= 0:
        return img

    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    s = strength / 100.0
    h = int(3 + s * 7)  # 3–10
    denoised = cv2.fastNlMeansDenoisingColored(img_u8, None, h, h, 7, 21)
    return denoised.astype(np.float32) / 255.0


# AI denoise model cache (load once, reuse)
_nafnet_model = None
_scunet_model = None


def denoise_nafnet(img: np.ndarray, _strength: float) -> np.ndarray:
    """NAFNet neural denoising via nafnetlib.

    Args:
        img: float32 BGR [0, 1]
        _strength: unused (NAFNet has no strength param, applied or not)
    """
    if not HAS_NAFNET:
        raise ImportError("nafnetlib not installed. pip install nafnetlib")

    from PIL import Image as PILImage

    global _nafnet_model
    if _nafnet_model is None:
        _nafnet_model = _NafnetProcessor("sidd_width64", device="cuda")

    # Convert float32 BGR → PIL RGB
    rgb_u8 = (np.clip(img[:, :, ::-1], 0, 1) * 255).astype(np.uint8)
    pil = PILImage.fromarray(rgb_u8)

    result_pil = _nafnet_model.process(pil)

    result_rgb = np.array(result_pil, dtype=np.float32) / 255.0
    return result_rgb[:, :, ::-1].copy()  # RGB → BGR


def denoise_image(img: np.ndarray, strength: float, method: str) -> np.ndarray:
    """Route to appropriate denoise method.

    Args:
        img: float32 BGR [0, 1]
        strength: 0 to 100
        method: one of DENOISE_METHODS
    """
    if strength <= 0:
        return img

    if method == "bilateral":
        return denoise_bilateral(img, strength)
    elif method == "nlmeans":
        return denoise_nlmeans(img, strength)
    elif method == "ai_nafnet":
        return denoise_nafnet(img, strength)
    # ai_scunet would use spandrel — placeholder for when model .pth is available
    return img


# ══════════════════════════════════════════════════════════════════════════════
# Colour correction — chart-based and histogram matching
# ══════════════════════════════════════════════════════════════════════════════

# Gated imports — colour-science and cv2.mcc are optional
try:
    import colour as _colour
    HAS_COLOUR_SCIENCE = True
except ImportError:
    _colour = None
    HAS_COLOUR_SCIENCE = False

HAS_MCC = hasattr(cv2, 'mcc')


def detect_chart_mcc(img_bgr_u8: np.ndarray) -> Optional[np.ndarray]:
    """Detect ColorChecker Classic 24 using cv2.mcc and return measured patches.

    Args:
        img_bgr_u8: uint8 BGR image

    Returns:
        (24, 3) float64 sRGB array of measured patch colours, or None if
        detection failed or cv2.mcc is unavailable.
    """
    if not HAS_MCC:
        return None

    detector = cv2.mcc.CCheckerDetector.create()
    if not detector.process(img_bgr_u8, cv2.mcc.MCC24):
        return None

    checkers = detector.getListColorChecker()
    if not checkers:
        return None

    checker = checkers[0]
    # getChartsRGB returns (24*3,) or (24, 4) depending on version
    charts_rgb = checker.getChartsRGB()
    if charts_rgb is None:
        return None

    # Reshape to (24, 3) — channels may be in columns 1,2,3 with count in 0
    charts = np.array(charts_rgb, dtype=np.float64)
    if charts.ndim == 1:
        charts = charts.reshape(-1, 3)
    elif charts.shape[1] == 4:
        # Format: [count, R, G, B] per patch
        charts = charts[:, 1:4]

    # Normalize to [0, 1] sRGB
    patches = charts / 255.0
    if len(patches) < 24:
        return None

    return patches[:24]


def get_reference_patches() -> Optional[np.ndarray]:
    """Get ColorChecker Classic 24 reference sRGB values from colour-science.

    Returns:
        (24, 3) float64 sRGB array, or None if colour-science unavailable.
    """
    if not HAS_COLOUR_SCIENCE:
        return None

    cc = _colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
    ref_srgb = []
    for _name, xyY in cc.data.items():
        XYZ = _colour.xyY_to_XYZ(xyY)
        sRGB = _colour.XYZ_to_sRGB(XYZ)
        ref_srgb.append(sRGB)
    return np.array(ref_srgb, dtype=np.float64)


def fit_chart_correction(measured: np.ndarray, reference: np.ndarray,
                         method: str = 'Finlayson 2015',
                         degree: int = 2) -> np.ndarray:
    """Compute a colour correction matrix from chart patches.

    Args:
        measured: (24, 3) measured sRGB patch values [0, 1]
        reference: (24, 3) reference sRGB patch values [0, 1]
        method: 'Finlayson 2015', 'Cheung 2004', or 'Vandermonde'
        degree: polynomial degree (1–4, higher = better fit, risk of overfitting)

    Returns:
        Correction matrix (shape depends on method/degree).

    Raises:
        ImportError: if colour-science is not installed
    """
    if not HAS_COLOUR_SCIENCE:
        raise ImportError(
            "colour-science is not installed. Install with: pip install colour-science"
        )
    return _colour.characterisation.matrix_colour_correction(
        measured, reference, method=method, degree=degree,
    )


def apply_chart_correction(img: np.ndarray, ccm: np.ndarray,
                           method: str = 'Finlayson 2015',
                           degree: int = 2) -> np.ndarray:
    """Apply a pre-computed colour correction matrix to an image.

    Args:
        img: float32 BGR [0, 1]
        ccm: correction matrix from fit_chart_correction
        method: must match the method used to fit
        degree: must match the degree used to fit

    Returns:
        Corrected float32 BGR [0, 1]
    """
    if not HAS_COLOUR_SCIENCE:
        raise ImportError("colour-science is not installed")

    # colour-science expects RGB
    rgb = img[:, :, ::-1].astype(np.float64)
    corrected = _colour.characterisation.apply_matrix_colour_correction(
        rgb, ccm, method=method, degree=degree,
    )
    bgr = np.clip(corrected, 0, 1).astype(np.float32)[:, :, ::-1]
    return bgr


def match_histograms(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match the histogram of source to reference, per-channel in LAB.

    Args:
        source: float32 BGR [0, 1]
        reference: float32 BGR [0, 1]

    Returns:
        Histogram-matched float32 BGR [0, 1]
    """
    # Convert to uint8 LAB for histogram operations
    src_u8 = (np.clip(source, 0, 1) * 255).astype(np.uint8)
    ref_u8 = (np.clip(reference, 0, 1) * 255).astype(np.uint8)

    src_lab = cv2.cvtColor(src_u8, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(ref_u8, cv2.COLOR_BGR2LAB)

    result = np.zeros_like(src_lab)
    for c in range(3):
        src_hist, _ = np.histogram(src_lab[:, :, c].flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(ref_lab[:, :, c].flatten(), 256, [0, 256])
        src_cdf = src_hist.cumsum().astype(np.float64)
        ref_cdf = ref_hist.cumsum().astype(np.float64)
        src_cdf /= src_cdf[-1] + 1e-10
        ref_cdf /= ref_cdf[-1] + 1e-10
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            lut[i] = min(np.searchsorted(ref_cdf, src_cdf[i]), 255)
        result[:, :, c] = cv2.LUT(src_lab[:, :, c], lut)

    matched_bgr = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return matched_bgr.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# Adjustment functions
# ══════════════════════════════════════════════════════════════════════════════

def adjust_exposure(img: np.ndarray, ev: float) -> np.ndarray:
    """Apply exposure compensation in EV stops.

    Args:
        img: float32 BGR image [0, 1]
        ev: exposure value in stops (-10 to +10).
            +1 = 2x brighter, -1 = 0.5x brightness.

    Returns:
        Adjusted image, clipped to [0, 1].
    """
    if ev == 0.0:
        return img
    multiplier = np.float32(2.0 ** ev)
    return np.clip(img * multiplier, 0.0, 1.0)


def adjust_shadows_highlights(img: np.ndarray, shadows: float, highlights: float) -> np.ndarray:
    """Targeted shadow/highlight recovery in LAB.

    Uses a smooth quadratic mask so the adjustment fades naturally.
    At slider=100, dark/bright regions shift by ~25 L units (out of 100).

    Args:
        img: float32 BGR [0, 1]
        shadows: -100 to +100 (positive lifts shadows)
        highlights: -100 to +100 (negative recovers highlights)
    """
    if shadows == 0.0 and highlights == 0.0:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    L_norm = L / 100.0

    adjustment = np.zeros_like(L_norm)
    if shadows != 0:
        # Quadratic mask: peaks at L=0, fades to zero at L=0.5+
        shadow_mask = np.clip(1.0 - L_norm * 2.0, 0.0, 1.0) ** 2
        adjustment += (shadows / 100.0) * shadow_mask * 25.0
    if highlights != 0:
        # Quadratic mask: peaks at L=1, fades to zero at L=0.5-
        highlight_mask = np.clip(L_norm * 2.0 - 1.0, 0.0, 1.0) ** 2
        adjustment += (highlights / 100.0) * highlight_mask * 25.0

    lab[:, :, 0] = np.clip(L + adjustment, 0, 100)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def adjust_whites_blacks(img: np.ndarray, whites: float, blacks: float) -> np.ndarray:
    """Adjust the extreme tonal endpoints in LAB.

    Whites targets the top ~20% of luminance; Blacks targets the bottom ~20%.
    Uses cubic masks for a sharper falloff than shadows/highlights.

    Args:
        img: float32 BGR [0, 1]
        whites: -100 to +100
        blacks: -100 to +100
    """
    if whites == 0.0 and blacks == 0.0:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    L_norm = L / 100.0

    adjustment = np.zeros_like(L_norm)
    if whites != 0:
        mask = np.clip(L_norm * 1.25 - 0.25, 0.0, 1.0) ** 3
        adjustment += (whites / 100.0) * mask * 15.0
    if blacks != 0:
        mask = np.clip(1.0 - L_norm * 1.25, 0.0, 1.0) ** 3
        adjustment += (blacks / 100.0) * mask * 15.0

    lab[:, :, 0] = np.clip(L + adjustment, 0, 100)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def adjust_contrast(img: np.ndarray, amount: float) -> np.ndarray:
    """Contrast adjustment on L-channel using a sigmoid S-curve.

    At amount=+100, the midtone separation roughly doubles.
    At amount=-100, the image is nearly flat grey.

    Args:
        img: float32 BGR [0, 1]
        amount: -100 to +100
    """
    if amount == 0.0:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0] / 100.0  # normalize to 0–1
    # Sigmoid-based S-curve: strength controls the steepness
    # At amount=50 the effect is moderate; at 100 it's strong but not clipping
    k = amount * 0.04  # range: -4 to +4
    # Centered sigmoid: shifts midpoint contrast
    L_centered = L - 0.5
    L = 0.5 + L_centered * (1.0 + k * (0.25 - L_centered ** 2) * 4.0)
    lab[:, :, 0] = np.clip(L * 100.0, 0, 100)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def adjust_white_balance(img: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    """White balance via simplified blackbody RGB multipliers.

    Args:
        img: float32 BGR [0, 1]
        temperature: offset from neutral. Positive = warmer (more red),
                     negative = cooler (more blue). Range approx -100 to +100.
        tint: green/magenta shift, -100 to +100.
    """
    if temperature == 0.0 and tint == 0.0:
        return img

    # Temperature → R/B multiplier (simplified model)
    # At temperature=100: R *= 1.15, B *= 0.85 (~15% shift, visible but not blown)
    r_mult = 1.0 + temperature * 0.0015
    b_mult = 1.0 - temperature * 0.0015
    g_mult = 1.0

    # Tint → G multiplier
    # At tint=100: G *= 1.08 (~8% shift)
    if tint != 0.0:
        g_mult += tint * 0.0008

    # Normalize so min multiplier is 1.0 (prevent darkening)
    min_m = min(r_mult, g_mult, b_mult)
    if min_m > 0:
        r_mult /= min_m
        g_mult /= min_m
        b_mult /= min_m

    result = img.copy()
    result[:, :, 2] *= np.float32(r_mult)  # BGR: channel 2 = R
    result[:, :, 1] *= np.float32(g_mult)  # channel 1 = G
    result[:, :, 0] *= np.float32(b_mult)  # channel 0 = B
    return np.clip(result, 0.0, 1.0)


def adjust_saturation(img: np.ndarray, amount: float) -> np.ndarray:
    """Global saturation adjustment in HSV.

    Args:
        img: float32 BGR [0, 1]
        amount: -100 to +100 (0 = no change, -100 = grayscale, +100 = 2x)
    """
    if amount == 0.0:
        return img

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    factor = 1.0 + amount / 100.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_vibrance(img: np.ndarray, amount: float) -> np.ndarray:
    """Selective saturation — boost less-saturated pixels more.

    Args:
        img: float32 BGR [0, 1]
        amount: -100 to +100
    """
    if amount == 0.0:
        return img

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    # Boost inversely proportional to current saturation
    boost = (1.0 - s) * (amount / 100.0)
    hsv[:, :, 1] = np.clip(s + boost, 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_sharpen(img: np.ndarray, amount: float, radius: float = 1.0,
                   threshold: float = 0.0) -> np.ndarray:
    """LAB unsharp mask with adaptive edge protection.

    Args:
        img: float32 BGR [0, 1]
        amount: 0 to 100
        radius: 0.5 to 5.0 (Gaussian sigma)
        threshold: 0 to 100 (noise gate)
    """
    if amount <= 0:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0] / 100.0

    amount_scaled = amount / 50.0
    threshold_scaled = threshold / 100.0
    sigma = radius
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    L_blurred = cv2.GaussianBlur(L, (kernel_size, kernel_size), sigma,
                                 borderType=cv2.BORDER_REFLECT)
    unsharp = L - L_blurred

    # Threshold gate
    if threshold_scaled > 0:
        edge_strength = np.abs(unsharp)
        edge_mask = np.where(edge_strength > threshold_scaled, 1.0,
                             edge_strength / (threshold_scaled + 1e-8))
        unsharp = unsharp * edge_mask

    # Adaptive: reduce sharpening in extremes
    highlight_mask = np.where(L > 0.9, 1.0 - (L - 0.9) * 10.0, 1.0)
    shadow_mask = np.where(L < 0.1, L * 10.0, 1.0)
    adaptive = np.clip(highlight_mask * shadow_mask, 0.1, 1.0)

    L_sharp = np.clip(L + unsharp * amount_scaled * adaptive, 0, 1) * 100.0
    lab[:, :, 0] = L_sharp
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def adjust_clahe(img: np.ndarray, clip_limit: float) -> np.ndarray:
    """CLAHE on L-channel.

    Args:
        img: float32 BGR [0, 1]
        clip_limit: 0 = disabled, typical 1.0–4.0
    """
    if clip_limit <= 0:
        return img

    # CLAHE needs uint8
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result.astype(np.float32) / 255.0


def adjust_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """Remove vignette by applying radial gain.

    Args:
        img: float32 BGR [0, 1]
        strength: 0 = none, 1.0 = default, 2.0 = aggressive
    """
    if strength <= 0:
        return img

    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max_r
    gain = (1.0 + strength * r ** 2).astype(np.float32)
    return np.clip(img * gain[:, :, np.newaxis], 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Video utilities
# ══════════════════════════════════════════════════════════════════════════════

import json as _json
import shutil
import subprocess

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m4v", ".webm"}


def is_video_file(path: str) -> bool:
    """Check if path has a video extension."""
    return Path(path).suffix.lower() in _VIDEO_EXTENSIONS


def probe_video(path: str) -> Optional[dict]:
    """Get video metadata via ffprobe.

    Returns dict with keys: width, height, fps, duration, frame_count, codec.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = _json.loads(result.stdout)
    except Exception:
        return None

    # Find video stream
    video_stream = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            video_stream = s
            break
    if not video_stream:
        return None

    # Parse fps from r_frame_rate (e.g. "30000/1001")
    rfr = video_stream.get("r_frame_rate", "30/1")
    try:
        num, den = rfr.split("/")
        fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        fps = 30.0

    duration = float(data.get("format", {}).get("duration", 0))
    frame_count = int(video_stream.get("nb_frames", 0))
    if frame_count == 0 and duration > 0:
        frame_count = int(duration * fps)

    return {
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": fps,
        "duration": duration,
        "frame_count": frame_count,
        "codec": video_stream.get("codec_name", ""),
    }


def extract_single_frame(video_path: str, timestamp: float) -> Optional[np.ndarray]:
    """Extract a single frame at the given timestamp via ffmpeg.

    Args:
        video_path: path to video file
        timestamp: time in seconds

    Returns:
        float32 BGR [0,1] image, or None on failure
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    cmd = [
        ffmpeg, "-y",
        "-ss", f"{timestamp:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode != 0:
            return None
    except Exception:
        return None

    raw = result.stdout
    if not raw:
        return None

    # Need to know frame dimensions — probe first or use stderr
    # Parse dimensions from stderr
    import re
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    match = re.search(r"(\d{2,5})x(\d{2,5})", stderr_text)
    if not match:
        return None

    w, h = int(match.group(1)), int(match.group(2))
    expected = w * h * 3
    if len(raw) < expected:
        return None

    frame = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(h, w, 3)
    return frame.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# GPU preview pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _has_torch_cuda() -> bool:
    global _TORCH_CUDA
    try:
        return _TORCH_CUDA
    except NameError:
        pass
    try:
        import torch
        _TORCH_CUDA = torch.cuda.is_available()
    except ImportError:
        _TORCH_CUDA = False
    return _TORCH_CUDA


def apply_adjustments_gpu(img: np.ndarray, state: AdjustmentState) -> np.ndarray:
    """Apply tone/colour adjustments on CUDA. ~10ms for 2MP on RTX 3090 Ti.

    Handles: exposure, white balance, shadows, highlights, whites, blacks,
    contrast, saturation, vibrance. Sharpening, denoise, CLAHE, and vignette
    are not included (they need spatial kernels better suited to CPU/OpenCV).

    Args:
        img: float32 BGR [0, 1]
        state: slider values

    Returns:
        float32 BGR [0, 1]
    """
    import torch

    device = torch.device("cuda")
    t = torch.from_numpy(img).to(device)

    # 1. White balance
    if state.temperature != 0.0 or state.tint != 0.0:
        r_m = 1.0 + state.temperature * 0.0015
        b_m = 1.0 - state.temperature * 0.0015
        g_m = 1.0 + state.tint * 0.0008
        mn = min(r_m, g_m, b_m)
        if mn > 0:
            r_m /= mn; g_m /= mn; b_m /= mn
        t = t.clone()
        t[:, :, 2] *= r_m
        t[:, :, 1] *= g_m
        t[:, :, 0] *= b_m
        t = torch.clamp(t, 0, 1)

    # 2. Exposure
    if state.exposure != 0.0:
        t = torch.clamp(t * (2.0 ** state.exposure), 0, 1)

    # 3. Shadows / Highlights / Whites / Blacks (luminance-preserving)
    has_tonal = (state.shadows != 0 or state.highlights != 0
                 or state.whites != 0 or state.blacks != 0)
    if has_tonal:
        lum = 0.2126 * t[:, :, 2] + 0.7152 * t[:, :, 1] + 0.0722 * t[:, :, 0]
        adj = torch.zeros_like(lum)
        if state.shadows != 0:
            adj += (state.shadows / 100.0) * torch.clamp(1.0 - lum * 2.0, 0, 1).pow(2) * 0.25
        if state.highlights != 0:
            adj += (state.highlights / 100.0) * torch.clamp(lum * 2.0 - 1.0, 0, 1).pow(2) * 0.25
        if state.whites != 0:
            adj += (state.whites / 100.0) * torch.clamp(lum * 1.25 - 0.25, 0, 1).pow(3) * 0.15
        if state.blacks != 0:
            adj += (state.blacks / 100.0) * torch.clamp(1.0 - lum * 1.25, 0, 1).pow(3) * 0.15
        new_lum = torch.clamp(lum + adj, 1e-6, 1.0)
        scale = new_lum / lum.clamp(min=1e-6)
        t = torch.clamp(t * scale.unsqueeze(-1), 0, 1)

    # 4. Contrast (sigmoid S-curve on luminance)
    if state.contrast != 0.0:
        lum = 0.2126 * t[:, :, 2] + 0.7152 * t[:, :, 1] + 0.0722 * t[:, :, 0]
        k = state.contrast * 0.04
        Lc = lum - 0.5
        new_lum = torch.clamp(0.5 + Lc * (1.0 + k * (0.25 - Lc ** 2) * 4.0), 1e-6, 1.0)
        scale = new_lum / lum.clamp(min=1e-6)
        t = torch.clamp(t * scale.unsqueeze(-1), 0, 1)

    # 5. Saturation
    if state.saturation != 0.0:
        lum = 0.2126 * t[:, :, 2] + 0.7152 * t[:, :, 1] + 0.0722 * t[:, :, 0]
        factor = 1.0 + state.saturation / 100.0
        t = torch.clamp(lum.unsqueeze(-1) + (t - lum.unsqueeze(-1)) * factor, 0, 1)

    # 6. Vibrance (boost low-saturation pixels more)
    if state.vibrance != 0.0:
        lum = 0.2126 * t[:, :, 2] + 0.7152 * t[:, :, 1] + 0.0722 * t[:, :, 0]
        diff = t - lum.unsqueeze(-1)
        sat_level = (diff ** 2).sum(dim=-1, keepdim=True).sqrt() + 1e-8
        boost = (1.0 - sat_level / (sat_level.max() + 1e-8)) * (state.vibrance / 100.0)
        t = torch.clamp(lum.unsqueeze(-1) + diff * (1.0 + boost), 0, 1)

    return t.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# CPU pipeline compositor
# ══════════════════════════════════════════════════════════════════════════════

def apply_adjustments(img: np.ndarray, state: AdjustmentState) -> np.ndarray:
    """Apply the full adjustment pipeline in correct order (CPU path).

    Pipeline order:
      1. RAW Development (handled by load_raw before this function)
      2. Noise Reduction
      3. White Balance
      4. Exposure
      5. Shadows / Highlights / Whites / Blacks
      6. Contrast
      7. Colour (saturation, vibrance)
      8. Colour correction (Phase D)
      9. Sharpening
      10. Corrections (vignette, CLAHE)

    Args:
        img: float32 BGR image [0, 1]
        state: current slider values

    Returns:
        Adjusted image (float32 BGR [0, 1]).
    """
    result = img

    # 2. Noise Reduction (before amplifying noise with other ops)
    if state.denoise_strength > 0:
        result = denoise_image(result, state.denoise_strength, state.denoise_method)

    # 3. White Balance
    if state.temperature != 0.0 or state.tint != 0.0:
        result = adjust_white_balance(result, state.temperature, state.tint)

    # 4. Exposure
    if state.exposure != 0.0:
        result = adjust_exposure(result, state.exposure)

    # 5. Shadows / Highlights
    if state.shadows != 0.0 or state.highlights != 0.0:
        result = adjust_shadows_highlights(result, state.shadows, state.highlights)

    # 5b. Whites / Blacks
    if state.whites != 0.0 or state.blacks != 0.0:
        result = adjust_whites_blacks(result, state.whites, state.blacks)

    # 6. Contrast
    if state.contrast != 0.0:
        result = adjust_contrast(result, state.contrast)

    # 7. Colour
    if state.saturation != 0.0:
        result = adjust_saturation(result, state.saturation)
    if state.vibrance != 0.0:
        result = adjust_vibrance(result, state.vibrance)

    # 9. Sharpening
    if state.sharpen_amount > 0:
        result = adjust_sharpen(result, state.sharpen_amount,
                                state.sharpen_radius, state.sharpen_threshold)

    # 10. Corrections
    if state.clahe_clip > 0:
        result = adjust_clahe(result, state.clahe_clip)
    if state.vignette_strength > 0:
        result = adjust_vignette(result, state.vignette_strength)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Batch operations
# ══════════════════════════════════════════════════════════════════════════════

def normalize_exposures(images: list[np.ndarray],
                        target_L: Optional[float] = None) -> list[np.ndarray]:
    """Normalize exposure across a batch by matching median luminance.

    Args:
        images: list of float32 BGR [0, 1] images
        target_L: target median luminance (0–255 in LAB L). If None, uses
                  the batch mean.

    Returns:
        List of exposure-normalized images.
    """
    if not images:
        return images

    # Compute median L for each image
    medians = []
    for img in images:
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
        medians.append(float(np.median(lab[:, :, 0])))

    if target_L is None:
        target_L = np.mean(medians)

    results = []
    for img, med in zip(images, medians):
        if med < 1:
            results.append(img)
            continue
        scale = target_L / med
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * scale, 0, 255)
        result_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        results.append(result_bgr.astype(np.float32) / 255.0)

    return results

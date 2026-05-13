"""
SAM 3 Gated Model Setup — Token, Access & Weight Management
=============================================================

Extracted from LichtFeld Studio's 360 plugin (setup_checks.py + sam3_compat.py).
Handles the full onboarding flow for SAM 3's gated HuggingFace model:

    1. Check if a HuggingFace token is saved locally
    2. Verify the token has access to facebook/sam3
    3. Classify all HF error types into user-friendly states
    4. Download model weights once access is confirmed
    5. Import SAM 3's image API with compatibility shims

The stage machine drives the UI:
    needs_token -> needs_access -> ready_to_install -> needs_weights -> ready

Usage:
    # Check current state
    report = check_sam3_setup()
    print(report.overall_stage)   # "needs_token", "needs_access", "ready", etc.
    print(report.message)         # User-facing message
    print(report.next_action)     # What the user should do next

    # User pastes a token
    report = verify_hf_token_detailed("hf_abc123...")
    if report.overall_stage == "ready_to_install":
        download_model_weights()

    # Import the image API (after install)
    build_model, Processor = import_sam3_image_api()

Dependencies:
    pip install huggingface_hub

    SAM 3 itself (facebook/sam3) is installed separately — the gating flow
    must succeed BEFORE you can install or download the model.
"""
from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

SAM3_MODEL_ID = "facebook/sam3"

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Import Compatibility Shims (from sam3_compat.py)
# ═══════════════════════════════════════════════════════════════════════════════
#
# SAM 3's package eagerly imports video/training modules at import time, even
# when callers only need the image API. On Windows (and in frozen .exe builds),
# some of those transitive dependencies are not importable even though the
# image-model path itself works fine.
#
# These stubs satisfy the eager imports without pulling in the video/training
# stack. Only needed if you use SAM 3's image API (build_sam3_image_model +
# Sam3Processor), NOT the video/multiplex API.

_IMAGE_IMPORT_STUBS_ACTIVE = False


def _register_stub(module_name: str, **attrs: Any) -> None:
    """Register a lightweight module stub if it is not already loaded."""
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module


def enable_sam3_image_import_compat() -> None:
    """Install import stubs needed for image-only SAM 3 usage.

    These placeholders satisfy eager imports from sam3.model_builder that are
    only required for video/training flows. They are intentionally minimal
    because the image path is all we need for masking.
    """
    global _IMAGE_IMPORT_STUBS_ACTIVE
    if _IMAGE_IMPORT_STUBS_ACTIVE:
        return

    _register_stub(
        "sam3.train.data.collator",
        BatchedDatapoint=object,
    )
    _register_stub(
        "sam3.model.sam3_video_inference",
        Sam3VideoInferenceWithInstanceInteractivity=type(
            "Sam3VideoInferenceWithInstanceInteractivity", (), {}
        ),
    )
    _register_stub(
        "sam3.model.sam3_video_predictor",
        Sam3VideoPredictorMultiGPU=type("Sam3VideoPredictorMultiGPU", (), {}),
    )
    _IMAGE_IMPORT_STUBS_ACTIVE = True


def import_sam3_image_api():
    """Import the SAM 3 image-model API with compatibility shims.

    Returns:
        (build_sam3_image_model, Sam3Processor) tuple

    Raises:
        ImportError: if sam3 package is not installed
    """
    enable_sam3_image_import_compat()
    from sam3 import build_sam3_image_model  # type: ignore[import-untyped]
    from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[import-untyped]
    return build_sam3_image_model, Sam3Processor


def get_sam3_bpe_path() -> str:
    """Return a usable BPE vocabulary path for the SAM 3 tokenizer.

    ADAPT THIS for your project layout. The sam3 package expects a BPE vocab
    file at sam3/assets/bpe_simple_vocab_16e6.txt.gz. If you're using an
    editable install (pip install -e .), it resolves via package resources.
    If you're using a PyPI wheel or frozen .exe, you may need to vendor the
    file and adjust the path below.
    """
    from pathlib import Path

    BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"
    candidates: list[Path] = []

    try:
        import sam3  # type: ignore[import-untyped]
        package_dir = Path(sam3.__file__).resolve().parent
        candidates.append(package_dir / "assets" / BPE_FILENAME)
        candidates.append(package_dir.parent / "assets" / BPE_FILENAME)
    except ImportError:
        pass

    # --- ADAPT: add your project-specific fallback paths here ---
    # Example for a frozen exe:
    #   candidates.append(Path(sys._MEIPASS) / "assets" / BPE_FILENAME)
    # Example for a repo-relative path:
    #   candidates.append(Path(__file__).resolve().parent / "assets" / BPE_FILENAME)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"SAM 3 tokenizer vocabulary not found. Expected one of: "
        f"{', '.join(str(path) for path in candidates)}"
    )


def sam3_image_api_available() -> bool:
    """Return True if the image-model API is importable right now."""
    try:
        import_sam3_image_api()
        get_sam3_bpe_path()
        return True
    except (ImportError, FileNotFoundError):
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: State Detection & Reporting
# ═══════════════════════════════════════════════════════════════════════════════

_hf_access_cache: bool | None = None


@dataclass
class Sam3SetupReport:
    """User-facing SAM 3 setup status for onboarding UI.

    The overall_stage field drives the UI state machine:
        needs_token    -> show token input field
        needs_access   -> show "request access on HuggingFace" instructions
        ready_to_install -> show "Install SAM 3" button
        needs_weights  -> show "Download weights" button
        ready          -> SAM 3 is fully operational
        error          -> show error + retry button
    """
    token_status: str = "missing"       # missing | saved | verified | invalid | network_error
    access_status: str = "unknown"      # unknown | pending | granted | network_error
    runtime_status: str = "missing"     # missing | installed | broken
    weights_status: str = "missing"     # missing | present | failed
    overall_stage: str = "needs_token"  # needs_token | needs_access | ready_to_install | needs_weights | ready | error
    message: str = ""                   # User-facing summary
    next_action: str = ""               # What the user should do next
    detail: str = ""                    # Technical detail for logs/tooltips


def _check_hf_token() -> bool:
    """Check if a HuggingFace token is saved on this machine."""
    try:
        from huggingface_hub import get_token
        token = get_token()
        return token is not None and len(token) > 0
    except Exception:
        return False


def _check_hf_access() -> bool:
    """Check if saved token has access to SAM 3 model.

    Result is cached in-memory after first successful check to avoid
    network calls on every UI refresh.
    """
    global _hf_access_cache
    if _hf_access_cache is not None:
        return _hf_access_cache
    try:
        from huggingface_hub import get_token, model_info
        token = get_token()
        if not token:
            return False
        info = model_info(SAM3_MODEL_ID, token=token)
        result = info is not None
        if result:
            _hf_access_cache = True
        return result
    except Exception:
        return False


def _check_sam3_installed() -> bool:
    """Check if SAM 3 image API is importable."""
    return sam3_image_api_available()


def _check_weights_downloaded() -> bool:
    """Check if SAM 3 weights exist in HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(SAM3_MODEL_ID, "config.json")
        return result is not None and isinstance(result, str)
    except Exception:
        return False


def _classify_hf_access_exception(
    exc: Exception,
    *,
    token_verified: bool,
) -> tuple[str, str, str]:
    """Map HuggingFace verification failures to user-facing setup states.

    Returns:
        (token_status, access_status, detail_message) tuple
    """
    text = f"{type(exc).__name__}: {exc}".lower()

    # Token itself is bad
    if any(term in text for term in (
        "401", "unauthorized", "invalid token", "invalid user token",
        "authentication", "whoami",
    )):
        return "invalid", "unknown", "HuggingFace rejected this token."

    # Token is valid but model access not yet granted
    if any(term in text for term in (
        "403", "forbidden", "gated", "awaiting review", "access to model",
        "restricted", "not in the authorized list", "not have access",
    )):
        return (
            ("verified" if token_verified else "saved"),
            "pending",
            "SAM 3 model access is still pending HuggingFace approval.",
        )

    # Network issues
    if any(term in text for term in (
        "timed out", "connection", "network", "dns", "ssl",
        "temporarily unavailable", "name or service not known",
        "connection aborted", "connection reset",
    )):
        return (
            ("verified" if token_verified else "saved"),
            "network_error",
            "Could not reach HuggingFace to verify SAM 3 access.",
        )

    # Unknown error
    return (
        ("verified" if token_verified else "saved"),
        "unknown",
        "Could not verify SAM 3 access.",
    )


def _build_sam3_setup_report(
    *,
    has_token: bool = False,
    has_access: bool = False,
    has_sam3: bool = False,
    has_weights: bool = False,
    token_status: str | None = None,
    access_status: str | None = None,
    runtime_status: str | None = None,
    weights_status: str | None = None,
    detail: str = "",
) -> Sam3SetupReport:
    """Build a user-facing report from individual state booleans.

    This is the core stage machine. The order of checks matters:
    error > needs_token > needs_access > ready_to_install > needs_weights > ready
    """
    token_status = token_status or ("saved" if has_token else "missing")
    access_status = access_status or ("granted" if has_access else "unknown")
    runtime_status = runtime_status or ("installed" if has_sam3 else "missing")
    weights_status = weights_status or ("present" if has_weights else "missing")

    # Upgrade token_status if access is already granted
    if access_status == "granted" and token_status == "saved":
        token_status = "verified"

    # --- Error state ---
    if runtime_status == "broken":
        return Sam3SetupReport(
            token_status=token_status,
            access_status=access_status,
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="error",
            message="SAM 3 install failed.",
            next_action="Retry the install, then click Re-check Setup.",
            detail=detail,
        )

    # --- Token states ---
    if token_status == "missing":
        return Sam3SetupReport(
            token_status=token_status,
            access_status="unknown",
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="needs_token",
            message="SAM 3 requires a HuggingFace token.",
            next_action="Paste a HuggingFace token, then click Verify Access.",
            detail=detail,
        )

    if token_status == "invalid":
        return Sam3SetupReport(
            token_status=token_status,
            access_status="unknown",
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="needs_token",
            message="HuggingFace rejected this token.",
            next_action="Create a new token at huggingface.co/settings/tokens and try again.",
            detail=detail,
        )

    # --- Access states ---
    if access_status in ("network_error",) or token_status == "network_error":
        return Sam3SetupReport(
            token_status=token_status,
            access_status=access_status,
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="needs_access",
            message="Could not verify SAM 3 access right now.",
            next_action="Check your internet connection, then try again.",
            detail=detail,
        )

    if access_status == "pending":
        return Sam3SetupReport(
            token_status=token_status,
            access_status=access_status,
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="needs_access",
            message="SAM 3 model access is still pending approval.",
            next_action=(
                "Visit huggingface.co/facebook/sam3 to request access, "
                "then wait for approval and click Re-check."
            ),
            detail=detail,
        )

    if access_status != "granted":
        return Sam3SetupReport(
            token_status=token_status,
            access_status=access_status,
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="needs_access",
            message="SAM 3 access has not been verified yet.",
            next_action="Click Check Setup to verify your HuggingFace token.",
            detail=detail,
        )

    # --- Install state ---
    if runtime_status != "installed":
        return Sam3SetupReport(
            token_status=token_status,
            access_status=access_status,
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="ready_to_install",
            message="SAM 3 access approved — ready to install.",
            next_action="Click Install SAM 3 to download the runtime.",
            detail=detail,
        )

    # --- Weights state ---
    if weights_status != "present":
        return Sam3SetupReport(
            token_status=token_status,
            access_status=access_status,
            runtime_status=runtime_status,
            weights_status=weights_status,
            overall_stage="needs_weights",
            message="SAM 3 runtime installed but model weights are missing.",
            next_action="Click Download Weights.",
            detail=detail,
        )

    # --- Ready ---
    return Sam3SetupReport(
        token_status=token_status,
        access_status=access_status,
        runtime_status=runtime_status,
        weights_status=weights_status,
        overall_stage="ready",
        message="SAM 3 is ready to use.",
        next_action="",
        detail=detail,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Actions (verify, download, forget)
# ═══════════════════════════════════════════════════════════════════════════════


def check_sam3_setup() -> Sam3SetupReport:
    """Run all probes and return a complete setup report.

    This is the main entry point. Call it on app launch, after setup
    actions, or when the user clicks "Re-check Setup".
    """
    has_token = _check_hf_token()
    has_access = _check_hf_access() if has_token else False
    has_sam3 = _check_sam3_installed()
    has_weights = _check_weights_downloaded()

    return _build_sam3_setup_report(
        has_token=has_token,
        has_access=has_access,
        has_sam3=has_sam3,
        has_weights=has_weights,
    )


def verify_hf_token_detailed(token: str) -> Sam3SetupReport:
    """Verify a HuggingFace token and return a detailed setup report.

    This saves the token locally via huggingface_hub.login() if it's valid,
    then checks whether it has access to the gated SAM 3 model.

    Args:
        token: Raw HuggingFace token string (e.g. "hf_abc123...")

    Returns:
        Sam3SetupReport with current state after verification
    """
    global _hf_access_cache

    token = token.strip()
    if not token:
        return _build_sam3_setup_report(token_status="missing")

    try:
        from huggingface_hub import login, model_info

        # Save the token locally (persists across sessions)
        login(token=token)

        # Check if this token has access to the gated model
        info = model_info(SAM3_MODEL_ID, token=token)
        if info is not None:
            _hf_access_cache = True
            return _build_sam3_setup_report(
                has_token=True,
                has_access=True,
                has_sam3=_check_sam3_installed(),
                has_weights=_check_weights_downloaded(),
                token_status="verified",
                access_status="granted",
            )
    except Exception as exc:
        logger.warning("HF token verification failed: %s", exc)
        token_status, access_status, detail = _classify_hf_access_exception(
            exc, token_verified=True,
        )
        return _build_sam3_setup_report(
            has_token=token_status not in {"missing", "invalid"},
            has_access=access_status == "granted",
            has_sam3=_check_sam3_installed(),
            has_weights=_check_weights_downloaded(),
            token_status=token_status,
            access_status=access_status,
            detail=detail,
        )

    # model_info returned None (unexpected)
    return _build_sam3_setup_report(
        has_token=True,
        has_access=False,
        has_sam3=_check_sam3_installed(),
        has_weights=_check_weights_downloaded(),
        token_status="verified",
        access_status="unknown",
    )


def forget_hf_token() -> bool:
    """Remove the saved HuggingFace login from this machine."""
    global _hf_access_cache
    try:
        from huggingface_hub import logout
        logout()
        _hf_access_cache = None
        return True
    except Exception as exc:
        logger.warning("Could not forget saved HuggingFace token: %s", exc)
        return False


def download_model_weights(on_progress=None) -> bool:
    """Download SAM 3 model weights from HuggingFace.

    Args:
        on_progress: Optional callback(message: str) for status updates

    Returns:
        True on success, False on failure
    """
    try:
        from huggingface_hub import snapshot_download

        if on_progress:
            on_progress("Downloading SAM 3 weights (~3.5 GB)...")

        snapshot_download(SAM3_MODEL_ID)

        if on_progress:
            on_progress("SAM 3 weights downloaded.")
        return True
    except Exception as exc:
        logger.error("SAM 3 weight download failed: %s", exc)
        if on_progress:
            on_progress(f"Download failed: {exc}")
        return False

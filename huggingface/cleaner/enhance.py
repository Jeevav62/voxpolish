import torch
from df.enhance import enhance as _enhance
from df.enhance import init_df
from df.io import load_audio, save_audio

_model = None
_df_state = None
_device = None


def load_model(post_filter: bool = True):
    global _model, _df_state, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Enhance] Device: {_device}" + (f" — {torch.cuda.get_device_name(0)}" if _device.type == "cuda" else ""))
    # log_level="ERROR" silences df's per-file resampling INFO/WARNING spam
    _model, _df_state, _ = init_df(post_filter=post_filter, log_level="ERROR")
    _model = _model.to(_device)
    _model.eval()
    print("[Enhance] Warming up...")
    with torch.no_grad():
        _dummy = torch.zeros(1, _df_state.sr())
        _enhance(_model, _df_state, _dummy)
    if _device.type == "cuda":
        torch.cuda.synchronize()
    print("[Enhance] Ready.")


def enhance_audio(audio: torch.Tensor, atten_lim_db: float = None) -> torch.Tensor:
    """Enhance a CPU audio tensor. Returns enhanced CPU tensor."""
    with torch.no_grad():
        return _enhance(_model, _df_state, audio, atten_lim_db=atten_lim_db)


def sample_rate() -> int:
    return _df_state.sr()

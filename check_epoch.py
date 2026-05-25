"""Print epoch/best_f1 for one or more resume checkpoints on the bionne-v2 volume.
Usage: python check_epoch.py mdeberta_ru_v2.pt [mdeberta_bilingual_v2.pt ...]
"""
import sys
import modal

app = modal.App.lookup("bionne-r-pipeline", create_if_missing=False)
volume = modal.Volume.from_name("bionne-v2")

with modal.enable_output():
    with app.run():
        from BioNNE_R.modal_app import inspect_checkpoints  # noqa: not used at import
        # Use the function from the deployed app
        pass

# Simpler: just spin up a fresh container using the same volume
check_image = modal.Image.debian_slim(python_version="3.10").pip_install("torch")
tmp_app = modal.App("check-epoch-tmp")

@tmp_app.function(image=check_image, volumes={"/vol": volume}, timeout=120, cpu=2)
def _check(names):
    import torch
    from pathlib import Path
    results = {}
    volume.reload()
    for name in names:
        p = Path("/vol/checkpoints") / (name + ".resume")
        if p.exists():
            s = torch.load(str(p), map_location="cpu", weights_only=False)
            results[name] = {"epoch": s.get("epoch", 0),
                             "best_f1": s.get("best_macro_f1", s.get("best_f1", 0.0))}
        else:
            results[name] = {"epoch": 0, "best_f1": 0.0}
    return results

with tmp_app.run():
    names = sys.argv[1:] or ["mdeberta_ru_v2.pt", "mdeberta_bilingual_v2.pt"]
    res = _check.remote(names)
    for name, info in res.items():
        print(f"{name}: epoch={info['epoch']}/10  best_f1={info['best_f1']:.4f}")

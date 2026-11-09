"""Where the network's step time goes: forward ms per top-level module (synchronised hooks) and the
top CUDA kernels of forward + backward (``torch.profiler``). Random input; the network as the recipe
builds it (``_create_model``)."""
import time

import torch

from benchmarks.perf.common import applied_config, load_config, synthetic_class_table


def profile_network(config=None, data="synthetic", classes=2, batch=16, seed=0, overrides=(), device="auto",
                    steps=5, top=25) -> dict:
    from torch.profiler import ProfilerActivity, profile

    cfg = load_config(config, overrides)
    if data == "synthetic":
        synthetic_class_table(cfg, classes)
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) or device == "cuda" else "cpu")
    cfg.train.use_gpus = dev.type == "cuda"
    with applied_config(cfg) as mc:
        from tomocpt.training.helpers import _create_model

        torch.manual_seed(seed)
        model, _, _ = _create_model(mc.train)
        net = model.model.to(dev).train()
        opt = torch.optim.RAdam(net.parameters(), lr=1e-4)
        x = torch.randn(int(batch), 1, 64, 64, 64, device=dev)
        bf16 = str(getattr(mc.train.network.TORCH_FLOAT_PRECISION, "value", "")) == "bf16-mixed" and dev.type == "cuda"

        def step():
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
                out = net(x)[0]
            loss = torch.sigmoid(out.float()).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        def sync():
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

        for _ in range(3):
            step()
        names = {m: n for n, m in net.named_children()}
        fwd = {}

        def pre(m, i):
            sync()
            m._g8_t = time.perf_counter()

        def post(m, i, o):
            sync()
            fwd[names[m]] = fwd.get(names[m], 0.0) + (time.perf_counter() - m._g8_t) * 1e3

        hooks = [h for m in names for h in (m.register_forward_pre_hook(pre), m.register_forward_hook(post))]
        for _ in range(steps):
            step()
        for h in hooks:
            h.remove()
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
        with profile(activities=acts) as prof:
            for _ in range(3):
                step()
            sync()
        key = "self_device_time_total" if dev.type == "cuda" else "self_cpu_time_total"
        events = sorted(prof.key_averages(), key=lambda e: -getattr(e, key, 0.0))
        total = sum(getattr(e, key, 0.0) for e in events) or 1.0
        kernels = [{"name": e.key[:120], "ms_per_step": getattr(e, key, 0.0) / 1e3 / 3,
                    "share": getattr(e, key, 0.0) / total, "calls_per_step": e.count / 3}
                   for e in events[:top]]
    return {"forward_ms_per_module": {k: v / steps for k, v in fwd.items()}, "top_kernels": kernels,
            "batch": int(batch), "device": str(dev)}

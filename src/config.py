"""
Hardware detection + model sizing + device placement.

Designed so the SAME code works on:
 - a CPU-only dev box (tiny models, for smoke-testing)
 - a single GPU box (small/medium model, everything on one device)
 - a multi-GPU box, e.g. 4x T4 on Clore.ai (models spread across GPUs)

No internet access is required at runtime -- everything here is local
introspection (torch.cuda.*).
"""
import os
import torch


class HW:
    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.n_gpu = torch.cuda.device_count() if self.cuda else 0
        self.gpu_mem_gb = []
        for i in range(self.n_gpu):
            props = torch.cuda.get_device_properties(i)
            self.gpu_mem_gb.append(round(props.total_memory / (1024**3), 1))
        self.cpu_count = os.cpu_count() or 2

    @property
    def tier(self):
        if self.n_gpu == 0:
            return "tiny"          # CPU only -> smoke-test sized models
        min_mem = min(self.gpu_mem_gb) if self.gpu_mem_gb else 0
        if self.n_gpu == 1 and min_mem <= 20:
            return "small"          # e.g. single T4 (16GB)
        if self.n_gpu >= 2:
            return "medium"         # multi-GPU, e.g. 4x T4 -> spread agents out
        return "small"

    def devices(self):
        """
        Returns a dict mapping each of the three agents to a torch.device.
        - 0 GPUs  : all on cpu
        - 1 GPU   : all three share the single GPU (sized down to fit)
        - >=2 GPUs: predictor, analyser, trader each get a dedicated GPU,
                    extra GPUs (if any) are used for parallel env rollouts.
        """
        if self.n_gpu == 0:
            d = torch.device("cpu")
            return {"predictor": d, "analyser": d, "trader": d, "rollout": [d]}
        if self.n_gpu == 1:
            d = torch.device("cuda:0")
            return {"predictor": d, "analyser": d, "trader": d, "rollout": [d]}
        # n_gpu >= 2
        gpus = [torch.device(f"cuda:{i}") for i in range(self.n_gpu)]
        mapping = {
            "predictor": gpus[0 % len(gpus)],
            "analyser": gpus[1 % len(gpus)],
            "trader": gpus[2 % len(gpus)],
        }
        # remaining GPUs (e.g. the 4th on a 4x T4 box) used to run extra
        # parallel day-rollouts to speed up data collection.
        extra = gpus[3:] if len(gpus) > 3 else [gpus[-1]]
        mapping["rollout"] = extra
        return mapping

    def summary(self):
        lines = [
            f"CPU cores        : {self.cpu_count}",
            f"CUDA available   : {self.cuda}",
            f"GPU count        : {self.n_gpu}",
        ]
        for i, m in enumerate(self.gpu_mem_gb):
            lines.append(f"  GPU[{i}] memory  : {m} GB")
        lines.append(f"Resource tier    : {self.tier}")
        return "\n".join(lines)


# Model size presets per tier: (hidden_dim, num_layers, embed_dim)
MODEL_SIZES = {
    "tiny":   dict(hidden=32,  layers=1, embed=8,  ppo_hidden=64,  batch_days_parallel=1),
    "small":  dict(hidden=128, layers=2, embed=16, ppo_hidden=256, batch_days_parallel=2),
    "medium": dict(hidden=256, layers=3, embed=32, ppo_hidden=512, batch_days_parallel=4),
    "large":  dict(hidden=512, layers=4, embed=64, ppo_hidden=1024, batch_days_parallel=8),
}


def get_hw_and_sizes():
    hw = HW()
    sizes = MODEL_SIZES[hw.tier]
    return hw, sizes


if __name__ == "__main__":
    hw, sizes = get_hw_and_sizes()
    print(hw.summary())
    print("Model sizes:", sizes)
    print("Device map:", hw.devices())

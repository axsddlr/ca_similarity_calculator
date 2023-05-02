import sys
import torch
import hashlib
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class CrossAttnCalculator:
    def __init__(self, model):
        self.model = model

    def cal_cross_attn(self, to_q, to_k, to_v, rand_input):
        hidden_dim, embed_dim = to_q.shape
        attn_to_q = nn.Linear(hidden_dim, embed_dim, bias=False)
        attn_to_k = nn.Linear(hidden_dim, embed_dim, bias=False)
        attn_to_v = nn.Linear(hidden_dim, embed_dim, bias=False)
        attn_to_q.load_state_dict({"weight": to_q})
        attn_to_k.load_state_dict({"weight": to_k})
        attn_to_v.load_state_dict({"weight": to_v})

        return torch.einsum(
            "ik, jk -> ik",
            F.softmax(
                torch.einsum(
                    "ij, kj -> ik", attn_to_q(rand_input), attn_to_k(rand_input)
                ),
                dim=-1,
            ),
            attn_to_v(rand_input),
        )

    def eval(self, n, input):
        qk = f"model.diffusion_model.output_blocks.{n}.1.transformer_blocks.0.attn1.to_q.weight"
        uk = f"model.diffusion_model.output_blocks.{n}.1.transformer_blocks.0.attn1.to_k.weight"
        vk = f"model.diffusion_model.output_blocks.{n}.1.transformer_blocks.0.attn1.to_v.weight"
        atoq, atok, atov = self.model[qk], self.model[uk], self.model[vk]

        attn = self.cal_cross_attn(atoq, atok, atov, input)
        return attn


class ModelEvaluator:
    @staticmethod
    def model_hash(filename):
        try:
            with open(filename, "rb") as file:
                m = hashlib.sha256()
                file.seek(0x100000)
                m.update(file.read(0x10000))
                return m.hexdigest()[0:8]
        except FileNotFoundError:
            return "NOFILE"

    @staticmethod
    def convert_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        return new_state_dict

    @staticmethod
    def load_model(path):
        if path.suffix == ".safetensors":
            model = load_file(path, device="cpu")
            return model.state_dict() if hasattr(model, "state_dict") else model
        else:
            ckpt = torch.load(path, map_location="cpu")
            return ckpt["state_dict"] if "state_dict" in ckpt else ckpt


def main():
    if len(sys.argv) < 3:
        print("Usage: script.py base_model other_model [other_models...]")
        sys.exit(1)

    file1 = Path(sys.argv[1])
    files = sys.argv[2:]

    seed = 114514
    torch.manual_seed(seed)
    print(f"seed: {seed}")

    print(f"Loading base model: {file1}")
    model_a = ModelEvaluator.load_model(file1)
    print("Base model loaded")
    calc_a = CrossAttnCalculator(model_a)

    print()
    print(f"base: {file1.name} [{ModelEvaluator.model_hash(file1)}]")
    print()

    map_attn_a = {}
    map_rand_input = {}
    for n in range(3, 11):
        hidden_dim, embed_dim = model_a[
            f"model.diffusion_model.output_blocks.{n}.1.transformer_blocks.0.attn1.to_q.weight"
        ].shape
        rand_input = torch.randn([embed_dim, hidden_dim])

        map_attn_a[n] = calc_a.eval(n, rand_input)
        map_rand_input[n] = rand_input

    del model_a

    for file2 in files:
        file2 = Path(file2)
        model_b = ModelEvaluator.load_model(file2)
        calc_b = CrossAttnCalculator(model_b)
        sims = []
        for n in range(3, 11):
            attn_a = map_attn_a[n]
            attn_b = calc_b.eval(n, map_rand_input[n])

            sim = torch.mean(torch.cosine_similarity(attn_a, attn_b))
            sims.append(sim)

        print(
            f"{file2} [{ModelEvaluator.model_hash(file2)}] - {torch.mean(torch.stack(sims)) * 1e2:.2f}%"
        )
        print()


if __name__ == "__main__":
    main()

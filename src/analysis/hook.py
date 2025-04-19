from types import MethodType

import timm
import torch
import torch.nn as nn
import torch.nn.quantized as nnq


def hook_softmax_gelu(model: nn.Module) -> tuple[list[torch.Tensor]]:
    softmax_outputs = []
    gelu_outputs = []

    def custom_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(dim=0)  # equivalent to q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        softmax_outputs.append(attn.detach().cpu())
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    model.eval().cpu()
    for module in model.modules():
        if isinstance(module, timm.models.vision_transformer.Attention):
            module.forward = MethodType(custom_forward, module)
        if isinstance(module, torch.nn.GELU):
            module.register_forward_hook(
                lambda m, i, o: gelu_outputs.append(o.detach().cpu())
            )

    return softmax_outputs, gelu_outputs


def get_activations(
    model: nn.Module, input: torch.Tensor, layers=None
) -> tuple[dict[str, torch.Tensor]]:
    if layers is None:
        layers = (
            nn.Conv2d,
            nn.Linear,
            nn.LayerNorm,
            nn.Softmax,
            nnq.Conv2d,
            nnq.Linear,
        )

    inputs = {}
    outputs = {}

    def _get_act(name):
        def hook(module, input, output):
            inputs[name] = input[0].detach()
            outputs[name] = output.detach()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, layers):
            module.register_forward_hook(_get_act(name))
    model(input)
    return inputs, outputs


def get_weights(model, layers=None, quantized_layers=None) -> dict[str, torch.Tensor]:
    if layers is None:
        layers = (nn.Conv2d, nn.Linear)
    if quantized_layers is None:
        quantized_layers = (nnq.Conv2d, nnq.Linear)

    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, layers):
            weights[name] = module.weight.data
        elif isinstance(module, quantized_layers):
            weights[name] = module.weight()
    return weights

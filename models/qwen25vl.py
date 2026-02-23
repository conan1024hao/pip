"""
Qwen2.5-VL PIP wrapper: dumps LLM attention from first generated token
to image placeholder tokens.
"""
import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Optional transformers import
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
    AutoProcessor = None

class Qwen25VL_PIP(nn.Module):
    """Qwen2.5-VL wrapper that provides LLM first-token attention to image tokens."""

    def __init__(self, huggingface_root=None, model_name="Qwen/Qwen2.5-VL-3B-Instruct", attack_position=None, attn_layer_idx=-1):
        super(Qwen25VL_PIP, self).__init__()
        if Qwen2_5_VLForConditionalGeneration is None or AutoProcessor is None:
            raise ImportError("transformers is required for Qwen2.5-VL. Install with: pip install transformers")

        model_path = model_name
        if huggingface_root:
            if os.path.isabs(model_name):
                model_path = model_name
            else:
                local_candidate = os.path.join(huggingface_root, model_name)
                if os.path.exists(local_candidate):
                    model_path = local_candidate

        # Use eager attention so we can capture weights from the vision encoder
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda",
            attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.device = torch.device("cuda")
        self.eval().requires_grad_(False)

        self.model_name = model_name
        self.attack_position = attack_position
        self.attn_layer_idx = int(attn_layer_idx)

    def _prepare_model_inputs(self, x, question):
        """Convert tensor image + question to full model inputs."""
        x = torch.clamp(x, min=0, max=1)
        # To PIL: (C,H,W) in [0,1] -> uint8
        img_np = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Question: {question} Short answer:"},
                ],
            }
        ]
        # Build prompt first, then call processor(text=..., images=...) so text/image tokens stay aligned.
        prompt = self.processor.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        try:
            inputs = self.processor(
                text=[prompt],
                images=[pil_image],
                padding=True,
                return_tensors="pt",
            )
        except TypeError:
            inputs = self.processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt",
            )

        model_inputs = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                if k == "pixel_values":
                    model_inputs[k] = v.to(self.device, dtype=torch.float16)
                else:
                    model_inputs[k] = v.to(self.device)
            else:
                model_inputs[k] = v
        return model_inputs

    def get_attention(self, x, question):
        """
        Return attention from first generated LLM token to image tokens.
        Shape: (1, num_heads, 32), selecting one configurable decoder layer.
        """
        inputs = self._prepare_model_inputs(x, question)
        if "input_ids" not in inputs:
            raise RuntimeError("Processor did not return input_ids")

        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(self.model.model.config, "image_token_id", None)
        if image_token_id is None:
            raise RuntimeError("Could not find image_token_id in model config")

        image_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
        if image_positions.numel() == 0:
            raise RuntimeError("No image placeholder tokens found in input_ids")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_attentions=True,
        )
        if outputs.attentions is None or len(outputs.attentions) == 0:
            raise RuntimeError(
                "Could not capture LLM attentions. Ensure generate(..., output_attentions=True) is supported."
            )

        # attentions[step][layer]: (batch, num_heads, q_len, k_len)
        first_step_attns = outputs.attentions[0]
        if not isinstance(first_step_attns, (list, tuple)) or len(first_step_attns) == 0:
            raise RuntimeError("Unexpected attentions format from generation output")

        layer_attn_list = []
        for layer_attn in first_step_attns:
            if layer_attn.dim() != 4:
                continue
            # first generated token query -> all keys
            token_to_all = layer_attn[0, :, -1, :]  # (num_heads, k_len)
            token_to_image = token_to_all[:, image_positions]  # (num_heads, num_img_tokens)
            n_tok = min(32, token_to_image.size(1))
            token_to_image = token_to_image[:, :n_tok]
            if n_tok < 32:
                token_to_image = torch.nn.functional.pad(token_to_image, (0, 32 - n_tok), value=0.0)
            layer_attn_list.append(token_to_image)

        if len(layer_attn_list) == 0:
            raise RuntimeError("No valid layer attentions found for first generated token")

        layer_stack = torch.stack(layer_attn_list, dim=0)  # (num_layers, num_heads, 32)
        num_layers = layer_stack.size(0)
        layer_idx = self.attn_layer_idx
        if layer_idx < 0:
            layer_idx = num_layers + layer_idx
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"attn_layer_idx out of range: {self.attn_layer_idx} for num_layers={num_layers}")
        return layer_stack[layer_idx:layer_idx+1].float().cpu()

    def predict(self, x, question):
        inputs = self._prepare_model_inputs(x, question)
        inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        in_len = inputs["input_ids"].shape[1]
        out_ids = generated_ids[0][in_len:]
        return self.processor.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

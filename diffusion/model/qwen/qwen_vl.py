from typing import List, Optional, Tuple, Union

import torch
from PIL import Image

# =========================================================
# Optional Qwen-VL imports (guarded)
# =========================================================
try:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    QWEN_AVAILABLE = True
except Exception:
    process_vision_info = None
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None
    QWEN_AVAILABLE = False


# =========================================================
# Qwen-VL Embedder (REAL or DUMMY)
# =========================================================
class QwenVLEmbedder:
    """
    Qwen-VL embedder.

    - If QWEN_AVAILABLE == True:
        Uses real Qwen2.5-VL model.
    - If QWEN_AVAILABLE == False:
        Acts as a dummy stub to satisfy Sana builder imports
        (image-only inference path).
    """

    def __init__(self, *args, **kwargs):
        if not QWEN_AVAILABLE:
            # Dummy mode
            self.enabled = False
            self.device = "cpu"
            return

        # -------------------------------------------------
        # Real Qwen-VL initialization
        # -------------------------------------------------
        self.enabled = True

        model_id = kwargs.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        device = kwargs.get("device", None)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.torch_dtype
        ).to(self.device)

        self.processor.tokenizer.padding_side = "left"

        # -----------------------------
        # Prompt templates
        # -----------------------------
        self.tokenizer_max_length = 300

        self.prompt_template_encode = (
            "<|im_start|>system\n"
            "Describe the image by detailing the color, shape, size, texture, "
            "quantity, text, spatial relationships of the objects and background:"
            "<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.prompt_template_encode_start_idx = 34

        self.image_prompt_template_encode = (
            "<|im_start|>system\n"
            "Describe the key features of the input image (color, shape, size, "
            "texture, objects, background), then explain how the user's text "
            "instruction should alter or modify the image."
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>{}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.image_prompt_template_encode_start_idx = 64

    # =====================================================
    # Internal helper
    # =====================================================
    def _extract_masked_hidden(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[torch.Tensor]:
        split_hidden_states = []
        for i in range(hidden_states.shape[0]):
            mask_indices = attention_mask[i].nonzero(as_tuple=False).squeeze()
            extracted_states = hidden_states[i, mask_indices, :]
            split_hidden_states.append(extracted_states)
        return split_hidden_states

    # =====================================================
    # Public API — TEXT
    # =====================================================
    def get_prompt_embeds(
        self, prompt: Union[str, List[str]], max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if not self.enabled:
            raise RuntimeError(
                "Qwen-VL is disabled. Text embedding path should not be used "
                "for image-only Sana inference."
            )

        dtype = self.text_encoder.dtype
        prompts = [prompt] if isinstance(prompt, str) else prompt
        txt_with_template = [self.prompt_template_encode.format(p) for p in prompts]

        txt_tokens = self.processor.tokenizer(
            txt_with_template,
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        unpadded = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        prompt_only = [e[self.prompt_template_encode_start_idx :] for e in unpadded]

        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device)
            for e in prompt_only
        ]

        max_seq_len = max_length or max(e.size(0) for e in prompt_only)

        prompt_embeds = torch.stack(
            [
                torch.cat(
                    [u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]
                )
                for u in prompt_only
            ]
        )
        encoder_attention_mask = torch.stack(
            [
                torch.cat(
                    [u, u.new_zeros(max_seq_len - u.size(0))]
                )
                for u in attn_mask_list
            ]
        )

        return prompt_embeds.to(dtype=dtype), encoder_attention_mask

    # =====================================================
    # Public API — IMAGE + TEXT
    # =====================================================
    def get_image_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        image: Union[Image.Image, List[Image.Image]],
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if not self.enabled:
            raise RuntimeError(
                "Qwen-VL is disabled. Image-text embedding path should not be used."
            )

        prompts = [prompt] if isinstance(prompt, str) else prompt
        images = [image] if isinstance(image, Image.Image) else image

        txt = [self.image_prompt_template_encode.format(p) for p in prompts]

        model_inputs = self.processor(
            text=txt,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device, self.torch_dtype)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]
        split_hidden = self._extract_masked_hidden(
            hidden_states, model_inputs.attention_mask
        )
        split_hidden = [
            e[self.image_prompt_template_encode_start_idx :] for e in split_hidden
        ]

        max_seq_len = max_length or max(e.size(0) for e in split_hidden)

        prompt_embeds = torch.stack(
            [
                torch.nn.functional.pad(
                    u, (0, 0, 0, max_seq_len - u.size(0))
                )
                for u in split_hidden
            ]
        )

        encoder_attention_mask = torch.stack(
            [
                torch.nn.functional.pad(
                    torch.ones(u.size(0), device=u.device),
                    (0, max_seq_len - u.size(0)),
                )
                for u in split_hidden
            ]
        )

        return prompt_embeds, encoder_attention_mask

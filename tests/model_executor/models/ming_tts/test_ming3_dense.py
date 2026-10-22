# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming-3.0-tts-0.5B on the dense Ming TTS pipeline."""

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAEConfig
from vllm_omni.model_executor.models.ming_tts.constants import (
    AGGREGATOR_HIDDEN_SIZE,
    HISTORY_PATCH_SIZE,
    LATENT_DIM,
    LLM_HIDDEN_SIZE,
    LLM_VOCAB_SIZE,
    MING3_DENSE_PATCH_SIZE,
    SAMPLE_RATE,
    VAE_PATCH_SIZE,
)
from vllm_omni.model_executor.models.ming_tts.prompt_assembly import (
    build_prompt_token_ids,
    estimate_decode_steps_for_duration,
    resolve_effective_runtime_controls,
)
from vllm_omni.model_executor.models.ming_tts.validation import validate_ming_tts_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def _dense_cfg(*, patch_size: int, history_patch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        audio_dummy_token_id=151705,
        audio_eos_token_id=151704,
        text_eos_token_id=151669,
        audio_tokenizer_config=AudioVAEConfig(
            sample_rate=SAMPLE_RATE,
            patch_size=VAE_PATCH_SIZE,
            enc_kwargs={"latent_dim": LATENT_DIM, "input_dim": 882, "hop_size": 882},
            dec_kwargs={"latent_dim": LATENT_DIM, "output_dim": 882},
        ),
        latent_dim=LATENT_DIM,
        patch_size=patch_size,
        history_patch_size=history_patch_size,
        llm_hidden_size=LLM_HIDDEN_SIZE,
        llm_vocab_size=LLM_VOCAB_SIZE,
        sample_rate=SAMPLE_RATE,
        vae_patch_size=VAE_PATCH_SIZE,
        llm_config={"hidden_size": LLM_HIDDEN_SIZE},
        aggregator_config={"hidden_size": AGGREGATOR_HIDDEN_SIZE},
        ditar_config={"hidden_size": AGGREGATOR_HIDDEN_SIZE},
        latent_chunk_size=1,
        latent_left_context=0,
        max_decode_steps=1,
        stop_head_threshold=0.5,
        stop_head_min_steps=0,
    )


def test_validation_accepts_ming3_dense_patch_sizes():
    validate_ming_tts_config(_dense_cfg(patch_size=MING3_DENSE_PATCH_SIZE, history_patch_size=MING3_DENSE_PATCH_SIZE))


def test_validation_rejects_mixed_patch_sizes():
    with pytest.raises(ValueError, match="history_patch_size mismatch"):
        validate_ming_tts_config(_dense_cfg(patch_size=MING3_DENSE_PATCH_SIZE, history_patch_size=HISTORY_PATCH_SIZE))


class _Tokenizer:
    """Encodes each string as one id so the prompt layout is easy to read back."""

    unk_token_id = -1

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<audioPatch>": 1}

    def encode(self, text: str) -> list[int]:
        return [self.vocab.setdefault(text, len(self.vocab) + 1)]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def decode_all(self, ids: list[int]) -> list[str]:
        names = {v: k for k, v in self.vocab.items()}
        return [names[i] for i in ids]


@pytest.mark.parametrize(
    ("variant", "instruction_end"),
    [("dense", "<|endoftext|>"), ("ming3_dense", "<|im_end|>")],
)
def test_instruction_caption_terminator(variant: str, instruction_end: str):
    tokenizer = _Tokenizer()
    ids = build_prompt_token_ids(variant, tokenizer, prompt="P", text="hello", instruction="calm")
    pieces = tokenizer.decode_all(ids)
    caption = pieces.index("calm")
    assert pieces[caption + 1] == instruction_end
    assert pieces[caption + 2] == "<audio>"


@pytest.mark.parametrize(("patch_size", "steps"), [(4, 32), (MING3_DENSE_PATCH_SIZE, 125)])
def test_duration_hint_counts_steps_for_the_patch_size(patch_size: int, steps: int):
    assert estimate_decode_steps_for_duration(10.0, patch_size=patch_size) == steps
    controls = resolve_effective_runtime_controls(text="Duration: 10s. Hello.", patch_size=patch_size)
    assert max(controls.values()) == steps + 3

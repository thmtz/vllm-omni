# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Long input split into fragments, each run as its own engine request."""

import asyncio
import re
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture
from vllm import SamplingParams

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech, _max_parallel_fragments
from vllm_omni.entrypoints.openai.speech_usage import SpeechOutputTokenCounter
from vllm_omni.entrypoints.openai.tts_adapters.base import PreparedRequest
from vllm_omni.model_executor.models.ming_flash_omni.text_processing import get_semantic_length
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.mm_outputs import MultimodalPayload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SAMPLE_RATE = 24000
# 12 sentences of 10 words: about 120 semantic units, over Ming's 50 per fragment.
LONG_TEXT = " ".join(f"Sentence number {n} talks about the weather in the old town." for n in range(12))


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text)


def _Output(request_id: str, audio, tokens_out: int | None = None) -> OmniRequestOutput:
    """A real engine-shaped output: the AR stage attaches a MultimodalPayload to the completion."""
    payload = MultimodalPayload(
        tensors={} if audio is None or isinstance(audio, list) else {"audio": audio},
        metadata={"sr": torch.tensor(SAMPLE_RATE), **({"audio": audio} if isinstance(audio, list) else {})},
    )
    source = SimpleNamespace(outputs=[SimpleNamespace(index=0, multimodal_output=payload)], finished=True)
    metrics = (
        {"stage_metrics": {"0": {"num_tokens_out": tokens_out, "finish_reason": "stop"}}}
        if tokens_out is not None
        else {}
    )
    return OmniRequestOutput.from_stage_output(
        source, request_id=request_id, stage_id=0, final_output_type="audio", metrics=metrics
    )


@pytest.fixture
def ming_tts_server(mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.tts_adapters.base.load_supported_speakers",
        return_value=set(),
    )
    mocker.patch(
        "vllm_omni.entrypoints.openai.tts_adapters.base.load_codec_frame_rate",
        return_value=None,
    )
    engine_client = mocker.MagicMock()
    engine_client.errored = False
    engine_client.model_config = mocker.MagicMock(
        model="inclusionAI/Ming-omni-tts-0.5B",
        hf_config=SimpleNamespace(min_token_text_ratio=2, max_token_text_ratio=20),
        async_chunk=False,
    )
    engine_client.default_sampling_params_list = [SimpleNamespace(max_tokens=2048, min_tokens=None, extra_args=None)]
    engine_client.tts_batch_max_items = 32
    engine_client.stage_configs = [
        SimpleNamespace(
            engine_args=SimpleNamespace(
                model_stage="llm", model_arch="MingTTSForConditionalGeneration", worker_type="ar"
            ),
            tts_args={},
        )
    ]
    models = mocker.MagicMock()
    models.is_base_model.return_value = True
    server = OmniOpenAIServingSpeech(engine_client=engine_client, models=models, request_logger=mocker.MagicMock())

    async def build(request, sampling_params_list, has_inline_ref_audio):
        return PreparedRequest(prompt={"prompt": request.input}, tts_params={}, model_type="ming_tts")

    server._adapter.build = mocker.AsyncMock(side_effect=build)
    return server


def _fragment_samples(prompt: dict) -> int:
    # Audio length proportional to the fragment's word count, so joins are checkable.
    return 100 * len(_words(prompt["prompt"]))


def _final_only_generate(events: list):
    async def generate(*, prompt, request_id, **kwargs):
        events.append(("start", request_id))
        await asyncio.sleep(0)
        samples = _fragment_samples(prompt)
        yield _Output(request_id, torch.ones(1, samples), tokens_out=samples // 10)
        events.append(("end", request_id))

    return generate


class TestMingSplitLongInput:
    def test_short_input_is_not_split(self, ming_tts_server):
        request = OpenAICreateSpeechRequest(input="Hello there. How are you?", voice="lingguang")
        assert ming_tts_server._adapter.split_long_input(request) is None

    def test_long_input_splits_into_bounded_fragments_without_losing_words(self, ming_tts_server):
        request = OpenAICreateSpeechRequest(input=LONG_TEXT, voice="lingguang")
        fragments = ming_tts_server._adapter.split_long_input(request)

        assert fragments is not None and len(fragments) > 1
        assert all(get_semantic_length(fragment) <= 50 for fragment in fragments)
        assert _words(" ".join(fragments)) == _words(LONG_TEXT)

    @pytest.mark.parametrize("rate", [24000, 44100])
    def test_ming_offers_24khz_and_native_output(self, ming_tts_server, rate):
        request = OpenAICreateSpeechRequest(input="Hi.", voice="lingguang", sample_rate=rate)
        assert ming_tts_server._validate_speech_sample_rate(request) is None

    def test_style_instructions_and_podcast_are_not_split(self, ming_tts_server):
        styled = OpenAICreateSpeechRequest(input=LONG_TEXT, instructions="calm and slow")
        podcast = OpenAICreateSpeechRequest(input=LONG_TEXT, ref_audio=["a.wav", "b.wav"])
        assert ming_tts_server._adapter.split_long_input(styled) is None
        assert ming_tts_server._adapter.split_long_input(podcast) is None


class TestPrepareSpeechGenerationSplit:
    def test_without_opt_in_long_input_is_one_engine_request(self, ming_tts_server):
        ming_tts_server.engine_client.generate.return_value = "generator"
        request = OpenAICreateSpeechRequest(input=LONG_TEXT, voice="lingguang")

        asyncio.run(ming_tts_server._prepare_speech_generation(request, request_id="speech-1"))

        ming_tts_server.engine_client.generate.assert_called_once()
        assert ming_tts_server.engine_client.generate.call_args.kwargs["prompt"] == {"prompt": LONG_TEXT}

    def test_word_timestamps_rejected_when_input_is_split(self, ming_tts_server):
        request = OpenAICreateSpeechRequest(input=LONG_TEXT, voice="lingguang", word_timestamps=True)
        with pytest.raises(ValueError, match="word_timestamps"):
            asyncio.run(
                ming_tts_server._prepare_speech_generation(request, request_id="speech-1", split_long_input=True)
            )

    def test_non_streaming_joins_fragment_audio_and_sums_usage(self, ming_tts_server):
        events: list = []
        ming_tts_server.engine_client.generate.side_effect = _final_only_generate(events)
        request = OpenAICreateSpeechRequest(input=LONG_TEXT, voice="lingguang", response_format="pcm")
        fragments = ming_tts_server._adapter.split_long_input(request)

        usage: list = []
        audio_bytes, _ = asyncio.run(
            ming_tts_server._generate_audio_bytes(request, request_id="speech-1", usage_out=usage)
        )

        expected_samples = sum(100 * len(_words(fragment)) for fragment in fragments)
        assert len(audio_bytes) == 2 * expected_samples  # 16-bit mono PCM
        assert usage[0].output_tokens == expected_samples // 10
        request_ids = [request_id for kind, request_id in events if kind == "start"]
        assert request_ids == [f"speech-1-part{index}" for index in range(len(fragments))]
        # Non-streaming fragments run concurrently: all start before any ends.
        assert [kind for kind, _ in events[: len(fragments)]] == ["start"] * len(fragments)

    def test_non_streaming_fragment_cap_bounds_concurrency(self, ming_tts_server):
        ming_tts_server._max_parallel_fragments = 2
        running = 0
        peak = 0

        async def generate(*, prompt, request_id, **kwargs):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            samples = _fragment_samples(prompt)
            yield _Output(request_id, torch.ones(1, samples), tokens_out=samples // 10)
            running -= 1

        ming_tts_server.engine_client.generate.side_effect = generate
        request = OpenAICreateSpeechRequest(input=LONG_TEXT, voice="lingguang", response_format="pcm")
        fragments = ming_tts_server._adapter.split_long_input(request)
        assert len(fragments) > 2

        audio_bytes, _ = asyncio.run(ming_tts_server._generate_audio_bytes(request, request_id="speech-1"))

        assert peak == 2
        assert len(audio_bytes) == 2 * sum(_fragment_samples({"prompt": fragment}) for fragment in fragments)

    @pytest.mark.parametrize("value", ["-1", "two"])
    def test_fragment_cap_rejects_a_bad_value(self, monkeypatch, value):
        monkeypatch.setenv("VLLM_OMNI_TTS_MAX_PARALLEL_FRAGMENTS", value)
        with pytest.raises(ValueError, match="VLLM_OMNI_TTS_MAX_PARALLEL_FRAGMENTS"):
            _max_parallel_fragments()

    def test_empty_middle_fragment_fails_the_non_streaming_request(self, ming_tts_server, mocker):
        mocker.patch.object(ming_tts_server._adapter, "split_long_input", return_value=["One.", "Two.", "Three."])

        async def generate(*, prompt, request_id, **kwargs):
            samples = 0 if request_id.endswith("part1") else 10
            yield _Output(request_id, torch.ones(1, samples), tokens_out=1)

        ming_tts_server.engine_client.generate.side_effect = generate
        request = OpenAICreateSpeechRequest(input="One. Two. Three.", voice="lingguang", response_format="pcm")

        with pytest.raises(ValueError, match="part1 produced empty audio"):
            asyncio.run(ming_tts_server._generate_audio_bytes(request, request_id="speech-1"))

    def test_empty_middle_fragment_stops_the_stream(self, ming_tts_server, mocker):
        mocker.patch.object(ming_tts_server._adapter, "split_long_input", return_value=["One.", "Two.", "Three."])
        started: list[str] = []

        async def generate(*, prompt, request_id, **kwargs):
            started.append(request_id)
            yield _Output(request_id, [] if request_id.endswith("part1") else [torch.ones(3)])

        ming_tts_server.engine_client.generate.side_effect = generate
        request = OpenAICreateSpeechRequest(input="One. Two. Three.", voice="lingguang", stream=True)

        async def run():
            _, generator, _ = await ming_tts_server._prepare_speech_generation(
                request, request_id="speech-1", split_long_input=True
            )
            return [res async for res in generator]

        with pytest.raises(ValueError, match="part1 produced no audio"):
            asyncio.run(run())
        assert started == ["speech-1-part0", "speech-1-part1"]

    def test_fragments_of_a_seeded_request_get_distinct_seeds(self, ming_tts_server, mocker):
        mocker.patch.object(ming_tts_server._adapter, "split_long_input", return_value=["One.", "Two."])
        ming_tts_server.engine_client.generate.return_value = "generator"
        request = OpenAICreateSpeechRequest(input="One. Two.", voice="lingguang", seed=7, stream=True)
        seeds: list[int] = []

        def capture(engine_requests, **kwargs):
            seeds.extend(params[0].seed for _, _, params in engine_requests)
            return "split-generator"

        mocker.patch.object(ming_tts_server, "_generate_split_input", side_effect=capture)
        ming_tts_server.engine_client.default_sampling_params_list = [
            SamplingParams(max_tokens=2048),
            SamplingParams(max_tokens=1),
        ]

        asyncio.run(ming_tts_server._prepare_speech_generation(request, request_id="speech-1", split_long_input=True))

        assert seeds == [7, 8]

    def test_non_streaming_joins_fragments_in_text_order_when_they_finish_out_of_order(self, ming_tts_server, mocker):
        mocker.patch.object(ming_tts_server._adapter, "split_long_input", return_value=["One.", "Two.", "Three."])

        async def generate(*, prompt, request_id, **kwargs):
            index = int(request_id.rsplit("part", 1)[1])
            await asyncio.sleep(0.03 * (2 - index))  # the last fragment finishes first
            yield _Output(request_id, torch.full((1, 4), float(index + 1)), tokens_out=1)

        ming_tts_server.engine_client.generate.side_effect = generate
        request = OpenAICreateSpeechRequest(input="One. Two. Three.", voice="lingguang")

        async def run():
            _, generator, _ = await ming_tts_server._prepare_speech_generation(
                request, request_id="speech-1", split_long_input=True
            )
            outputs = [res async for res in generator]
            audio_output, audio_key = ming_tts_server._extract_audio_output(outputs[-1])
            return ming_tts_server._final_audio_tensor(audio_output[audio_key])

        joined = torch.as_tensor(asyncio.run(run())).reshape(-1)
        assert joined.tolist() == [1.0] * 4 + [2.0] * 4 + [3.0] * 4

    def test_streaming_runs_fragments_in_order_with_one_growing_chunk_list(self, ming_tts_server, mocker):
        mocker.patch.object(
            ming_tts_server._adapter,
            "split_long_input",
            return_value=["First part.", "Second part."],
        )
        events: list = []
        chunks = {"speech-1-part0": [torch.ones(3), torch.ones(4)], "speech-1-part1": [torch.ones(5)]}

        async def generate(*, prompt, request_id, **kwargs):
            events.append(("start", request_id))
            for count in range(1, len(chunks[request_id]) + 1):
                yield _Output(request_id, chunks[request_id][:count])
            events.append(("end", request_id))

        ming_tts_server.engine_client.generate.side_effect = generate
        request = OpenAICreateSpeechRequest(input="First part. Second part.", voice="lingguang", stream=True)

        async def run():
            _, generator, _ = await ming_tts_server._prepare_speech_generation(
                request, request_id="speech-1", split_long_input=True
            )
            return [[chunk.numel() for chunk in res.multimodal_output["audio"]] async for res in generator], chunks

        seen, _ = asyncio.run(run())

        assert seen == [[3], [3, 4], [3, 4, 5]]
        assert events == [
            ("start", "speech-1-part0"),
            ("end", "speech-1-part0"),
            ("start", "speech-1-part1"),
            ("end", "speech-1-part1"),
        ]

    def test_failed_fragment_cancels_the_others(self, ming_tts_server, mocker):
        mocker.patch.object(
            ming_tts_server._adapter,
            "split_long_input",
            return_value=["First part.", "Second part."],
        )
        cancelled: list[str] = []

        async def generate(*, prompt, request_id, **kwargs):
            if request_id.endswith("part0"):
                raise RuntimeError("engine failure")
            try:
                await asyncio.sleep(10)
                yield _Output(request_id, torch.ones(1, 10))
            except asyncio.CancelledError:
                cancelled.append(request_id)
                raise

        ming_tts_server.engine_client.generate.side_effect = generate
        request = OpenAICreateSpeechRequest(input="First part. Second part.", voice="lingguang")

        async def run():
            _, generator, _ = await ming_tts_server._prepare_speech_generation(
                request, request_id="speech-1", split_long_input=True
            )
            return [res async for res in generator]

        with pytest.raises(RuntimeError, match="engine failure"):
            asyncio.run(run())
        assert cancelled == ["speech-1-part1"]


def test_usage_counter_sums_engine_requests_and_takes_max_within_one():
    counter = SpeechOutputTokenCounter()
    counter.observe(_Output("a", None, tokens_out=10))
    counter.observe(_Output("a", None, tokens_out=12))
    counter.observe(_Output("b", None, tokens_out=7))
    assert counter.total() == 19

import base64
import json
import queue
import threading
from collections.abc import Callable
from urllib.parse import urlencode

import numpy as np
from websockets.sync.client import ClientConnection, connect

STT_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
STT_MODEL = "scribe_v2_realtime"
# Scribe force-commits after ~36s of uncommitted audio, cutting mid-phrase, so
# long dictations are committed at the speaker's own pauses before that.
MIN_SEGMENT_SECONDS = 20.0
MAX_SEGMENT_SECONDS = 34.0
PAUSE_SECONDS = 0.3
# A chunk quieter than this fraction of the loudest chunk so far is a pause.
PAUSE_LEVEL = 0.1
# Scribe rejects a commit with less uncommitted audio than this and closes the socket.
MIN_COMMIT_SECONDS = 0.3
RESULT_TIMEOUT_SECONDS = 10.0
BYTES_PER_SAMPLE = 2


# Streams audio to Scribe while recording so the transcript is ready almost
# as soon as the key is released, instead of uploading the whole clip after.
class TranscriptionSession:
    def __init__(
        self,
        samplerate: int,
        api_key: str,
        keyterms: list[str],
        on_transcript: Callable[[str], None],
    ):
        self._samplerate = samplerate
        self._api_key = api_key
        self._keyterms = keyterms
        self._on_transcript = on_transcript
        self._chunks: queue.Queue[bytes | None] = queue.Queue()
        self._keep = False
        threading.Thread(target=self._run, daemon=True).start()

    def feed(self, chunk: bytes):
        self._chunks.put(chunk)

    def finish(self, keep: bool):
        self._keep = keep
        self._chunks.put(None)

    def _run(self):
        results: queue.Queue[dict] = queue.Queue()
        with connect(self._url(), additional_headers={"xi-api-key": self._api_key}) as ws:
            threading.Thread(target=_forward_results, args=(ws, results), daemon=True).start()
            commits = self._stream_audio(ws)
            segments = [_committed_text(results.get(timeout=RESULT_TIMEOUT_SECONDS)) for _ in range(commits)]
        if self._keep:
            self._on_transcript(" ".join(segment for segment in segments if segment))

    def _url(self) -> str:
        params = [
            ("model_id", STT_MODEL),
            ("audio_format", f"pcm_{self._samplerate}"),
            ("commit_strategy", "manual"),
        ]
        params += [("keyterms", term) for term in self._keyterms]
        return f"{STT_URL}?{urlencode(params)}"

    def _stream_audio(self, ws: ClientConnection) -> int:
        segmenter = _Segmenter(self._samplerate)
        commits = 0
        while (chunk := self._chunks.get()) is not None:
            segmenter.add(chunk)
            commit = segmenter.is_commit_point()
            self._send(ws, chunk, commit)
            if commit:
                segmenter.mark_committed()
                commits += 1
        if segmenter.has_committable_audio():
            self._send(ws, b"", commit=True)
            commits += 1
        return commits

    def _send(self, ws: ClientConnection, chunk: bytes, commit: bool):
        ws.send(json.dumps({
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(chunk).decode(),
            "commit": commit,
            "sample_rate": self._samplerate,
        }))


# Tracks uncommitted audio and the current run of quiet so commits land in pauses.
class _Segmenter:
    def __init__(self, samplerate: int):
        self._bytes_per_second = samplerate * BYTES_PER_SAMPLE
        self._uncommitted = 0.0
        self._pause = 0.0
        self._peak = 0.0

    def add(self, chunk: bytes):
        seconds = len(chunk) / self._bytes_per_second
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        level = float(np.sqrt(np.mean(samples**2)))
        self._peak = max(self._peak, level)
        self._pause = self._pause + seconds if level < self._peak * PAUSE_LEVEL else 0.0
        self._uncommitted += seconds

    def is_commit_point(self) -> bool:
        if self._uncommitted >= MAX_SEGMENT_SECONDS:
            return True
        return self._uncommitted >= MIN_SEGMENT_SECONDS and self._pause >= PAUSE_SECONDS

    def has_committable_audio(self) -> bool:
        return self._uncommitted >= MIN_COMMIT_SECONDS

    def mark_committed(self):
        self._uncommitted = 0.0
        self._pause = 0.0


def _forward_results(ws: ClientConnection, results: queue.Queue):
    for raw in ws:
        message = json.loads(raw)
        if message["message_type"] == "committed_transcript" or "error" in message:
            results.put(message)


def _committed_text(message: dict) -> str:
    if message["message_type"] != "committed_transcript":
        raise RuntimeError(f"Scribe {message['message_type']}: {message['error']}")
    return message["text"]

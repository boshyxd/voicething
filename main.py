import io
import os
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import keyboard
import numpy as np
import pystray
import pyperclip
import sounddevice as sd
from deepgram import DeepgramClient
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

SAMPLE_RATE = 16000
MIN_DURATION = 0.3
CHIME_DIR = Path(__file__).parent / "chimes"
WIN_KEYS = {"left windows", "right windows"}
CTRL_KEYS = {"left ctrl", "right ctrl", "ctrl"}

_api_key = os.environ.get("DEEPGRAM_API_KEY")
if not _api_key:
    print("DEEPGRAM_API_KEY not set in .env", file=sys.stderr)
    sys.exit(1)

_client = DeepgramClient(api_key=_api_key)
_recording = False
_audio_frames: list[np.ndarray] = []
_record_start = 0.0
_stream: sd.InputStream | None = None
_win_held = False
_ctrl_held = False
_clipboard_only = False
_tray_icon: pystray.Icon | None = None


def _generate_chimes():
    CHIME_DIR.mkdir(exist_ok=True)
    start_path = CHIME_DIR / "start.wav"
    stop_path = CHIME_DIR / "stop.wav"
    if start_path.exists() and stop_path.exists():
        return
    amp = 0.08
    note_dur = 0.08
    t = np.linspace(0, note_dur, int(SAMPLE_RATE * note_dur), endpoint=False)
    envelope = np.sin(np.linspace(0, np.pi, len(t)))
    c5 = np.sin(2 * np.pi * 523.25 * t) * envelope * amp
    e5 = np.sin(2 * np.pi * 659.25 * t) * envelope * amp
    start_chime = np.concatenate([c5, e5])
    stop_chime = np.concatenate([e5, c5])
    for path, data in ((start_path, start_chime), (stop_path, stop_chime)):
        samples = (data * 32767).astype(np.int16)
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples.tobytes())


def _play_chime(name):
    path = CHIME_DIR / f"{name}.wav"
    if not path.exists():
        return
    with wave.open(str(path), "r") as wf:
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    sd.play(audio, SAMPLE_RATE)


def _audio_callback(indata, _frames, _time_info, _status):
    _audio_frames.append(indata.copy())


def _start_recording():
    global _recording, _record_start, _stream, _audio_frames
    if _recording:
        return
    _audio_frames = []
    _stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=_audio_callback
    )
    _stream.start()
    _record_start = time.monotonic()
    _recording = True
    _play_chime("start")


def _stop_recording():
    global _recording, _stream
    if not _recording:
        return
    _recording = False
    _stream.stop()
    _stream.close()
    _stream = None
    elapsed = time.monotonic() - _record_start
    _play_chime("stop")
    if elapsed < MIN_DURATION:
        return
    audio = np.concatenate(_audio_frames)
    threading.Thread(target=_transcribe_and_paste, args=(audio,), daemon=True).start()


def _transcribe_and_paste(audio: np.ndarray):
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    response = _client.listen.v1.media.transcribe_file(
        request=buf.getvalue(), model="nova-3", smart_format=True
    )
    transcript = response.results.channels[0].alternatives[0].transcript
    if not transcript.strip():
        return
    pyperclip.copy(transcript)
    if not _clipboard_only:
        keyboard.send("ctrl+v")


def _on_key(event):
    global _win_held, _ctrl_held
    name = event.name.lower() if event.name else ""
    is_win = name in WIN_KEYS
    is_ctrl = name in CTRL_KEYS
    if not is_win and not is_ctrl:
        return
    if event.event_type == "down":
        if is_win:
            _win_held = True
        if is_ctrl:
            _ctrl_held = True
        if _win_held and _ctrl_held:
            _start_recording()
    else:
        if is_win:
            _win_held = False
        if is_ctrl:
            _ctrl_held = False
        _stop_recording()


def _create_tray_icon(color=(100, 180, 255)):
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([8, 8, size - 8, size - 8], fill=color)
    return img


def _toggle_mode(icon, _item):
    global _clipboard_only
    _clipboard_only = not _clipboard_only
    color = (220, 80, 80) if _clipboard_only else (100, 180, 255)
    icon.icon = _create_tray_icon(color)
    label = "Clipboard only" if _clipboard_only else "Type & paste"
    icon.title = f"Voice Typer — {label}"


def _notify(icon: pystray.Icon, message: str):
    try:
        icon.notify(message, "Voice Typer")
    except Exception:
        pass


def _kill_processes(icon: pystray.Icon, *image_names: str):
    killed = []
    not_found = []
    failed = []
    for name in image_names:
        result = subprocess.run(
            ["taskkill", "/F", "/T", "/IM", name],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode == 0:
            killed.append(name)
            continue
        stderr = (result.stderr or "").lower()
        stdout = (result.stdout or "").lower()
        if "not found" in stderr or "no running instance" in stderr or "not found" in stdout:
            not_found.append(name)
        else:
            failed.append(name)
    if killed:
        _notify(icon, f"Terminated: {', '.join(killed)}")
    elif not failed:
        _notify(icon, f"No running processes found.")
    if failed:
        _notify(icon, f"Failed to terminate: {', '.join(failed)}")


def _exit_app(icon: pystray.Icon, _item):
    keyboard.unhook_all()
    icon.stop()


def main():
    _generate_chimes()
    keyboard.hook(_on_key)
    icon = pystray.Icon(
        "voicething",
        _create_tray_icon(),
        "Voice Typer",
        menu=pystray.Menu(
            pystray.MenuItem(
                lambda _item: "Mode: clipboard only" if _clipboard_only else "Mode: type & paste",
                _toggle_mode,
            ),
            pystray.MenuItem(
                "End all node.exe",
                lambda icon, _item: _kill_processes(icon, "node.exe"),
            ),
            pystray.MenuItem(
                "End all bash/cat/date.exe",
                lambda icon, _item: _kill_processes(icon, "bash.exe", "cat.exe", "date.exe"),
            ),
            pystray.MenuItem("Exit", _exit_app),
        ),
    )
    icon.run()


if __name__ == "__main__":
    if sys.executable.endswith("python.exe"):
        subprocess.Popen(
            [sys.executable.replace("python.exe", "pythonw.exe")] + sys.argv,
            creationflags=subprocess.DETACHED_PROCESS,
        )
        sys.exit(0)
    main()

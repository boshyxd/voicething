import os
import queue
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pystray
import pyperclip
import sounddevice as sd
from deepgram import DeepgramClient
from deepgram.extensions.types.sockets import ListenV1ControlMessage, ListenV1ResultsEvent
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from pynput import keyboard as pkeyboard

load_dotenv()

SAMPLE_RATE = 16000
MIN_DURATION = 0.3
WINDOWS_TAIL_PADDING = 0.25
IDLE_TRAY_COLOR = (100, 180, 255)
RECORDING_TRAY_COLOR = (70, 200, 100)
CLIPBOARD_TRAY_COLOR = (220, 80, 80)
CHIME_DIR = Path(__file__).parent / "chimes"
IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform.startswith("win")

if IS_MACOS:
    MOD1_KEYS = {pkeyboard.Key.alt, pkeyboard.Key.alt_l, pkeyboard.Key.alt_r}
    MOD2_KEYS = {pkeyboard.Key.shift, pkeyboard.Key.shift_l, pkeyboard.Key.shift_r}
    PASTE_MOD = pkeyboard.Key.cmd
    HOTKEY_LABEL = "Option+Shift"
else:
    MOD1_KEYS = {pkeyboard.Key.cmd, pkeyboard.Key.cmd_l, pkeyboard.Key.cmd_r}
    MOD2_KEYS = {pkeyboard.Key.ctrl, pkeyboard.Key.ctrl_l, pkeyboard.Key.ctrl_r}
    PASTE_MOD = pkeyboard.Key.ctrl
    HOTKEY_LABEL = "Win+Ctrl"

if IS_MACOS:
    from AVFoundation import AVAudioEngine

_api_key = os.environ.get("DEEPGRAM_API_KEY")
if not _api_key:
    raise RuntimeError("DEEPGRAM_API_KEY not set in .env")

_client = DeepgramClient(api_key=_api_key)
_recording = False
_session: "_TranscriptionSession | None" = None
_chimes: dict[str, np.ndarray] = {}
_record_start = 0.0
_stream: sd.InputStream | None = None
_engine = AVAudioEngine.alloc().init() if IS_MACOS else None
_capture_rate = SAMPLE_RATE
_mod1_held = False
_mod2_held = False
_clipboard_only = False
_tray_icon: pystray.Icon | None = None
_kb_controller = pkeyboard.Controller()
_listener: pkeyboard.Listener | None = None
_audio_queue: queue.Queue[str] = queue.Queue()


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


def _load_chimes():
    for name in ("start", "stop"):
        with wave.open(str(CHIME_DIR / f"{name}.wav"), "r") as wf:
            raw = wf.readframes(wf.getnframes())
        _chimes[name] = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0


def _play_chime(name):
    if IS_MACOS:
        subprocess.Popen(["afplay", str(CHIME_DIR / f"{name}.wav")])
        return
    sd.play(_chimes[name], SAMPLE_RATE, latency="low")


# Streams audio to Deepgram while recording so the transcript is ready almost
# as soon as the key is released, instead of uploading the whole clip after.
class _TranscriptionSession:
    def __init__(self, samplerate: int):
        self._samplerate = samplerate
        self._chunks: queue.Queue[bytes | None] = queue.Queue()
        self._keep = False
        threading.Thread(target=self._run, daemon=True).start()

    def feed(self, chunk: bytes):
        self._chunks.put(chunk)

    def finish(self, keep: bool):
        self._keep = keep
        self._chunks.put(None)

    def _run(self):
        parts: list[str] = []
        with _client.listen.v1.connect(
            model="nova-3",
            encoding="linear16",
            sample_rate=str(self._samplerate),
            smart_format="true",
        ) as sock:
            reader = threading.Thread(target=_collect_final_transcripts, args=(sock, parts))
            reader.start()
            while (chunk := self._chunks.get()) is not None:
                sock.send_media(chunk)
            sock.send_control(ListenV1ControlMessage(type="CloseStream"))
            reader.join()
        if self._keep:
            _deliver_transcript(" ".join(part for part in parts if part))


def _collect_final_transcripts(sock, parts: list[str]):
    for message in sock:
        if isinstance(message, ListenV1ResultsEvent) and message.is_final:
            parts.append(message.channel.alternatives[0].transcript)


def _audio_callback(indata, _frames, _time_info, _status):
    _session.feed(indata.tobytes())


def _tap_block(buf, _when):
    n = int(buf.frameLength())
    samples = np.frombuffer(buf.floatChannelData()[0].as_buffer(n), dtype=np.float32)
    _session.feed((np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes())


def _start_recording():
    global _recording, _record_start, _stream, _session, _capture_rate
    if _recording:
        return
    _play_chime("start")
    if IS_MACOS:
        node = _engine.inputNode()
        fmt = node.outputFormatForBus_(0)
        _capture_rate = int(fmt.sampleRate())
        _session = _TranscriptionSession(_capture_rate)
        node.installTapOnBus_bufferSize_format_block_(0, 1024, fmt, _tap_block)
        ok, err = _engine.startAndReturnError_(None)
        if not ok:
            node.removeTapOnBus_(0)
            raise RuntimeError(f"AVAudioEngine start failed: {err}")
    else:
        _capture_rate = SAMPLE_RATE
        _session = _TranscriptionSession(_capture_rate)
        _stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=_audio_callback
        )
        _stream.start()
    _record_start = time.monotonic()
    _recording = True
    _refresh_tray_icon()


def _stop_recording():
    global _recording, _stream
    if not _recording:
        return
    _recording = False
    _play_chime("stop")
    _refresh_tray_icon()
    elapsed = time.monotonic() - _record_start
    if IS_MACOS:
        _engine.stop()
        _engine.inputNode().removeTapOnBus_(0)
    else:
        stream = _stream
        _stream = None
        if IS_WINDOWS:
            time.sleep(WINDOWS_TAIL_PADDING)
        # abort() returns in ~10ms; stop() waits ~400ms on MME draining buffers
        # the tail padding has already collected.
        stream.abort()
        stream.close()
    _session.finish(keep=elapsed >= MIN_DURATION)


# Audio capture start/stop must never run on the pynput event-tap thread and
# must never run concurrently. PortAudio's CoreAudio backend deadlocks when
# stream lifecycle calls race its property listeners (which is also why macOS
# uses AVAudioEngine and afplay instead), freezing the tap and leaving the mic
# stuck on.
def _audio_worker():
    while True:
        command = _audio_queue.get()
        if command == "start":
            _start_recording()
        elif command == "stop":
            _stop_recording()


def _send_paste():
    with _kb_controller.pressed(PASTE_MOD):
        _kb_controller.press("v")
        _kb_controller.release("v")


def _deliver_transcript(transcript: str):
    if not transcript.strip():
        return
    pyperclip.copy(transcript)
    if not _clipboard_only:
        _send_paste()


def _on_press(key):
    global _mod1_held, _mod2_held
    if key in MOD1_KEYS:
        _mod1_held = True
    elif key in MOD2_KEYS:
        _mod2_held = True
    else:
        return
    if _mod1_held and _mod2_held:
        _audio_queue.put("start")


def _on_release(key):
    global _mod1_held, _mod2_held
    if key in MOD1_KEYS:
        _mod1_held = False
    elif key in MOD2_KEYS:
        _mod2_held = False
    else:
        return
    _audio_queue.put("stop")


def _create_tray_icon(color=IDLE_TRAY_COLOR):
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([8, 8, size - 8, size - 8], fill=color)
    return img


def _refresh_tray_icon(icon=None):
    target = icon or _tray_icon
    if target is None:
        return
    if _recording:
        color = RECORDING_TRAY_COLOR
    elif _clipboard_only:
        color = CLIPBOARD_TRAY_COLOR
    else:
        color = IDLE_TRAY_COLOR
    target.icon = _create_tray_icon(color)


def _toggle_mode(icon, _item):
    global _clipboard_only
    _clipboard_only = not _clipboard_only
    _refresh_tray_icon(icon)
    label = "Clipboard only" if _clipboard_only else "Type & paste"
    icon.title = f"Voice Typer ({HOTKEY_LABEL}) — {label}"


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


def _kill_processes_except_self(icon: pystray.Icon, *image_names: str):
    current_pid = os.getpid()
    image_names_literal = ", ".join(f"'{name.lower()}'" for name in image_names)
    ps_script = f"""
$names = @({image_names_literal})
$currentPid = {current_pid}
$processes = Get-CimInstance Win32_Process |
    Where-Object {{ $names -contains $_.Name.ToLowerInvariant() -and $_.ProcessId -ne $currentPid }}
if (-not $processes) {{
    exit 2
}}
$failed = @()
foreach ($process in $processes) {{
    try {{
        Stop-Process -Id $process.ProcessId -Force -ErrorAction Stop
    }} catch {{
        $failed += $process.Name
    }}
}}
if ($failed.Count -gt 0) {{
    Write-Error ($failed -join ', ')
    exit 1
}}
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode == 0:
        _notify(icon, f"Terminated: {', '.join(image_names)}")
    elif result.returncode == 2:
        _notify(icon, "No running processes found.")
    else:
        failed = (result.stderr or "unknown error").strip()
        _notify(icon, f"Failed to terminate: {failed}")


def _exit_app(icon: pystray.Icon, _item):
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None
    icon.stop()


def _build_menu():
    items = [
        pystray.MenuItem(
            lambda _item: "Mode: clipboard only" if _clipboard_only else "Mode: type & paste",
            _toggle_mode,
        ),
    ]
    if IS_WINDOWS:
        items.extend([
            pystray.MenuItem(
                "End all node.exe",
                lambda icon, _item: _kill_processes(icon, "node.exe"),
            ),
            pystray.MenuItem(
                "End all bash/cat/date/grep.exe",
                lambda icon, _item: _kill_processes(icon, "bash.exe", "cat.exe", "date.exe", "grep.exe"),
            ),
            pystray.MenuItem(
                "End all git/gh.exe",
                lambda icon, _item: _kill_processes(
                    icon,
                    "git.exe",
                    "gh.exe",
                    "git-lfs.exe",
                    "git-credential-manager.exe",
                    "git-remote-http.exe",
                    "git-remote-https.exe",
                    "git-remote-ftp.exe",
                    "git-remote-ftps.exe",
                    "git-upload-pack.exe",
                    "git-receive-pack.exe",
                ),
            ),
            pystray.MenuItem(
                "End all Python except this app",
                lambda icon, _item: _kill_processes_except_self(
                    icon,
                    "python.exe",
                    "pythonw.exe",
                    "py.exe",
                    "pyw.exe",
                ),
            ),
            pystray.MenuItem(
                "End all terminals",
                lambda icon, _item: _kill_processes(
                    icon,
                    "OpenConsole.exe",
                    "WindowsTerminal.exe",
                    "terminal.exe",
                    "wt.exe",
                    "powershell.exe",
                    "pwsh.exe",
                    "conhost.exe",
                    "ConsoleHost.exe",
                ),
            ),
        ])
    items.append(pystray.MenuItem("Exit", _exit_app))
    return pystray.Menu(*items)


def _hide_dock_icon_macos():
    try:
        from AppKit import NSApplication
        NSApplication.sharedApplication().setActivationPolicy_(1)
    except Exception:
        pass


def main():
    global _listener, _tray_icon
    if IS_MACOS:
        _hide_dock_icon_macos()
    icon = pystray.Icon(
        "voicething",
        _create_tray_icon(),
        f"Voice Typer ({HOTKEY_LABEL})",
        menu=_build_menu(),
    )
    _tray_icon = icon
    _generate_chimes()
    _load_chimes()
    threading.Thread(target=_audio_worker, daemon=True).start()
    _listener = pkeyboard.Listener(on_press=_on_press, on_release=_on_release)
    _listener.start()
    icon.run()


if __name__ == "__main__":
    if IS_WINDOWS and sys.executable.endswith("python.exe"):
        subprocess.Popen(
            [sys.executable.replace("python.exe", "pythonw.exe")] + sys.argv,
            creationflags=subprocess.DETACHED_PROCESS,
        )
        sys.exit(0)
    if IS_MACOS and os.environ.get("VOICETHING_DETACHED") != "1":
        env = os.environ.copy()
        env["VOICETHING_DETACHED"] = "1"
        subprocess.Popen(
            [sys.executable] + sys.argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        sys.exit(0)
    main()

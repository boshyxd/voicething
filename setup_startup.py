import os
import sys
from pathlib import Path

STARTUP_DIR = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Startup"
VBS_PATH = STARTUP_DIR / "voicething.vbs"


def install():
    script_dir = Path(__file__).parent.resolve()
    vbs_content = (
        'Set WshShell = CreateObject("WScript.Shell")\n'
        f'WshShell.CurrentDirectory = "{script_dir}"\n'
        'WshShell.Run "pythonw.exe main.py", 0, False\n'
    )
    VBS_PATH.write_text(vbs_content)


def remove():
    if not VBS_PATH.exists():
        return
    VBS_PATH.unlink()


if __name__ == "__main__":
    if "--remove" in sys.argv:
        remove()
    else:
        install()

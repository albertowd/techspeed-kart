"""Smoke-test an AC car: launch acs.exe windowed in practice, screenshot, read the log, restore configs.

    python tools/dev/ac_test.py <car_id> <skin> <out.png> [seconds] [track] [layout]
"""
import os
import shutil
import subprocess
import sys
import time

AC = r"D:\SteamLibrary\steamapps\common\assettocorsa"
DOCS = os.path.join(os.path.expanduser("~"), "OneDrive", "Documents", "Assetto Corsa")
CFG = os.path.join(DOCS, "cfg")

car, skin, shot = sys.argv[1], sys.argv[2], sys.argv[3]
wait = int(sys.argv[4]) if len(sys.argv) > 4 else 50
track = sys.argv[5] if len(sys.argv) > 5 else "techspeed_velopark"
layout = sys.argv[6] if len(sys.argv) > 6 else "vp_1500"

RACE = f"""[HEADER]
VERSION=2
[RACE]
TRACK={track}
CONFIG_TRACK={layout}
MODEL={car}
MODEL_CONFIG=
SKIN={skin}
CARS=1
AI_LEVEL=100
FIXED_SETUP=0
PENALTIES=0
RACE_LAPS=5
DRIFT_MODE=0
JUMP_START_PENALTY=0
[CAR_0]
SETUP=
SKIN={skin}
MODEL=-
MODEL_CONFIG=
BALLAST=0
RESTRICTOR=0
DRIVER_NAME=Test
NATIONALITY=Brazil
NATION_CODE=BRA
[SESSION_0]
NAME=Practice
TYPE=1
DURATION_MINUTES=30
SPAWN_SET=START
[REPLAY]
ACTIVE=0
[REMOTE]
ACTIVE=0
[BENCHMARK]
ACTIVE=0
[RESTART]
ACTIVE=0
[GHOST_CAR]
ENABLED=0
RECORDING=0
PLAYING=0
LOAD=0
FILE=
[LIGHTING]
SUN_ANGLE=-45
TIME_MULT=1
CLOUD_SPEED=0.2
[TEMPERATURE]
AMBIENT=22
ROAD=26
[WEATHER]
NAME=3_clear
[WIND]
SPEED_KMH_MIN=0
SPEED_KMH_MAX=0
DIRECTION_DEG=0
[DYNAMIC_TRACK]
SESSION_START=100
RANDOMNESS=0
SESSION_TRANSFER=100
LAP_GAIN=1
[GROOVE]
VIRTUAL_LAPS=10
MAX_LAPS=30
STARTING_LAPS=0
[LAP_INVALIDATOR]
ALLOWED_TYRES_OUT=-1
[OPTIONS]
USE_MPH=0
"""

backups = []
for name in ("race.ini", "video.ini"):
    p = os.path.join(CFG, name)
    shutil.copy(p, p + ".kart_test_bak")
    backups.append(p)
try:
    open(os.path.join(CFG, "race.ini"), "w").write(RACE)
    vid = open(os.path.join(CFG, "video.ini") + ".kart_test_bak").read().splitlines()
    out = []
    for line in vid:
        k = line.split("=")[0].strip().upper()
        if k == "FULLSCREEN":
            line = "FULLSCREEN=0"
        elif k == "WIDTH":
            line = "WIDTH=1600"
        elif k == "HEIGHT":
            line = "HEIGHT=900"
        out.append(line)
    open(os.path.join(CFG, "video.ini"), "w").write("\n".join(out) + "\n")
    proc = subprocess.Popen([os.path.join(AC, "acs.exe")], cwd=AC)
    t0 = time.time()
    while time.time() - t0 < wait and proc.poll() is None:
        time.sleep(1)
    alive = proc.poll() is None
    if alive and os.environ.get("AC_DRIVE"):
        # bring AC forward, click the start (logo) button, wait, screenshot cockpit, F1 -> next camera
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ac_click.ps1")])
        time.sleep(int(os.environ.get("AC_DRIVE")))
    ps = ("Add-Type -AssemblyName System.Windows.Forms,System.Drawing;"
          "$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds;"
          "$bmp=New-Object System.Drawing.Bitmap $b.Width,$b.Height;"
          "$g=[System.Drawing.Graphics]::FromImage($bmp);$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size);"
          f"$bmp.Save('{shot}');")
    shots = bool(os.environ.get("KART_TEST_SCREENSHOT"))      # opt-in: captures the whole desktop
    if shots:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps])
    if shots and alive and os.environ.get("AC_DRIVE"):
        keys = ("Add-Type -AssemblyName System.Windows.Forms;$w=New-Object -ComObject WScript.Shell;"
                "$w.AppActivate((Get-Process acs).Id)|Out-Null;Start-Sleep 1;[System.Windows.Forms.SendKeys]::SendWait('{F1}');Start-Sleep 3;")
        subprocess.run(["powershell", "-NoProfile", "-Command", keys])
        subprocess.run(["powershell", "-NoProfile", "-Command", ps.replace(shot, shot.replace(".png", "_b.png"))])
    if alive:
        proc.kill()
    print("acs alive after %ds: %s (exit %s)" % (wait, alive, proc.poll()))
finally:
    time.sleep(2)
    for p in backups:
        shutil.copy(p + ".kart_test_bak", p)
        os.remove(p + ".kart_test_bak")
    print("configs restored")

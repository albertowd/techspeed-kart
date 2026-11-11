"""Smoke-test the BeamNG kart: launch windowed on gridmap with a config, screenshot, close, grep the log.

    python tools/dev/bng_test.py <config.pc> <out.png> [seconds]
"""
import os
import re
import subprocess
import sys
import time

EXE = r"D:\SteamLibrary\steamapps\common\BeamNG.drive\BeamNG.drive.exe"
LOG = os.path.join(os.environ["LOCALAPPDATA"], "BeamNG", "BeamNG.drive", "current", "beamng.log")

pc, shot = sys.argv[1], sys.argv[2]
wait = int(sys.argv[3]) if len(sys.argv) > 3 else 120
proc = subprocess.Popen([EXE, "-level", "gridmap_v2", "-vehicleConfig", "techspeed_kart/" + pc],
                        cwd=os.path.dirname(EXE))
t0 = time.time()
while time.time() - t0 < wait and proc.poll() is None:
    time.sleep(2)
ps = ("Add-Type -AssemblyName System.Windows.Forms,System.Drawing;"
      "$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds;"
      "$bmp=New-Object System.Drawing.Bitmap $b.Width,$b.Height;"
      "$g=[System.Drawing.Graphics]::FromImage($bmp);$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size);"
      f"$bmp.Save('{shot}');")
if os.environ.get("KART_TEST_SCREENSHOT"):   # opt-in: captures the whole desktop
    subprocess.run(["powershell", "-NoProfile", "-Command", ps])
alive = proc.poll() is None
subprocess.run(["taskkill", "/IM", "BeamNG.drive.x64.exe", "/F"], capture_output=True)
subprocess.run(["taskkill", "/IM", "BeamNG.drive.exe", "/F"], capture_output=True)
print("alive:", alive)
time.sleep(3)
txt = open(LOG, encoding="utf-8", errors="replace").read().splitlines()
pat = re.compile(r"techspeed|tskart|jbeam|flexbod|\|E\||\|W\|.*(mesh|material|vehicle)", re.I)
hits = [l for l in txt if pat.search(l)]
print("\n".join(hits[-80:]))

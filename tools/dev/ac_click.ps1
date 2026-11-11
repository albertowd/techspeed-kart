# Bring the acs window to the front and left-click at an offset from its top-left corner.
param([int]$dx = 105, [int]$dy = 140)
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class W {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(int f, int x, int y, int d, int e);
  public struct RECT { public int L, T, R, B; }
}
"@
$p = Get-Process acs -ErrorAction Stop | Select-Object -First 1
$h = $p.MainWindowHandle
[W]::SetForegroundWindow($h) | Out-Null
Start-Sleep -Milliseconds 800
$r = New-Object W+RECT
[W]::GetWindowRect($h, [ref]$r) | Out-Null
[W]::SetCursorPos($r.L + $dx, $r.T + $dy) | Out-Null
Start-Sleep -Milliseconds 300
[W]::mouse_event(2, 0, 0, 0, 0)
Start-Sleep -Milliseconds 80
[W]::mouse_event(4, 0, 0, 0, 0)
"window: $($r.L),$($r.T) - $($r.R),$($r.B)"

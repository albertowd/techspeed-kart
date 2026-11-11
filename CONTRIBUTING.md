# Contributing to techspeed-kart

How the karts are made and built: the repository's layout, the tools, the sources and what each converter expects.
For what the mod is and how to install it, see the [README](README.md).

## Folders

| Folder | What is in it |
|---|---|
| `source/` | The karts' sources: `rental/` (v2) and `rental_v1/`, each a multi-piece `.blend` (with an `.fbx` copy) and its textures (with kart 1's livery), and `logos/` (the Velopark logo the liveries carry). Both games are built from these. See [Sources](#sources). |
| `tools/` | Python scripts that make the sources, convert them to each game, paint the liveries and render the pictures. `tools/dev/` holds the checks and the photo-matching tools. |
| `references/` | Photos and videos of the real karts the model is made from, and the GoKart 0.5c mod archive. |
| `docs/` | The pictures in the README, which the overtake.gg page uses too. |
| `overtake/` | The overtake.gg description; its `Version:` line is the mod's version. |
| `.github/workflows/` | The release build. |

The `.blend`, `.fbx`, `.kn5` and `.dds` files and the source textures are stored with Git LFS. Install
[Git LFS](https://git-lfs.com) before cloning.

## Setting up

The tools need Python 3.13 and the packages in `requirements.txt` (`bpy` is Blender as a Python module):

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

They also read files that are kept out of the repository:

| Path | What it is | How to get it back |
|---|---|---|
| `reference/GoKart_0.5c/` | The files the Assetto Corsa build takes from the GoKart 0.5c mod: its sound banks. | `7z x references/GoKart_0.5c.7z -oreference/GoKart_0.5c gokart125/sfx gokart390/sfx -r` |
| `build/` | The built karts, the painted liveries and the archives. | `python tools/build_all.py` |

## Building

```
.venv\Scripts\python tools\build_all.py                 # pictures, both games and the archives in build/dist
.venv\Scripts\python tools\build_all.py --no-previews   # skip the livery preview renders
.venv\Scripts\python tools\build_all.py --regen-source  # throw the .blend sources away and make them again
.venv\Scripts\python tools\build_all.py --release       # every livery (36 per version), as a release ships
.venv\Scripts\python tools\render_docs.py               # the pictures in docs/
```

It makes, with the version from `overtake/OVERTAKE.md`:

- `build/dist/techspeed-kart-<version>-ac.7z`: the six Assetto Corsa rentals as `content/cars/<kart>`.
- `build/dist/techspeed-kart-<version>-beamng.zip`: one BeamNG.drive vehicle (`vehicles/techspeed_kart`) with the
  six rentals as configurations.

Nothing is copied into the games; the unpacked builds are in `build/ac` and `build/beamng`. Install the archives by
hand to try them (see the README).

A local build carries one livery per kart (number 1 of each fleet), to keep it quick. The full fleets are built only
for a release: publishing a GitHub release runs `.github/workflows/release.yml`, which builds with `--release` and
attaches both archives to the release. The release tag must match the version in `overtake/OVERTAKE.md` (`v0.1.0` or
`0.1.0`), and the workflow needs the repository secret `GOKART_SFX_URL`, a download link to `GoKart_0.5c.7z` for the
AC sound banks (`--release` stops without them rather than ship silent karts).

| Kart | AC folder | BeamNG config |
|---|---|---|
| Techspeed Rental v2 9hp | `techspeed_rental_9hp` | `rental_9hp` |
| Techspeed Rental v2 11hp | `techspeed_rental_11hp` | `rental_11hp` |
| Techspeed Rental v2 13hp | `techspeed_rental_13hp` | `rental_13hp` |
| Techspeed Rental v1 9hp | `techspeed_rental_v1_9hp` | `rental_v1_9hp` |
| Techspeed Rental v1 11hp | `techspeed_rental_v1_11hp` | `rental_v1_11hp` |
| Techspeed Rental v1 13hp | `techspeed_rental_v1_13hp` | `rental_v1_13hp` |

Engine curves, weights, gearing, brakes, fuel and tyre grip are in `tools/cars.py`. The racing kart's definitions stay
there too but it isn't built for now (`cars.BUILD_VARIANTS`).

## Sources

`tools/kart_model.py` describes the karts from the photos and standard CIK sizes, and `tools/generate_source.py` turns
that into `source/<variant>/<variant>.blend`. From there the `.blend` is the source: the build reads it and never
writes it again unless `--regen-source` is given, so changes made in Blender reach both games. What the converters
expect:

- Every object hangs from an empty named after a game node: `BODY`, `STEER_HR` (the steering wheel, turning about its
  Z axis), `WHEEL_LF`, `WHEEL_RF`, `WHEEL_LR`, `WHEEL_RR` (everything under them spins with the wheel) and `SUSP_LF`,
  `SUSP_RF` (the front spindles, which steer).
- Material names are the keys of `tools/materials.py`; add a new material there.
- Wheel positions and tyre sizes for the physics are measured from the `WHEEL_` empties and the `tyre` meshes.
- Modifiers are applied when building. Axes: x right, y forward, z up, origin on the ground halfway between the axles.
- Objects hidden from render are not exported. Some parts come with an alternative kept in the `.blend`, hidden: the
  Honda air cleaner (the box is exported; the green cone filter, `air_filter*`, is hidden) and, on the v2, the ballast
  box beside the left pod with its strap swung open and the weights in it (`pod_ballast_strap_open` and the weights
  `pod_ballast_steel`, `_tank_plastic`, `_oil_cap` and `_engine_red`; the closed strap, `pod_ballast_lid`, is
  exported). Unhide one and hide the other in Blender to switch, or set `air_cleaner` in `kart_model.VARIANTS`.

### Liveries

The liveries are painted by `tools/textures.py` from its `SKINS` list, one 2048 px atlas per skin. They are not kept in
the source: every build paints the ones it carries into `build/skins/<variant>/<skin>/` (`tools/paint_skins.py`), one
per kart locally, all 36 in a release. Only kart 1's livery is in the source, as `source/<variant>/textures/body_livery.png`
(the one the `.blend` shows). To change a livery, change `SKINS` or the painting code in `tools/textures.py`.

The rentals come in their fleets' liveries, numbered 1 to 36 (`FLEET_SIZE`): the v2 in gloss black with red and orange
flames, the v1 in black, red and gold; the Velopark logo is `source/logos/velopark.png`. Every third kart
(`WORN_EVERY`) runs worn tyres: its skin folder also holds a `tyre.png` without the new-tyre red line.

## The converters

`tools/export_ac.py` writes the kn5 models (`tools/kn5.py` reads and writes them), the collider and the data files.
`tools/ac_driver.py` seats AC's own driver (`driver_80`) in the kart: it writes the posture and the steering animation
from the driver's bones, kept in `tools/driver_80_skeleton.json` (refresh it from the game with
`python tools/dev/driver_skeleton.py`). Things that only work with Custom Shaders Patch:

- `car.ini` uses CSP's extended physics (`[HEADER] VERSION=extended-2`): only then does CSP run the car physics script
  `data/script.lua`, which holds first gear (no N or R) and replaces AC's clutch with the centrifugal clutch and its
  heat. `export_ac.CLUTCH_THERMALS` holds the clutch's thermal numbers for both games.
- The cockpit plays the exterior sound events (`sfx/GUIDs.txt`), with the engine louder through
  `[AUDIO_VOLUME] ENGINE_INT` in `extension/ext_config.ini`.
- AC's AI plans its fuel and pit stops from `data/fuel_cons.ini` (`KM_PER_LITER`), not from `car.ini`'s
  `CONSUMPTION`; `export_ac` works it out from the consumption. Without it the AI reckoned a tank lasted one lap,
  pitted on lap 1 and was marked retired. AC's `log.txt` shows the plan (`Max laps with max fuel`).

`tools/export_beamng.py` writes the Collada meshes and the jbeam: a rigid steel frame with the driver's weight on the
seat, spindles that steer about the kingpins, a centrifugal clutch (with BeamNG's own clutch heat), a solid rear axle
and no gearbox device, and the plastic bodywork as its own light, softly mounted part, so only it shakes. The driver
camera isn't treated as inside a cabin, so the onboard sound isn't muffled.

## Checks and photo tools

- `tools/dev/bng_step_check.py`: the BeamNG chassis' stability (every build should report `OK` for both).
- `tools/dev/bng_stability.py`: jbeam nodes too light for their beams.
- `tools/dev/check_clearance.py`: the seat stays at least 2 cm from the engine and radiator parts.
- `tools/dev/check_tube_ends.py <variant>`: tube ends that touch nothing (a tube stopping short in the air).
- `tools/dev/check_faces.py <variant>`: parts the games would draw inside out (AC and BeamNG draw only the front of
  each face, so an inside-out part shows as a hole; Blender's renders never show it).
- `tools/dev/photo_match.py`: renders the source from the cameras of the studio and track photos (fitted to the wheels
  and hubs) and puts each photo over its render in `build/renders/match/<tag>/`; `--set outline=<parts>|<parts>` draws
  each group's silhouette over the photo, the most reliable way to compare shapes.
- `tools/dev/photo_triangulate.py`: a point's position from its pixels in two or more of those photos;
  `tools/dev/photo_unproject.py`: where a pixel lands on a known plane.

## Tested in game

- Assetto Corsa: the rentals drive at Techspeed Velopark: cockpit cameras, steering, the pedals and the driver's feet
  (with Custom Shaders Patch). CSP loads the car physics script.
- BeamNG.drive 0.40: the rentals drive, with the pedals moving and the previews in the vehicle selector.
- Not yet seen in game: the driver's new posture in Assetto Corsa, the 36 fleet liveries, the onboard sound in both
  games, the clutch heat, the AI's fuel plan with `fuel_cons.ini`, and the latest bodywork (nose, left pod, ballast
  box).

## Known gaps

- The Assetto Corsa sounds come from GoKart 0.5c; ask its authors before releasing them. BeamNG.drive uses the game's
  three-cylinder engine samples.
- The physics are a first pass, not yet tuned by driving.
- In BeamNG.drive the fuel burn comes from the engine's power and efficiency, so a tank lasts ~3 hours instead of the
  real ~45-55 minutes.
- In BeamNG.drive, the steering wheel's direction and the driver camera still need a look.
- No LODs and no damage model.

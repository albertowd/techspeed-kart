"""Car definitions shared by the AC and BeamNG converters.

Engine torque is crank torque in Nm (AC power.lut gets it after chain losses).
Masses include an 80 kg driver for rentals and a 75 kg driver for racing.

Honda figures (net, as installed with air cleaner and muffler: the performance curves on
engines.honda.com, read 2000-3600 rpm; above 3600 they fall on at the curves' own rate):
  GX270 270 cc 77x58  6.3 kW (8.4 hp)  @ 3600   19.1 Nm @ 2500   (16.8 Nm @ 3600)
  GX340 337 cc 82x64  8.0 kW (10.7 hp) @ 3600   26.0 Nm @ 2500   (21.3 Nm @ 3600)
  GX390 389 cc 88x64  8.7 kW (11.7 hp) @ 3600   26.3 Nm @ 2500   (23.1 Nm @ 3600)
They were the gross figures (6.7 / 8.2 / 9.6 kW), ~1 / 0.4 / 2.4 Nm more at 3600 rpm than the engines give
fitted.
Rental karts have no gearbox: one chain ratio (24/65) behind a centrifugal clutch, so the top speed is the
speed governor's rpm on that ratio and the 11" rears (~0.0195 km/h per rpm). Governors per engine: 9hp
4000 rpm (~78 km/h; a new 9hp at Velopark ran a little over 75), 11hp 4100 (~80), 13hp 4200 (~82).

Racing 125: TaG-style 125 cc 2-stroke direct drive (~29 hp @ 11500 rpm),
CIK-style minimum weight 160 kg with driver. Placeholder until real data.
"""

RENTAL_COMMON = dict(
    variant="rental", source="rental",
    # centrifugal clutch: starts to bite at 2200, locks at 2800 rpm (~53 km/h on the 24/65 chain). Below
    # that it slips with the engine held near its torque peak, so slow corners taken flat out heat it up
    idle=1400, clutch_engage=(2200, 2800), clutch_lock=1.15,
    chain=(24, 65), chain_options=[(22, 65), (23, 65), (24, 65), (25, 65)],   # 24/65: governor at ~80 km/h
    engine_inertia=0.020, cg_height=0.27, front_weight=0.42,
    brake_torque=150, brake_front_share=0.0, fuel_l=4.0, fuel_max=6.0,
    tyre_grip=0.93,                    # harder rental compound
    steer_lock=100, steer_ratio=4.0, ffb=18,
)

CARS = {
    "techspeed_rental_9hp": dict(RENTAL_COMMON,
        name="Techspeed Rental v2 9hp", engine="Honda GX270", cc=270, mass=190, limiter=4000,
        torque=[(0, 0), (1400, 14.0), (2000, 18.6), (2500, 19.1), (3000, 18.6), (3600, 16.8),
                (4000, 15.2), (4300, 13.8), (4800, 10.3), (5200, 0)],
        # AC burns rpm * throttle * fuel_consumption / 1000 litres a second. At the track a rental runs two 25 min
        # heats on a tank: a lap at rental pace averages rpm * throttle ~2500 (0.074 l per 2.37 km lap of vp_2500
        # at 0.00019), so these give a full 6 l ~55 min (9hp), ~50 (11hp), ~45 (13hp), and the 4 l the karts start
        # with lasts a heat. (Honda's catalogue 2.5-3.5 L/h made a tank last over three hours.) The AI's fuel plan
        # comes from fuel_cons.ini, which export_ac works out from this.
        fuel_consumption=0.00070,
        bhp="9hp", top_kmh="~78"),
    "techspeed_rental_11hp": dict(RENTAL_COMMON,
        name="Techspeed Rental v2 11hp", engine="Honda GX340", cc=337, mass=195, limiter=4100,
        torque=[(0, 0), (1400, 19.0), (2000, 24.7), (2500, 26.0), (3000, 24.5), (3600, 21.3),
                (4000, 19.0), (4300, 17.0), (4800, 12.6), (5200, 0)],
        fuel_consumption=0.00078,                 # a full tank ~50 min
        bhp="11hp", top_kmh="~80"),
    "techspeed_rental_13hp": dict(RENTAL_COMMON,
        name="Techspeed Rental v2 13hp", engine="Honda GX390", cc=389, mass=196, limiter=4200,
        torque=[(0, 0), (1400, 20.0), (2000, 25.0), (2500, 26.3), (3000, 26.1), (3600, 23.1),
                (4000, 21.0), (4300, 19.2), (4800, 14.8), (5200, 0)],
        fuel_consumption=0.00085,                 # a full tank ~45 min
        bhp="13hp", top_kmh="~82"),
    "techspeed_racing_125": dict(
        variant="racing", source="racing",
        name="Techspeed Racing 125", engine="125cc 2-stroke TaG", cc=125, mass=160,
        torque=[(0, 0), (2500, 5.0), (4000, 7.5), (6000, 11.0), (8000, 15.5), (9500, 18.2),
                (10500, 19.0), (11500, 18.3), (12500, 15.5), (13500, 11.0), (14500, 6.0), (15500, 0)],
        idle=2800, limiter=15000, clutch_engage=(4500, 6500), clutch_lock=1.6,
        chain=(11, 80), chain_options=[(10, 80), (11, 80), (11, 76), (12, 80), (11, 84)],
        engine_inertia=0.004, cg_height=0.26, front_weight=0.42,
        brake_torque=200, brake_front_share=0.30, fuel_l=4.0, fuel_max=8.0,
        tyre_grip=1.0, fuel_consumption=0.0007,
        steer_lock=100, steer_ratio=4.0, ffb=20,
        bhp="~29hp", top_kmh="~110"),
}

# Rental v1 (kart_model.VARIANTS["rental_v1"]): the first protector, a band instead of the steel tubes, with no
# side pods or engine cover; the same engines. ~8 kg lighter: the two steel tube rings and their hoops weigh
# more than the band, pods and cover together.
V1_MASS_DELTA = -8
for _hp in ("9hp", "11hp", "13hp"):
    _v2 = CARS["techspeed_rental_" + _hp]
    CARS["techspeed_rental_v1_" + _hp] = dict(_v2, variant="rental_v1", source="rental_v1",
                                               name="Techspeed Rental v1 " + _hp, mass=_v2["mass"] + V1_MASS_DELTA)

# variants built and packed; the racing kart is parked for now (its definitions stay for later)
BUILD_VARIANTS = ("rental", "rental_v1")
ACTIVE_CARS = {cid: c for cid, c in CARS.items() if c["variant"] in BUILD_VARIANTS}


def release_build():
    """True for the release build (tools/build_all.py --release, run by the GitHub release workflow)."""
    import os
    return os.environ.get("TECHSPEED_RELEASE") == "1"


def built_skins(names):
    """The skins a build carries: the source keeps every livery of the fleet (textures.FLEET_SIZE), the release
    build ships them all, a local build only the first (fewer previews to render, smaller archives)."""
    names = sorted(names)
    return names if release_build() else names[:1]

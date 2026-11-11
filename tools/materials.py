"""Material table shared by the Blender preview, AC and BeamNG exporters.

texture: file name in the generated textures folder (None = flat colour).
skinnable: texture is overridden per skin (AC skins/<skin>/, BeamNG skin materials).
"""

MATERIALS = {
    # bodywork with the livery atlas
    "body_livery":      dict(color=(1, 1, 1), metal=0.0, rough=0.25, texture="body_livery.png", skinnable=True),
    "black_plastic":    dict(color=(0.025, 0.025, 0.028), metal=0.0, rough=0.55),
    "seat":             dict(color=(0.02, 0.02, 0.022), metal=0.0, rough=0.2),
    "fuel_hose":        dict(color=(0.80, 0.82, 0.78), metal=0.0, rough=0.15),   # clear fuel line
    "fuel_hose_yellow": dict(color=(0.80, 0.70, 0.20), metal=0.0, rough=0.2),    # aged fuel lines on the engine
    "tank_plastic":     dict(color=(0.85, 0.86, 0.84), metal=0.0, rough=0.45),
    "carbon":           dict(color=(1, 1, 1), metal=0.0, rough=0.2, texture="carbon.png"),
    # frame
    "chassis_gunmetal": dict(color=(0.16, 0.165, 0.17), metal=0.6, rough=0.45),
    "chassis_black":    dict(color=(0.03, 0.03, 0.03), metal=0.2, rough=0.45),
    "protector":        dict(color=(0.025, 0.025, 0.025), metal=0.1, rough=0.5),
    # metal parts
    "steel":            dict(color=(0.75, 0.75, 0.76), metal=1.0, rough=0.18),
    "aluminium":        dict(color=(0.80, 0.80, 0.82), metal=1.0, rough=0.3),
    "black_anodized":   dict(color=(0.03, 0.03, 0.035), metal=0.8, rough=0.35),
    "brake_disc":       dict(color=(0.12, 0.12, 0.12), metal=0.9, rough=0.4),
    "caliper":          dict(color=(0.02, 0.02, 0.02), metal=0.5, rough=0.3),
    "sprocket":         dict(color=(0.80, 0.58, 0.18), metal=1.0, rough=0.3),
    "chain":            dict(color=(0.30, 0.28, 0.25), metal=1.0, rough=0.5),
    "exhaust":          dict(color=(0.78, 0.78, 0.76), metal=1.0, rough=0.35),
    # wheels
    "tyre":             dict(color=(1, 1, 1), metal=0.0, rough=0.8, texture="tyre.png"),
    "rim_black":        dict(color=(0.03, 0.03, 0.035), metal=0.7, rough=0.35),
    "rim_alu":          dict(color=(0.86, 0.86, 0.88), metal=0.7, rough=0.32),   # satin polished aluminium: a mirror
                                                                                 # face only showed the dark protector
    # steering wheel
    "grip_black":       dict(color=(0.03, 0.03, 0.03), metal=0.0, rough=0.9),
    "grip_red":         dict(color=(0.65, 0.03, 0.03), metal=0.0, rough=0.8),
    "pod_stripe":       dict(color=(1, 1, 1), metal=0.0, rough=0.25, texture="pod_stripe.png"),
    "pedal_plate":      dict(color=(1, 1, 1), metal=0.3, rough=0.5, texture="pedal_plate.png"),
    "logo_badge":       dict(color=(1, 1, 1), metal=0.0, rough=0.3, texture="logo_badge.png"),
    # engines
    "engine_alu":       dict(color=(0.62, 0.63, 0.64), metal=0.9, rough=0.45),
    "engine_black":     dict(color=(0.05, 0.05, 0.05), metal=0.5, rough=0.5),
    "recoil_cover":     dict(color=(1, 1, 1), metal=0.0, rough=0.3, texture="recoil_cover.png"),
    "engine_red":       dict(color=(0.75, 0.02, 0.03), metal=0.0, rough=0.3),
    "air_filter":       dict(color=(0.03, 0.30, 0.18), metal=0.0, rough=0.9),   # green cone filter
    "oil_cap":          dict(color=(0.95, 0.72, 0.05), metal=0.0, rough=0.4),    # yellow oil filler caps
    "floor_alu":        dict(color=(0.74, 0.75, 0.77), metal=1.0, rough=0.38),   # plain bolted floor plate
    "floor_tread":      dict(color=(1, 1, 1), metal=0.4, rough=0.35, texture="tread_plate.png"),
    "radiator":         dict(color=(1, 1, 1), metal=0.6, rough=0.5, texture="radiator.png"),
    "radiator_shroud":  dict(color=(1, 1, 1), metal=0.0, rough=0.3, texture="radiator_shroud.png"),
}

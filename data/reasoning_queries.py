"""
data/reasoning_queries.py — Indirect / reasoning query bank per ice class.

Each query identifies an ice type by its PROPERTIES (formation, age, location,
texture, behaviour) and never by its name — analogous to asking to "segment the
fruit with the most vitamin C" instead of naming the orange. The segmentation
model must reason from world knowledge to decide whether the described ice is
present in the image and, if so, where.

Used by the reasoning-segmentation training mode
(config.DataConfig.reasoning_seg_mode). Keys must match config.ICE_CLASSES.
"""

REASONING_QUERIES = {
    "Young Ice": [
        "Segment the thin, recently frozen sea ice that is only days old and still developing.",
        "Find the newest, most fragile ice that has just begun to skin over the open water.",
        "Identify the sea ice in its earliest growth stage, before it has had time to thicken.",
        "Segment the smooth, dark, freshly formed ice that bends with the waves rather than breaking.",
        "Locate the youngest ice on the surface — the layer that froze most recently.",
        "Segment the delicate new ice that would not yet support any weight.",
    ],
    "First Year Ice": [
        "Segment the sea ice that froze during this single winter and has not yet survived a summer melt.",
        "Identify the ice that is at most one season old — thicker than fresh ice but not multi-season.",
        "Find the level sea ice formed within the current freezing season.",
        "Segment the seasonal ice that will likely melt away when summer returns.",
        "Locate the ice that grew this year and has never endured a melt season.",
        "Segment the moderately thick ice of a single winter's growth.",
    ],
    "Floating Ice": [
        "Segment the drifting ice that is not attached to land and moves with the current and wind.",
        "Identify the loose, mobile ice floating freely on the water surface.",
        "Find the unanchored ice that drifts rather than staying fastened to the shore.",
        "Segment the pack ice carried along by ocean currents.",
        "Locate the ice that is adrift and not fixed to any coastline.",
        "Segment the free-floating ice moving across the open water.",
    ],
    "Glaciers": [
        "Segment the thick land ice that formed from compressed snow over centuries and flows slowly downhill.",
        "Identify the heavily crevassed mass of ice that originates on land and creeps toward the sea.",
        "Find the ancient ice river that grinds slowly down a valley, fracturing as it moves.",
        "Segment the land-born ice tongue terminating where it meets open water at a calving front.",
        "Locate the ice that built up on land from accumulated snowfall and deforms under its own weight.",
        "Segment the slow-moving stream of land ice scarred by deep crevasses.",
    ],
    "Icebergs": [
        "Segment the large mass of freshwater ice that has calved from a glacier and now drifts in the open sea.",
        "Identify the towering chunk of ice that broke off land ice and floats with most of its bulk underwater.",
        "Find the isolated block of freshwater ice adrift in the ocean after breaking away from a glacier front.",
        "Segment the bright, blocky piece of ice that detached from a land glacier and floats out at sea.",
        "Locate the free-floating freshwater ice mass that originated on land and is now surrounded by water.",
        "Segment the calved ice island drifting in the open water.",
    ],
    "Old Ice": [
        "Segment the thick sea ice that has survived at least one full summer melt season.",
        "Identify the oldest, roughest sea ice that has persisted across multiple years.",
        "Find the weathered, ridged ice that has endured more than one melt and refreeze cycle.",
        "Segment the multi-season ice that is thicker and harder than any single winter's growth.",
        "Locate the long-lived sea ice that did not melt away over the summer.",
        "Segment the rugged, deformed ice that has lasted through several years.",
    ],
}


def all_classes():
    return list(REASONING_QUERIES.keys())

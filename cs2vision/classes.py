"""
CS2Vision detection classes, grouped by the BROADCAST INFORMATION CHANNEL they
belong to. The reframed research question is "how much authoritative state is
recoverable from the broadcast signal, and where are its limits", so classes
are organised by which surface of the broadcast carries the information.

  SPATIAL channel   -- the observer radar. Carries every player's position
                       (the broadcast already reconstructs this for you). This
                       is the primary object of study.
  STATUS  channel   -- HUD panels / floating tags. Carry health, armour,
                       economy. Recovered by OCR, scored against demo values.
  OCCLUSION channel -- utility (smoke) that removes information from the
                       WORLD view (not the radar -- see channel.py).
  EVENT   channel   -- kill feed, used for event-level sync.
  AUXILIARY         -- world-view player models. DE-EMPHASISED: the broadcast
                       already surfaces players via the radar and panels, so
                       world-view detection is a low-novelty enabler, not a
                       contribution. Kept optional for cross-checking only.
"""
CLASSES = [
    "blip_ct", "blip_t",                         # SPATIAL  (primary)
    "hud_health", "hud_armour", "hud_economy",   # STATUS   (OCR target)
    "smoke_radar", "smoke_active",               # OCCLUSION
    "killfeed",                                  # EVENT
    "player_ct", "player_t",                     # AUXILIARY (de-emphasised)
]
CLS_ID = {c: i for i, c in enumerate(CLASSES)}

CHANNELS = {
    "spatial":   ["blip_ct", "blip_t"],
    "status":    ["hud_health", "hud_armour", "hud_economy"],
    "occlusion": ["smoke_radar", "smoke_active"],
    "event":     ["killfeed"],
    "auxiliary": ["player_ct", "player_t"],
}
PRIMARY_CHANNELS = ["spatial", "status", "occlusion", "event"]
AUXILIARY_CLASSES = set(CHANNELS["auxiliary"])

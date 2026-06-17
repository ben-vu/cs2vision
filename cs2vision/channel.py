"""
Broadcast information-channel analysis (the reframed core contribution).

The research question is no longer "can we detect players" (the broadcast
already surfaces them) but "how much authoritative state is *in* the broadcast
signal, and what is the ceiling on recovering it". This module quantifies that
ceiling from the authoritative demo state + the radar geometry, independent of
any detector. It separates two fundamentally different quantities:

  * CHANNEL LIMIT  -- state that is not recoverable from the broadcast at all,
    no matter how good the method (e.g. two players closer than the radar can
    resolve collapse into one blip; sub-radar-pixel precision is simply not
    present in the signal).
  * METHOD ERROR   -- state that *is* in the signal but the pipeline missed.

A method can only ever reach the channel ceiling; reporting achieved accuracy
*as a fraction of the ceiling* is the honest measure of "broadcast-only
reconstruction", and the ceiling itself is a property of the broadcast medium
that is of independent scientific interest and transfers to any title whose
broadcast exposes a radar/minimap but no authoritative feed.

Note on smoke: on the production OBSERVER radar all alive players are shown
regardless of smoke, so smoke is an OCCLUSION limit on the *world-view* channel,
not the radar channel. The radar spatial channel is therefore smoke-robust --
a finding the broadcast framing tends to get wrong (and the old simulator
assumed otherwise).
"""
from __future__ import annotations

import numpy as np


def radar_precision_floor(rect, radar_meta, z: float = 0.0) -> dict:
    """Irreducible positional resolution from reading the on-screen radar.

    One on-screen radar-widget pixel spans `canonical_per_widget` canonical-map
    pixels = `world_per_widget` world units. The best any method can localise a
    blip is about half a widget pixel -- that is the channel's precision floor.
    """
    canonical_per_widget = rect.map_image_size / rect.w
    _, _, scale = radar_meta._params_for_z(z)
    world_per_widget = canonical_per_widget * scale
    return dict(
        canonical_per_widget_px=float(canonical_per_widget),
        world_per_widget_px=float(world_per_widget),
        floor_map_px=float(0.5 * canonical_per_widget),
        floor_world_units=float(0.5 * world_per_widget),
    )


def recoverability(gt_players, floor_map_px: float, assume_full_radar: bool = True):
    """Per-alive-player recoverability flags for one frame.

    position_in_signal : the observer radar carries this player's position
    separable          : no other alive player is within one radar pixel, so the
                         blip is uniquely resolvable (else channel-limited)
    status_in_signal   : the HUD panel carries this player's hp/economy
    """
    alive = [p for p in gt_players if p.get("alive", True) and "map_pos" in p]
    pos = np.array([p["map_pos"] for p in alive]) if alive else np.empty((0, 2))
    flags = []
    for i, p in enumerate(alive):
        separable = True
        if len(pos) > 1:
            d = np.linalg.norm(pos - pos[i], axis=1)
            d[i] = np.inf
            separable = bool(d.min() > 2 * floor_map_px)
        flags.append(dict(
            team=p.get("team"),
            position_in_signal=bool(assume_full_radar),
            separable=separable,
            status_in_signal=True,
            recoverable=bool(assume_full_radar and separable),
        ))
    return flags


def channel_ceiling(records, rect, radar_meta, assume_full_radar: bool = True) -> dict:
    """Aggregate the broadcast spatial-channel ceiling over a set of frames.

    `records` = assemble_clip output (each has 'players' with map_pos). Returns
    the ceiling fraction (upper bound on any method's positional recall) and the
    precision floor. Needs no detector -- it is a property of the channel.
    """
    floor = radar_precision_floor(rect, radar_meta)
    n_alive = n_recoverable = n_unresolvable = 0
    per_frame = []
    for rec in records:
        flags = recoverability(rec["players"], floor["floor_map_px"], assume_full_radar)
        a = len(flags)
        r = sum(f["recoverable"] for f in flags)
        n_alive += a
        n_recoverable += r
        n_unresolvable += sum(not f["separable"] for f in flags)
        per_frame.append(r / a if a else 1.0)
    return dict(
        precision_floor=floor,
        entities=n_alive,
        ceiling_fraction=float(n_recoverable / n_alive) if n_alive else 1.0,
        unresolvable_fraction=float(n_unresolvable / n_alive) if n_alive else 0.0,
        per_frame_ceiling=per_frame,
    )


def fraction_of_ceiling(method_fidelity: float, ceiling: dict) -> float:
    """Method recall expressed as a fraction of the channel ceiling -- the
    honest headline number for broadcast-only reconstruction."""
    c = ceiling["ceiling_fraction"]
    return float(method_fidelity / c) if c > 0 else float("nan")

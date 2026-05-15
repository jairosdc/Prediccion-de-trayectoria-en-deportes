"""
animation.py — Componente visual interactivo para cada deporte.
Usa tokens del Design System 'Apex Velocity' generado por Stitch MCP.
"""
import streamlit.components.v1 as components

# ============================================================
# Design tokens (Apex Velocity — Stitch MCP)
# ============================================================
_TOKENS = {
    "bg": "#131313",
    "glass": "rgba(6, 44, 18, 0.45)",
    "glass_border": "rgba(255,255,255,0.1)",
    "cyan": "#00dbe7",
    "lime": "#c3f400",
    "crimson": "#ff0055",
    "white": "#e5e2e1",
    "muted": "#8c938a",
    "font_head": "'Archivo Narrow', sans-serif",
    "font_body": "'Inter', sans-serif",
    "font_mono": "'JetBrains Mono', monospace",
}

# ============================================================
# Shared CSS base (injected in every animation)
# ============================================================
_BASE_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@700&family=Inter:wght@400;600&family=JetBrains+Mono:wght@600&display=swap');
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:transparent; overflow:hidden; font-family:{_TOKENS['font_body']}; color:{_TOKENS['white']}; }}
.glass {{ background:{_TOKENS['glass']}; backdrop-filter:blur(20px); border:1px solid {_TOKENS['glass_border']}; border-radius:12px; }}
.glow-cyan {{ box-shadow:0 0 14px {_TOKENS['cyan']}88; }}
.glow-lime {{ box-shadow:0 0 14px {_TOKENS['lime']}88; }}
.label {{ font-family:{_TOKENS['font_mono']}; font-size:11px; letter-spacing:0.12em; text-transform:uppercase; }}
.title {{ font-family:{_TOKENS['font_head']}; font-weight:700; }}
@keyframes pulse {{ 0%,100%{{opacity:.6}} 50%{{opacity:1}} }}
@keyframes shoot {{ 0%{{transform:translate(var(--sx),var(--sy)) scale(1);opacity:1}}
  100%{{transform:translate(var(--ex),var(--ey)) scale(.55);opacity:.85}} }}
.anim-shoot {{ animation:shoot .9s cubic-bezier(.22,1,.36,1) forwards; }}
@keyframes trail {{ 0%{{stroke-dashoffset:300;opacity:0}} 50%{{opacity:.9}} 100%{{stroke-dashoffset:0;opacity:.7}} }}
.anim-trail {{ stroke-dasharray:300; stroke-dashoffset:300; animation:trail .9s cubic-bezier(.22,1,.36,1) forwards; }}
"""


def _wrap(body_html: str, height: int = 420):
    """Wrap body content in a full HTML page and render via Streamlit."""
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
    <style>{_BASE_CSS}</style></head><body>{body_html}</body></html>"""
    components.html(html, height=height)


# ============================================================
# FOOTBALL  (left / center / right)
# ============================================================
def _football(zone: str):
    zones = {"left": 0, "center": 1, "right": 2}
    idx = zones.get(zone, 1)
    # Ball end positions  (css translate values)
    ends = [("-110px", "-130px"), ("0", "-140px"), ("110px", "-130px")]
    ex, ey = ends[idx]
    # SVG trail paths
    paths = ["M150,260 Q80,130 40,20", "M150,260 Q150,130 150,20", "M150,260 Q220,130 260,20"]
    zone_boxes = ""
    for i, lbl in enumerate(["LEFT", "CENTER", "RIGHT"]):
        active = "glow-cyan" if i == idx else ""
        bg = f"background:{_TOKENS['cyan']}22;" if i == idx else ""
        zone_boxes += f"""<div class="glass {active}" style="flex:1;display:flex;align-items:center;
            justify-content:center;{bg}border-radius:6px;margin:3px;">
            <span class="label" style="color:{_TOKENS['cyan'] if i==idx else _TOKENS['muted']}">{lbl}</span></div>"""

    _wrap(f"""
    <div class="glass" style="width:100%;max-width:520px;margin:auto;height:390px;position:relative;overflow:hidden;">
      <!-- Grass -->
      <div style="position:absolute;inset:0;background:linear-gradient(180deg,#062c12 0%,#0a421b 100%);opacity:.85;"></div>
      <div style="position:absolute;inset:0;background:repeating-linear-gradient(90deg,transparent,transparent 50px,rgba(255,255,255,.04) 50px,rgba(255,255,255,.04) 100px);"></div>
      <!-- Goal -->
      <div style="position:relative;z-index:2;width:80%;margin:24px auto 0;height:140px;border:5px solid rgba(255,255,255,.85);border-bottom:none;border-radius:8px 8px 0 0;display:flex;flex-direction:column;">
        <div style="position:absolute;inset:0;opacity:.15;background:radial-gradient(circle,#fff 1px,transparent 1px);background-size:12px 12px;"></div>
        <div style="display:flex;flex:1;padding:4px;">{zone_boxes}</div>
      </div>
      <!-- Trail SVG -->
      <svg style="position:absolute;bottom:40px;left:50%;transform:translateX(-50%);z-index:3;" width="300" height="280" viewBox="0 0 300 280">
        <defs><linearGradient id="tg" x1="0%" y1="100%" x2="0%" y2="0%">
          <stop offset="0%" stop-color="{_TOKENS['cyan']}" stop-opacity="0"/>
          <stop offset="100%" stop-color="{_TOKENS['cyan']}" stop-opacity="1"/></linearGradient></defs>
        <path d="{paths[idx]}" fill="none" stroke="url(#tg)" stroke-width="3" class="anim-trail"/>
      </svg>
      <!-- Ball -->
      <div class="anim-shoot" style="--sx:0;--sy:0;--ex:{ex};--ey:{ey};
        position:absolute;bottom:50px;left:50%;transform:translateX(-50%);z-index:4;
        width:28px;height:28px;border-radius:50%;background:#fff;
        box-shadow:0 0 18px rgba(255,255,255,.7);"></div>
      <!-- Label -->
      <div style="position:absolute;bottom:10px;width:100%;text-align:center;z-index:5;">
        <span class="title" style="font-size:13px;color:{_TOKENS['cyan']};">⚽ PENALTY KICK → {zone.upper()}</span></div>
    </div>""", 420)


# ============================================================
# TENNIS  (wide / body / center)
# ============================================================
def _tennis(zone: str):
    zones = {"wide": 0, "body": 1, "center": 2}
    idx = zones.get(zone, 1)
    ends = [("-120px", "-120px"), ("0", "-130px"), ("100px", "-120px")]
    ex, ey = ends[idx]
    colors = [_TOKENS['cyan'], _TOKENS['lime'], _TOKENS['cyan']]
    labels_list = ["WIDE", "BODY", "CENTER"]
    zone_boxes = ""
    for i, lbl in enumerate(labels_list):
        active = "glow-cyan" if i == idx else ""
        bg = f"background:{colors[i]}22;" if i == idx else ""
        zone_boxes += f"""<div class="glass {active}" style="flex:1;display:flex;align-items:center;
            justify-content:center;{bg}border-radius:6px;margin:3px;min-height:80px;">
            <span class="label" style="color:{colors[i] if i==idx else _TOKENS['muted']}">{lbl}</span></div>"""

    _wrap(f"""
    <div class="glass" style="width:100%;max-width:520px;margin:auto;height:390px;position:relative;overflow:hidden;">
      <!-- Court -->
      <div style="position:absolute;inset:0;background:linear-gradient(180deg,#1a4731 0%,#0d3520 100%);"></div>
      <div style="position:absolute;top:20px;left:10%;right:10%;bottom:60px;border:2px solid rgba(255,255,255,.3);border-radius:4px;"></div>
      <div style="position:absolute;top:20px;left:50%;width:2px;bottom:60px;background:rgba(255,255,255,.2);"></div>
      <!-- Service box zones -->
      <div style="position:relative;z-index:2;width:75%;margin:30px auto 0;display:flex;gap:4px;">{zone_boxes}</div>
      <!-- Ball trail -->
      <svg style="position:absolute;bottom:60px;left:50%;transform:translateX(-50%);z-index:3;" width="300" height="250" viewBox="0 0 300 250">
        <defs><linearGradient id="tgt" x1="0%" y1="100%" x2="0%" y2="0%">
          <stop offset="0%" stop-color="{_TOKENS['lime']}" stop-opacity="0"/>
          <stop offset="100%" stop-color="{_TOKENS['lime']}" stop-opacity="1"/></linearGradient></defs>
        <path d="{'M150,240 Q60,120 30,20' if idx==0 else 'M150,240 Q150,120 150,20' if idx==1 else 'M150,240 Q230,120 260,20'}"
          fill="none" stroke="url(#tgt)" stroke-width="3" class="anim-trail"/>
      </svg>
      <!-- Ball -->
      <div class="anim-shoot" style="--sx:0;--sy:0;--ex:{ex};--ey:{ey};
        position:absolute;bottom:70px;left:50%;transform:translateX(-50%);z-index:4;
        width:24px;height:24px;border-radius:50%;background:#ccff00;
        box-shadow:0 0 16px {_TOKENS['lime']}aa;"></div>
      <div style="position:absolute;bottom:10px;width:100%;text-align:center;z-index:5;">
        <span class="title" style="font-size:13px;color:{_TOKENS['lime']};">🎾 SERVE → {zone.upper()}</span></div>
    </div>""", 420)


# ============================================================
# BASKETBALL  (drive / mid_range / three_point)
# ============================================================
def _basketball(zone: str):
    zones = {"drive": 0, "mid_range": 1, "three_point": 2}
    idx = zones.get(zone, 1)
    ends = [("0", "-80px"), ("-60px", "-110px"), ("-100px", "-130px")]
    ex, ey = ends[idx]
    labels_list = ["DRIVE", "MID RANGE", "THREE PT"]
    colors_list = [_TOKENS['crimson'], _TOKENS['lime'], _TOKENS['cyan']]
    zone_boxes = ""
    for i, lbl in enumerate(labels_list):
        active = "glow-cyan" if i == idx else ""
        bg = f"background:{colors_list[i]}22;" if i == idx else ""
        zone_boxes += f"""<div class="glass {active}" style="flex:1;display:flex;align-items:center;
            justify-content:center;{bg}border-radius:6px;margin:3px;min-height:60px;">
            <span class="label" style="color:{colors_list[i] if i==idx else _TOKENS['muted']}">{lbl}</span></div>"""

    _wrap(f"""
    <div class="glass" style="width:100%;max-width:520px;margin:auto;height:390px;position:relative;overflow:hidden;">
      <div style="position:absolute;inset:0;background:linear-gradient(180deg,#2a1508 0%,#1a0d04 100%);"></div>
      <!-- Court lines -->
      <div style="position:absolute;top:30px;left:50%;transform:translateX(-50%);width:200px;height:100px;border:2px solid rgba(255,255,255,.2);border-radius:0 0 100px 100px;border-top:none;"></div>
      <div style="position:absolute;top:30px;left:50%;transform:translateX(-50%);width:60px;height:4px;background:rgba(255,255,255,.4);border-radius:2px;"></div>
      <!-- Zones -->
      <div style="position:relative;z-index:2;width:85%;margin:160px auto 0;display:flex;gap:4px;">{zone_boxes}</div>
      <!-- Ball -->
      <div class="anim-shoot" style="--sx:0;--sy:0;--ex:{ex};--ey:{ey};
        position:absolute;bottom:80px;left:50%;transform:translateX(-50%);z-index:4;
        width:26px;height:26px;border-radius:50%;background:#ff6b2b;
        box-shadow:0 0 16px rgba(255,107,43,.6);"></div>
      <div style="position:absolute;bottom:10px;width:100%;text-align:center;z-index:5;">
        <span class="title" style="font-size:13px;color:#ff6b2b;">🏀 SHOT → {zone.upper().replace('_',' ')}</span></div>
    </div>""", 420)


# ============================================================
# HANDBALL  (low_left / low_right / high_left / high_right / center)
# ============================================================
def _handball(zone: str):
    zones = {"high_left":0,"center":1,"high_right":2,"low_left":3,"low_right":4}
    idx = zones.get(zone, 1)
    ends = [("-110px","-130px"),("0","-130px"),("110px","-130px"),("-100px","-50px"),("100px","-50px")]
    ex, ey = ends[idx]
    labels_list = ["H-LEFT","CENTER","H-RIGHT","L-LEFT","L-RIGHT"]
    zone_boxes = ""
    for i, lbl in enumerate(labels_list):
        active = "glow-cyan" if i == idx else ""
        bg = f"background:{_TOKENS['cyan']}22;" if i == idx else ""
        w = "31%" if i < 3 else "48%"
        zone_boxes += f"""<div class="glass {active}" style="width:{w};display:flex;align-items:center;
            justify-content:center;{bg}border-radius:6px;margin:3px;min-height:55px;">
            <span class="label" style="color:{_TOKENS['cyan'] if i==idx else _TOKENS['muted']};font-size:10px;">{lbl}</span></div>"""

    _wrap(f"""
    <div class="glass" style="width:100%;max-width:520px;margin:auto;height:390px;position:relative;overflow:hidden;">
      <div style="position:absolute;inset:0;background:linear-gradient(180deg,#1a0a3a 0%,#0d0520 100%);"></div>
      <!-- Goal -->
      <div style="position:relative;z-index:2;width:80%;margin:24px auto 0;height:180px;border:5px solid rgba(255,255,255,.85);border-bottom:none;border-radius:8px 8px 0 0;">
        <div style="position:absolute;inset:0;opacity:.12;background:radial-gradient(circle,#fff 1px,transparent 1px);background-size:12px 12px;"></div>
        <div style="display:flex;flex-wrap:wrap;justify-content:center;padding:6px;height:100%;">
          {zone_boxes}
        </div>
      </div>
      <!-- Ball -->
      <div class="anim-shoot" style="--sx:0;--sy:0;--ex:{ex};--ey:{ey};
        position:absolute;bottom:60px;left:50%;transform:translateX(-50%);z-index:4;
        width:26px;height:26px;border-radius:50%;background:#fff;
        box-shadow:0 0 16px rgba(124,58,237,.6);border:2px solid #7c3aed;"></div>
      <div style="position:absolute;bottom:10px;width:100%;text-align:center;z-index:5;">
        <span class="title" style="font-size:13px;color:#7c3aed;">🤾 7M THROW → {zone.upper().replace('_',' ')}</span></div>
    </div>""", 420)


# ============================================================
# HOCKEY  (left_deke / right_deke / direct_shot / high_shot / low_shot)
# ============================================================
def _hockey(zone: str):
    zones = {"high_shot":0,"direct_shot":1,"left_deke":2,"right_deke":3,"low_shot":4}
    idx = zones.get(zone, 1)
    ends = [("0","-140px"),("0","-110px"),("-120px","-80px"),("120px","-80px"),("0","-40px")]
    ex, ey = ends[idx]
    labels_list = ["HIGH","DIRECT","L-DEKE","R-DEKE","LOW"]
    zone_boxes = ""
    for i, lbl in enumerate(labels_list):
        active = "glow-cyan" if i == idx else ""
        bg = f"background:{_TOKENS['cyan']}22;" if i == idx else ""
        w = "31%" if i < 2 else "48%" if i < 4 else "100%"
        zone_boxes += f"""<div class="glass {active}" style="width:{w};display:flex;align-items:center;
            justify-content:center;{bg}border-radius:6px;margin:3px;min-height:45px;">
            <span class="label" style="color:{_TOKENS['cyan'] if i==idx else _TOKENS['muted']};font-size:10px;">{lbl}</span></div>"""

    _wrap(f"""
    <div class="glass" style="width:100%;max-width:520px;margin:auto;height:390px;position:relative;overflow:hidden;">
      <div style="position:absolute;inset:0;background:linear-gradient(180deg,#0a2a3a 0%,#061820 100%);"></div>
      <div style="position:absolute;inset:0;opacity:.06;background:repeating-linear-gradient(0deg,transparent,transparent 30px,rgba(255,255,255,.1) 30px,rgba(255,255,255,.1) 31px);"></div>
      <!-- Goal -->
      <div style="position:relative;z-index:2;width:78%;margin:20px auto 0;height:190px;border:5px solid rgba(255,255,255,.85);border-bottom:none;border-radius:8px 8px 0 0;">
        <div style="position:absolute;inset:0;opacity:.1;background:radial-gradient(circle,#fff 1px,transparent 1px);background-size:14px 14px;"></div>
        <div style="display:flex;flex-wrap:wrap;justify-content:center;padding:6px;height:100%;">
          {zone_boxes}
        </div>
      </div>
      <!-- Puck -->
      <div class="anim-shoot" style="--sx:0;--sy:0;--ex:{ex};--ey:{ey};
        position:absolute;bottom:50px;left:50%;transform:translateX(-50%);z-index:4;
        width:28px;height:18px;border-radius:50%;background:#111;border:2px solid #555;
        box-shadow:0 0 14px rgba(0,219,231,.5);"></div>
      <div style="position:absolute;bottom:10px;width:100%;text-align:center;z-index:5;">
        <span class="title" style="font-size:13px;color:{_TOKENS['cyan']};">🏒 SHOOTOUT → {zone.upper().replace('_',' ')}</span></div>
    </div>""", 420)


# ============================================================
# ESPORTS  (aggressive_push / defensive_hold / flank / bait / direct_duel)
# ============================================================
def _esports(zone: str):
    zones = {"aggressive_push":0,"defensive_hold":1,"flank":2,"bait":3,"direct_duel":4}
    idx = zones.get(zone, 0)
    icons = ["⚔️","🛡️","↗️","🪤","🎯"]
    labels_list = ["PUSH","HOLD","FLANK","BAIT","DUEL"]
    colors_list = [_TOKENS['crimson'],_TOKENS['lime'],_TOKENS['cyan'],"#ff9900",_TOKENS['crimson']]

    items = ""
    for i, lbl in enumerate(labels_list):
        active = "glow-cyan" if i == idx else ""
        bg = f"background:{colors_list[i]}22;border:1px solid {colors_list[i]}88;" if i == idx else ""
        pulse = "animation:pulse 1.5s infinite;" if i == idx else ""
        items += f"""<div class="glass {active}" style="display:flex;flex-direction:column;align-items:center;
            justify-content:center;{bg}border-radius:8px;padding:12px 8px;min-width:80px;{pulse}">
            <span style="font-size:24px;">{icons[i]}</span>
            <span class="label" style="margin-top:6px;color:{colors_list[i] if i==idx else _TOKENS['muted']};font-size:10px;">{lbl}</span></div>"""

    _wrap(f"""
    <div class="glass" style="width:100%;max-width:540px;margin:auto;height:390px;position:relative;overflow:hidden;">
      <div style="position:absolute;inset:0;background:linear-gradient(135deg,#0a0015 0%,#150025 50%,#0a0015 100%);"></div>
      <div style="position:absolute;inset:0;opacity:.08;background:radial-gradient(circle at 50% 50%,{_TOKENS['cyan']}44,transparent 70%);"></div>
      <!-- HUD title -->
      <div style="position:relative;z-index:2;text-align:center;padding-top:20px;">
        <span class="title" style="font-size:20px;color:{_TOKENS['cyan']};">TACTICAL DECISION</span>
        <div class="label" style="color:{_TOKENS['muted']};margin-top:6px;">1v1 DUEL ANALYSIS</div>
      </div>
      <!-- Minimap -->
      <div style="position:relative;z-index:2;width:160px;height:160px;margin:16px auto;border:2px solid {_TOKENS['cyan']}44;border-radius:8px;overflow:hidden;">
        <div style="position:absolute;inset:0;background:linear-gradient(180deg,#0d1a0d 0%,#1a2a1a 100%);"></div>
        <div style="position:absolute;top:50%;left:50%;width:12px;height:12px;background:{colors_list[idx]};border-radius:50%;transform:translate(-50%,-50%);box-shadow:0 0 20px {colors_list[idx]};animation:pulse 1s infinite;"></div>
        <div style="position:absolute;inset:0;border:1px solid {_TOKENS['cyan']}22;"></div>
      </div>
      <!-- Decision buttons -->
      <div style="position:relative;z-index:2;display:flex;justify-content:center;gap:8px;padding:0 16px;flex-wrap:wrap;">
        {items}
      </div>
      <div style="position:absolute;bottom:10px;width:100%;text-align:center;z-index:5;">
        <span class="title" style="font-size:13px;color:{colors_list[idx]};">🎮 DECISION → {zone.upper().replace('_',' ')}</span></div>
    </div>""", 420)


# ============================================================
# Public dispatcher
# ============================================================
_RENDERERS = {
    "football": _football,
    "tennis": _tennis,
    "basketball": _basketball,
    "handball": _handball,
    "hockey": _hockey,
    "esports": _esports,
}


def render_sport_animation(sport: str, decision_zone: str):
    """Render the animation for any supported sport."""
    renderer = _RENDERERS.get(sport)
    if renderer:
        renderer(decision_zone)

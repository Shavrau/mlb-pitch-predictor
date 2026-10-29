"""
Bidirectional Streamlit Custom Components v2 (`st.components.v2`) for MLB Broadcast Telemetry.

Provides:
- `build_count_inspection_detail`: Synthesizes situational zone targeting, leverage, and scouting notes
  for an inspected `(balls, strikes)` count cell.
- `interactive_count_matrix_ccv2`: Clickable 4x3 MLB Count Matrix (`0-0` to `3-2`) built with CCv2
  that syncs `selected_count` back to Python without full-page reloads when wrapped in `@st.fragment`.
- `interactive_strike_zone_ccv2`: Clickable 13-zone Statcast Strike Zone (`1..9` Strike + `11..14` Chase)
  built with CCv2 that syncs `zone` back to Python.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from pitch_predictor.serving_client import (
    MLB_LEAGUE_PRIORS,
    PITCHER_AHEAD_COUNTS,
    PITCHER_BEHIND_COUNTS,
    normalize_priors,
)

VALID_COUNTS: tuple[str, ...] = (
    "0-0",
    "0-1",
    "0-2",
    "1-0",
    "1-1",
    "1-2",
    "2-0",
    "2-1",
    "2-2",
    "3-0",
    "3-1",
    "3-2",
)

VALID_STRIKE_ZONES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14)

CATEGORY_TELEMETRY_META: dict[str, dict[str, str]] = {
    "Fastball": {
        "code": "FF",
        "code_badge": "4-SEAM [FF]",
        "dark_hex": "#EF4444",
        "light_hex": "#DC2626",
        "shape": "▲",
    },
    "Breaking Ball": {
        "code": "BRK",
        "code_badge": "BREAKING [SL/ST]",
        "dark_hex": "#3B82F6",
        "light_hex": "#2563EB",
        "shape": "◆",
    },
    "Offspeed": {
        "code": "OFF",
        "code_badge": "OFFSPEED [CH/FS]",
        "dark_hex": "#10B981",
        "light_hex": "#059669",
        "shape": "●",
    },
}


def build_count_inspection_detail(
    count_key: str,
    predictions_by_count: dict[str, str],
    priors: dict[str, float],
    stand: str = "L",
    risp_active: bool = False,
) -> dict[str, Any]:
    """
    Build tactical scouting detail for a specific MLB count cell (`0-0` .. `3-2`).
    """
    clean_count = count_key if count_key in VALID_COUNTS else "0-0"
    predicted_cat = predictions_by_count.get(clean_count, "Fastball")
    if predicted_cat not in CATEGORY_TELEMETRY_META:
        predicted_cat = "Fastball"

    if clean_count in PITCHER_AHEAD_COUNTS:
        leverage = "Pitcher Ahead"
        leverage_pt = "Vantagem do Arremessador"
    elif clean_count in PITCHER_BEHIND_COUNTS:
        leverage = "Pitcher Behind"
        leverage_pt = "Pressão no Arremessador"
    else:
        leverage = "Neutral"
        leverage_pt = "Contagem Equilibrada"

    fb_p = float(priors.get("pitcher_fb_prior", MLB_LEAGUE_PRIORS[0]))
    bb_p = float(priors.get("pitcher_bb_prior", MLB_LEAGUE_PRIORS[1]))
    os_p = float(priors.get("pitcher_os_prior", MLB_LEAGUE_PRIORS[2]))
    fb_n, bb_n, os_n = normalize_priors(fb_p, bb_p, os_p)

    # Adjust baseline probabilities by situational count leverage for inspection clarity
    if leverage == "Pitcher Ahead":
        adj_fb, adj_bb, adj_os = normalize_priors(fb_n * 0.72, bb_n * 1.48, os_n * 1.25)
    elif leverage == "Pitcher Behind":
        adj_fb, adj_bb, adj_os = normalize_priors(fb_n * 1.42, bb_n * 0.65, os_n * 0.72)
    else:
        adj_fb, adj_bb, adj_os = (fb_n, bb_n, os_n)

    prob_map = {
        "Fastball": adj_fb,
        "Breaking Ball": adj_bb,
        "Offspeed": adj_os,
    }

    stand_clean = "L" if str(stand).upper() == "L" else "R"
    if predicted_cat == "Breaking Ball":
        if leverage == "Pitcher Ahead":
            rec_zones = ["Z14 (Chase Baixa-Dir)", "Z13 (Chase Baixa-Esq)", "Z9 (Borda Baixa)"]
        else:
            rec_zones = ["Z7 (Strike Baixa-Esq)", "Z9 (Strike Baixa-Dir)"]
    elif predicted_cat == "Offspeed":
        rec_zones = (
            ["Z13 (Chase Braço-Aberto)", "Z7 (Borda Inferior)", "Z8 (Base da Placa)"]
            if stand_clean == "L"
            else ["Z14 (Chase Braço-Aberto)", "Z9 (Borda Inferior)", "Z8 (Base da Placa)"]
        )
    else:
        if leverage == "Pitcher Behind":
            rec_zones = ["Z5 (Centro de Segurança)", "Z4 (Média-Esq)", "Z6 (Média-Dir)"]
        else:
            rec_zones = ["Z2 (Topo da Zona)", "Z1 (Alta-Esq)", "Z3 (Alta-Dir)"]

    risp_clause = (
        "Com RISP ativo, o plano prioriza evitar contato aéreo no centro da placa (Zona 5)."
        if risp_active
        else "Sem corredor em posição de anotar corrida, o arremessador ataca as bordas com agressividade."
    )

    if leverage == "Pitcher Ahead":
        tactical_note = (
            f"Em {clean_count} ({leverage_pt}), o modelo projeta {predicted_cat} buscando "
            f"expansão de zona e swing no vazio (whiff). {risp_clause}"
        )
    elif leverage == "Pitcher Behind":
        tactical_note = (
            f"Em {clean_count} ({leverage_pt}), o custo de ceder walk força retorno à zona de strike "
            f"com {predicted_cat} de maior comando. {risp_clause}"
        )
    else:
        tactical_note = (
            f"Em {clean_count} ({leverage_pt}), a chamada de {predicted_cat} estabelece "
            f"disfarce de trajetória desde o ponto de soltura contra rebatedor {stand_clean}HB. {risp_clause}"
        )

    meta = CATEGORY_TELEMETRY_META[predicted_cat]
    return {
        "count_key": clean_count,
        "predicted_category": predicted_cat,
        "code_badge": meta["code_badge"],
        "leverage": leverage,
        "leverage_pt": leverage_pt,
        "recommended_zones": rec_zones,
        "probabilities": prob_map,
        "tactical_note": tactical_note,
    }


# ---------------------------------------------------------------------------
# 1. Interactive 4x3 MLB Count Matrix (CCv2)
# ---------------------------------------------------------------------------

_COUNT_MATRIX_HTML = """
<div class="pp-matrix-wrap">
  <div id="pp-matrix-grid" class="pp-matrix-grid" role="grid" aria-label="Matriz Interativa de 12 Contagens da MLB"></div>
</div>
"""

_COUNT_MATRIX_CSS = """
.pp-matrix-wrap {
  font-family: var(--st-font, 'Space Grotesk', system-ui, sans-serif);
  color: var(--st-text-color, #F1F5F9);
  padding: 2px 0;
}
.pp-matrix-grid {
  display: grid;
  grid-template-columns: 92px repeat(3, minmax(0, 1fr));
  gap: 8px;
  align-items: stretch;
}
.pp-hdr {
  background: var(--st-secondary-background-color, #12141A);
  border: 1px solid var(--st-border-color, #222631);
  border-radius: 6px;
  padding: 7px 8px;
  text-align: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--st-text-color, #F1F5F9);
}
.pp-row-lbl {
  background: var(--st-secondary-background-color, #12141A);
  border: 1px solid var(--st-border-color, #222631);
  border-radius: 6px;
  padding: 8px 6px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  font-family: 'JetBrains Mono', monospace;
  font-variant-numeric: tabular-nums;
}
.pp-row-main {
  font-size: 12px;
  font-weight: 700;
  color: var(--st-text-color, #F1F5F9);
}
.pp-row-sub {
  font-size: 9.5px;
  opacity: 0.68;
  margin-top: 2px;
}
.pp-cell-btn {
  appearance: none;
  text-align: left;
  cursor: pointer;
  background: var(--st-secondary-background-color, #12141A);
  color: var(--st-text-color, #F1F5F9);
  border: 1px solid var(--st-border-color, #222631);
  border-left: 3px solid var(--cell-cat-color, #EF4444);
  border-radius: 6px;
  padding: 9px 11px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 5px;
  transition: border-color 0.14s ease, transform 0.12s ease;
}
.pp-cell-btn:hover {
  border-color: var(--cell-cat-color, #38BDF8);
  transform: translateY(-1px);
}
.pp-cell-btn:focus-visible {
  outline: 2px solid var(--accent-hex, #38BDF8);
  outline-offset: 2px;
}
.pp-cell-btn[aria-selected="true"] {
  border: 2px solid var(--accent-hex, #38BDF8);
  border-left: 4px solid var(--cell-cat-color, #EF4444);
  background: color-mix(in oklab, var(--st-secondary-background-color, #12141A) 88%, var(--cell-cat-color, #38BDF8) 12%);
}
.pp-cell-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 6px;
}
.pp-count-pill {
  font-family: 'JetBrains Mono', monospace;
  font-variant-numeric: tabular-nums;
  font-size: 12.5px;
  font-weight: 800;
  padding: 1px 7px;
  border-radius: 4px;
  border: 1px solid var(--st-border-color, #222631);
  background: var(--st-background-color, #0A0B0E);
}
.pp-lev-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  opacity: 0.82;
}
.pp-cat-line {
  font-size: 13.5px;
  font-weight: 700;
  letter-spacing: 0.01em;
}
.pp-code-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  color: var(--cell-cat-color, #EF4444);
  letter-spacing: 0.04em;
}
@media (forced-colors: active) {
  .pp-cell-btn[aria-selected="true"] {
    outline: 2px solid Highlight;
  }
}
"""

_COUNT_MATRIX_JS = """
export default function (component) {
  const { data, parentElement, setStateValue } = component;
  const grid = parentElement.querySelector("#pp-matrix-grid");
  if (!grid || !data) return () => {};

  const cells = Array.isArray(data.cells) ? data.cells : [];
  const selected = data.selected_count || "0-0";
  const accentHex = data.accent_hex || "#38BDF8";
  grid.style.setProperty("--accent-hex", accentHex);

  grid.innerHTML = "";

  const headers = ["B \\\\ S", "0 STRIKES", "1 STRIKE", "2 STRIKES"];
  headers.forEach((hText) => {
    const h = document.createElement("div");
    h.className = "pp-hdr";
    h.textContent = hText;
    grid.appendChild(h);
  });

  for (let b = 0; b < 4; b++) {
    const rowLbl = document.createElement("div");
    rowLbl.className = "pp-row-lbl";
    const mainSpan = document.createElement("span");
    mainSpan.className = "pp-row-main";
    mainSpan.textContent = b === 1 ? "1 BALL" : `${b} BALLS`;
    const subSpan = document.createElement("span");
    subSpan.className = "pp-row-sub";
    subSpan.textContent = `LINHA ${b}-X`;
    rowLbl.appendChild(mainSpan);
    rowLbl.appendChild(subSpan);
    grid.appendChild(rowLbl);

    for (let s = 0; s < 3; s++) {
      const cKey = `${b}-${s}`;
      const cellMeta = cells.find((c) => c.count_key === cKey) || {
        count_key: cKey,
        category: "Fastball",
        code_badge: "4-SEAM [FF]",
        shape: "▲",
        color: "#EF4444",
        leverage_short: "EVEN",
      };

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pp-cell-btn";
      btn.setAttribute("role", "gridcell");
      btn.setAttribute("aria-selected", cKey === selected ? "true" : "false");
      btn.setAttribute("aria-label", `Contagem ${cKey}: ${cellMeta.category} (${cellMeta.code_badge})`);
      btn.style.setProperty("--cell-cat-color", cellMeta.color);

      const topRow = document.createElement("div");
      topRow.className = "pp-cell-top";
      const pill = document.createElement("span");
      pill.className = "pp-count-pill";
      pill.textContent = cKey;
      const lev = document.createElement("span");
      lev.className = "pp-lev-tag";
      lev.textContent = cellMeta.leverage_short;
      topRow.appendChild(pill);
      topRow.appendChild(lev);

      const catLine = document.createElement("div");
      catLine.className = "pp-cat-line";
      catLine.textContent = `${cellMeta.shape} ${cellMeta.category}`;

      const badgeLine = document.createElement("div");
      badgeLine.className = "pp-code-badge";
      badgeLine.textContent = cellMeta.code_badge;

      btn.appendChild(topRow);
      btn.appendChild(catLine);
      btn.appendChild(badgeLine);

      btn.onclick = () => {
        setStateValue("selected_count", cKey);
      };

      grid.appendChild(btn);
    }
  }

  return () => {
    grid.innerHTML = "";
  };
}
"""

_COUNT_MATRIX_CCV2 = st.components.v2.component(
    "pp_interactive_count_matrix_v2",
    html=_COUNT_MATRIX_HTML,
    css=_COUNT_MATRIX_CSS,
    js=_COUNT_MATRIX_JS,
)


def interactive_count_matrix_ccv2(
    predictions_by_count: dict[str, str],
    *,
    initial_count: str = "0-0",
    accent_hex: str = "#38BDF8",
    theme_mode: str = "dark",
    key: str = "ccv2_count_matrix",
    on_selected_count_change: Callable[[], None] | None = None,
) -> str:
    """
    Mount the bidirectional 4x3 MLB Count Matrix Custom Component v2 and return the active count (`"0-0"`..`"3-2"`).
    """
    safe_initial = initial_count if initial_count in VALID_COUNTS else "0-0"
    is_light = str(theme_mode).lower().startswith("light")

    cells_payload: list[dict[str, str]] = []
    for c_key in VALID_COUNTS:
        cat = predictions_by_count.get(c_key, "Fastball")
        meta = CATEGORY_TELEMETRY_META.get(cat, CATEGORY_TELEMETRY_META["Fastball"])
        if c_key in PITCHER_AHEAD_COUNTS:
            lev_short = "AHEAD"
        elif c_key in PITCHER_BEHIND_COUNTS:
            lev_short = "BEHIND"
        else:
            lev_short = "EVEN"
        cells_payload.append(
            {
                "count_key": c_key,
                "category": cat,
                "code_badge": meta["code_badge"],
                "shape": meta["shape"],
                "color": meta["light_hex"] if is_light else meta["dark_hex"],
                "leverage_short": lev_short,
            }
        )

    existing_state = st.session_state.get(key)
    current_selected = safe_initial
    if isinstance(existing_state, dict) and existing_state.get("selected_count") in VALID_COUNTS:
        current_selected = str(existing_state["selected_count"])
    elif hasattr(existing_state, "selected_count") and existing_state.selected_count in VALID_COUNTS:
        current_selected = str(existing_state.selected_count)

    cb = on_selected_count_change if on_selected_count_change is not None else (lambda: None)
    result = _COUNT_MATRIX_CCV2(
        data={
            "cells": cells_payload,
            "selected_count": current_selected,
            "accent_hex": accent_hex,
        },
        default={"selected_count": safe_initial},
        key=key,
        on_selected_count_change=cb,
    )
    val = getattr(result, "selected_count", None)
    return str(val) if val in VALID_COUNTS else current_selected


# ---------------------------------------------------------------------------
# 2. Interactive 13-Zone Statcast Strike Zone (CCv2)
# ---------------------------------------------------------------------------

_STRIKE_ZONE_HTML = """
<div class="pp-sz-wrap">
  <div class="pp-sz-header">
    <span class="pp-sz-title">STATCAST TARGETING GRID (13 ZONAS)</span>
    <span id="pp-sz-readout" class="pp-sz-readout">Z5 • STRIKE</span>
  </div>
  <svg id="pp-sz-svg" viewBox="0 0 240 232" class="pp-sz-svg" role="listbox" aria-label="Zona de Strike Interativa Statcast"></svg>
</div>
"""

_STRIKE_ZONE_CSS = """
.pp-sz-wrap {
  font-family: var(--st-font, 'Space Grotesk', system-ui, sans-serif);
  color: var(--st-text-color, #F1F5F9);
  background: var(--st-secondary-background-color, #12141A);
  border: 1px solid var(--st-border-color, #222631);
  border-radius: 8px;
  padding: 8px 10px;
}
.pp-sz-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.04em;
}
.pp-sz-title {
  opacity: 0.78;
}
.pp-sz-readout {
  padding: 1px 7px;
  border-radius: 4px;
  background: var(--st-background-color, #0A0B0E);
  border: 1px solid var(--st-border-color, #222631);
  font-variant-numeric: tabular-nums;
}
.pp-sz-svg {
  width: 100%;
  max-width: 245px;
  margin: 0 auto;
  display: block;
}
.pp-zone-hit {
  cursor: pointer;
  transition: fill-opacity 0.12s ease, stroke-width 0.12s ease;
}
.pp-zone-hit:hover {
  fill-opacity: 0.42;
}
.pp-zone-hit:focus-visible {
  outline: 2px solid #F59E0B;
}
"""

_STRIKE_ZONE_JS = """
export default function (component) {
  const { data, parentElement, setStateValue } = component;
  const svg = parentElement.querySelector("#pp-sz-svg");
  const readout = parentElement.querySelector("#pp-sz-readout");
  if (!svg || !data) return () => {};

  const activeZone = Number(data.selected_zone || 5);
  const catColor = data.pitch_color || "#EF4444";
  const accentHex = data.accent_hex || "#38BDF8";
  const NS = "http://www.w3.org/2000/svg";

  svg.innerHTML = "";

  const chaseRects = [
    { id: 11, x: 18, y: 10, w: 102, h: 96, tx: 35, ty: 28, region: "CHASE ALTA-ESQ" },
    { id: 12, x: 120, y: 10, w: 102, h: 96, tx: 205, ty: 28, region: "CHASE ALTA-DIR" },
    { id: 13, x: 18, y: 106, w: 102, h: 96, tx: 35, ty: 192, region: "CHASE BAIXA-ESQ" },
    { id: 14, x: 120, y: 106, w: 102, h: 96, tx: 205, ty: 192, region: "CHASE BAIXA-DIR" },
  ];

  const zoneCenters = {
    11: [36, 28],
    12: [204, 28],
    13: [36, 188],
    14: [204, 188],
  };

  function bindZone(el, zId) {
    el.setAttribute("tabindex", "0");
    el.setAttribute("role", "option");
    el.setAttribute("aria-selected", zId === activeZone ? "true" : "false");
    el.onclick = () => setStateValue("zone", zId);
    el.onkeydown = (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        setStateValue("zone", zId);
      }
    };
  }

  chaseRects.forEach((cz) => {
    const isSel = cz.id === activeZone;
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "pp-zone-hit");
    bindZone(g, cz.id);

    const r = document.createElementNS(NS, "rect");
    r.setAttribute("x", String(cz.x));
    r.setAttribute("y", String(cz.y));
    r.setAttribute("width", String(cz.w));
    r.setAttribute("height", String(cz.h));
    r.setAttribute("rx", "5");
    r.setAttribute("fill", isSel ? catColor : "#334155");
    r.setAttribute("fill-opacity", isSel ? "0.48" : "0.14");
    r.setAttribute("stroke", isSel ? catColor : "#64748B");
    r.setAttribute("stroke-width", isSel ? "2" : "1.2");
    r.setAttribute("stroke-dasharray", "4,3");

    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", String(cz.tx));
    t.setAttribute("y", String(cz.ty));
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("fill", "currentColor");
    t.setAttribute("font-size", "10.5");
    t.setAttribute("font-family", "JetBrains Mono, monospace");
    t.setAttribute("font-weight", "700");
    t.textContent = `Z${cz.id}`;

    g.appendChild(r);
    g.appendChild(t);
    svg.appendChild(g);
  });

  for (let idx = 0; idx < 9; idx++) {
    const zId = idx + 1;
    const row = Math.floor(idx / 3);
    const col = idx % 3;
    const x = 52 + col * 45;
    const y = 38 + row * 45;
    zoneCenters[zId] = [x + 22.5, y + 22.5];
    const isSel = zId === activeZone;

    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "pp-zone-hit");
    bindZone(g, zId);

    const r = document.createElementNS(NS, "rect");
    r.setAttribute("x", String(x));
    r.setAttribute("y", String(y));
    r.setAttribute("width", "45");
    r.setAttribute("height", "45");
    r.setAttribute("fill", isSel ? catColor : "#0F172A");
    r.setAttribute("fill-opacity", isSel ? "0.72" : "0.42");
    r.setAttribute("stroke", isSel ? "#FFFFFF" : accentHex);
    r.setAttribute("stroke-width", isSel ? "2" : "1.2");

    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", String(x + 22.5));
    t.setAttribute("y", String(y + 26.5));
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("fill", isSel ? "#FFFFFF" : "currentColor");
    t.setAttribute("font-size", "12");
    t.setAttribute("font-family", "JetBrains Mono, monospace");
    t.setAttribute("font-weight", isSel ? "800" : "600");
    t.textContent = String(zId);

    g.appendChild(r);
    g.appendChild(t);
    svg.appendChild(g);
  }

  const borderBox = document.createElementNS(NS, "rect");
  borderBox.setAttribute("x", "52");
  borderBox.setAttribute("y", "38");
  borderBox.setAttribute("width", "135");
  borderBox.setAttribute("height", "135");
  borderBox.setAttribute("fill", "none");
  borderBox.setAttribute("stroke", accentHex);
  borderBox.setAttribute("stroke-width", "2.2");
  borderBox.setAttribute("pointer-events", "none");
  svg.appendChild(borderBox);

  if (zoneCenters[activeZone]) {
    const [cx, cy] = zoneCenters[activeZone];
    const ring = document.createElementNS(NS, "circle");
    ring.setAttribute("cx", String(cx));
    ring.setAttribute("cy", String(cy));
    ring.setAttribute("r", "12");
    ring.setAttribute("fill", "none");
    ring.setAttribute("stroke", "#F59E0B");
    ring.setAttribute("stroke-width", "2");
    ring.setAttribute("stroke-dasharray", "3,2");
    ring.setAttribute("pointer-events", "none");
    svg.appendChild(ring);
  }

  const plate = document.createElementNS(NS, "polygon");
  plate.setAttribute("points", "86,210 154,210 142,220 120,227 98,220");
  plate.setAttribute("fill", "#E2E8F0");
  plate.setAttribute("stroke", "#64748B");
  plate.setAttribute("stroke-width", "1.4");
  svg.appendChild(plate);

  if (readout) {
    const reg = activeZone >= 11 ? "CHASE / SHADOW" : (activeZone === 5 ? "HEART STRIKE" : "STRIKE ZONE");
    readout.textContent = `Z${activeZone} • ${reg}`;
    readout.style.borderColor = catColor;
  }

  return () => {
    svg.innerHTML = "";
  };
}
"""

_STRIKE_ZONE_CCV2 = st.components.v2.component(
    "pp_interactive_strike_zone_v2",
    html=_STRIKE_ZONE_HTML,
    css=_STRIKE_ZONE_CSS,
    js=_STRIKE_ZONE_JS,
)


def interactive_strike_zone_ccv2(
    selected_zone: int = 5,
    *,
    pitch_category: str = "Fastball",
    accent_hex: str = "#38BDF8",
    theme_mode: str = "dark",
    key: str = "ccv2_strike_zone",
    on_zone_change: Callable[[], None] | None = None,
) -> int:
    """
    Mount the bidirectional 13-zone Statcast Strike Zone Custom Component v2 and return the active zone (`1..9` or `11..14`).
    """
    safe_zone = int(selected_zone) if int(selected_zone) in VALID_STRIKE_ZONES else 5
    is_light = str(theme_mode).lower().startswith("light")
    meta = CATEGORY_TELEMETRY_META.get(pitch_category, CATEGORY_TELEMETRY_META["Fastball"])
    pitch_color = meta["light_hex"] if is_light else meta["dark_hex"]

    existing_state = st.session_state.get(key)
    current_zone = safe_zone
    if isinstance(existing_state, dict) and existing_state.get("zone") in VALID_STRIKE_ZONES:
        current_zone = int(existing_state["zone"])
    elif hasattr(existing_state, "zone") and existing_state.zone in VALID_STRIKE_ZONES:
        current_zone = int(existing_state.zone)

    cb = on_zone_change if on_zone_change is not None else (lambda: None)
    result = _STRIKE_ZONE_CCV2(
        data={
            "selected_zone": current_zone,
            "pitch_color": pitch_color,
            "accent_hex": accent_hex,
        },
        default={"zone": safe_zone},
        key=key,
        on_zone_change=cb,
    )
    val = getattr(result, "zone", None)
    try:
        parsed = int(val) if val is not None else current_zone
    except (TypeError, ValueError):
        parsed = current_zone
    return parsed if parsed in VALID_STRIKE_ZONES else current_zone

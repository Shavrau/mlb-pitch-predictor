---
name: frontend-design
description: Design system v2.0 (Broadcast Editorial & Anti-AI-Slop Dual-Mode) for MLB sports analytics, tactical dashboards, and console broadcast experiences. Covers Apple TV+ Friday Night Baseball & Baseball Savant aesthetics, Dual Mode (Dark Broadcast default + Light Editorial), dynamic MLB franchise accents, tabular typography, precision SVG construction, and Streamlit Custom Components v2 (st.components.v2).
license: Apache-2.0
allowed-tools: Read Write Edit Bash
metadata:
  version: "2.0"
  last-reviewed: "2026-10-29"
  skill-author: Antigravity Agentic Team
---

# Frontend Design System v2.0: Broadcast Editorial & Anti-AI-Slop Dual-Mode

## 1. Design Philosophy: Apple TV+ Broadcast × Baseball Savant

This design system eradicates generic "AI dashboard" visual traits (emoji-stuffed headers, cyan/magenta neon drop-shadows, heavy gradients, thick `5px` left borders, and nested boxes) in favor of an authentic **MLB Broadcast & Scouting Dossier** aesthetic:

1. **Zero Decorative Emojis in Telemetry & Analytical UI**: Never place emojis in page titles, metric headers, dropdown labels, table cells, or technical badges. Use official MLB Statcast codes (`4-SEAM [FF]`, `SWEEPER [ST]`, `SPLITTER [FS]`, `RHP`, `LHP`, `RISP`), clean geometric SVG indicators, and Streamlit Material Symbols (`:material/icon_name:`).
2. **Matte Carbon & Crisp Technical Paper (Dual Mode, Default Dark Broadcast)**:
   - **Dark Broadcast (Default)**: Matte carbon surfaces (`#0A0B0E` canvas, `#12141A` cards) with hairline `1px` borders (`#222631`) and zero diffuse neon `box-shadow`.
   - **Light Editorial (*Baseball Savant*)**: Cool technical paper (`#F8FAFC` canvas, `#FFFFFF` cards) with crisp `#CBD5E1` `1px` borders and deep slate ink (`#0F172A`).
   - All HTML, SVG, and CCv2 components MUST consume adaptive CSS custom properties (`--pp-bg`, `--pp-surface`, `--pp-surface-alt`, `--pp-border`, `--pp-text`, `--pp-muted`, backed by Streamlit's `--st-*` tokens) so toggling between Dark Broadcast and Light Editorial maintains WCAG AA contrast everywhere.
3. **Dynamic MLB Franchise Chromatic Identity**: Instead of generic cyan/purple accents, active athlete headers and telemetry strips adopt the official MLB franchise primary color of the selected pitcher (`#FDB827` Pirates Gold, `#DF4601` Orioles Orange, `#FD5A1E` Giants Orange, `#CE1141` Braves Scarlet, `#005A9C` Dodgers Blue).

---

## 2. Dual-Mode Semantic Token System

| CSS Variable | Dark Broadcast (Default) | Light Editorial (Savant) | Role |
|---|---|---|---|
| `--pp-bg` | `#0A0B0E` | `#F8FAFC` | Primary page/canvas background |
| `--pp-surface` | `#12141A` | `#FFFFFF` | Card, scorebug, and telemetry container |
| `--pp-surface-alt` | `#181B23` | `#F1F5F9` | Secondary strip, table header, subtle inset |
| `--pp-border` | `#222631` | `#CBD5E1` | Hairline `1px` structural border |
| `--pp-text` | `#F1F5F9` | `#0F172A` | Primary high-contrast ink |
| `--pp-muted` | `#94A3B8` | `#475569` | Secondary telemetry labels & captions |
| `--pitch-fastball` | `#EF4444` | `#DC2626` | Fastball family (`FF`, `SI`, `FC`) |
| `--pitch-breaking` | `#3B82F6` | `#2563EB` | Breaking Ball family (`SL`, `ST`, `CU`, `KC`) |
| `--pitch-offspeed` | `#10B981` | `#059669` | Offspeed family (`CH`, `FS`) |
| `--base-occupied` | `#F59E0B` | `#D97706` | Occupied base / high-leverage RISP indicator |

---

## 3. Typography & Micro-Hierarchy

1. **Display & Broadcast Headings (`Barlow Condensed`)**: Semi-condensed, uppercase or sentence-case with tight tracking (`0.03em–0.06em`), `font-weight: 700`.
2. **UI Body (`Space Grotesk`)**: Clean geometric grotesque for tactical prose, scouting notes, and controls.
3. **Telemetry & Tabular Data (`JetBrains Mono`)**: Every velocity (`101.8 MPH`), spin rate (`2,880 RPM`), angle (`52.8°`), count (`0-2`), latency (`42.1 ms`), and probability (`68.4%`) MUST apply `font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums;` so columns align to the pixel.

---

## 4. Streamlit 1.66+ & Custom Components v2 Architecture

1. **Native Theme First (`.streamlit/config.toml`)**: Define `[theme]`, `[theme.dark]`, `[theme.light]`, `[theme.dark.sidebar]`, and `[theme.light.sidebar]` in `.streamlit/config.toml`.
2. **Modern Layout Primitives**:
   - Use `width="stretch"` or `width="content"` (never deprecated `use_container_width`).
   - Isolate interactive inference blocks inside `@st.fragment` to prevent full-page flicker.
   - Never nest `st.columns()` inside another column beyond 1 level.
   - Avoid `st.divider()` clutter; rely on clean `1px` surface grouping and `st.space("small")`.
3. **Interactive Bidirectional SVG/HTML via `st.components.v2`**:
   - Register components once at module import level via `st.components.v2.component(...)` (never legacy `st.components.v1`).
   - Use `setStateValue()` / `setTriggerValue()` in ES module default exports (`export default function(component)`) and style inside Shadow DOM with `var(--st-*)` tokens.
   - Maintain deterministic `st.session_state` synchronization and XML-valid `<svg>` markup (`xml.etree.ElementTree` safe, single-line `_compact_html` when passed to `st.markdown`).

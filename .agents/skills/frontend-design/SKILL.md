---
name: frontend-design
description: Design system, UI/UX patterns, and visual styling guidelines for modern sports analytics and tactical dashboards. Covers color theory, typography, SVG graphic construction (baseball diamond, Statcast strike zone), dark-mode sports broadcast aesthetic (MLB/Statcast), visual hierarchy, and micro-interactions.
license: Apache-2.0
allowed-tools: Read Write Edit Bash
metadata:
  version: "1.0"
  last-reviewed: "2026-10-17"
  skill-author: Antigravity Agentic Team
---

# Frontend Design System: Sports Analytics & Tactical Dashboards

## 1. Design Philosophy: The Statcast Broadcast Aesthetic
The interface draws inspiration from modern MLB broadcast overlays (Apple TV+, ESPN Sunday Night Baseball, Statcast 3D). It combines a dark, distraction-free stadium background with high-contrast tactical elements, neon accents, and clean geometric data visualizations.

---

## 2. Color Palette & Tactical Semantics

| Token | Hex | Role | Semantic Meaning |
|---|---|---|---|
| `--bg-main` | `#0B0F19` | Fundo principal | Ambiente escuro estilo sala de controle/estádio |
| `--surface-card` | `#151E2E` | Cartões e painéis | Elevação e agrupamento de controles |
| `--surface-border`| `#26354A` | Bordas e divisores | Separação visual com baixo contraste |
| `--pitch-fastball`| `#EF4444` | Destaque Bola Rápida | Velocidade, calor, energia (Four-Seam, Sinker, Cutter) |
| `--pitch-breaking`| `#3B82F6` | Destaque Curva/Slide | Efeito, rotação, desvio de trajetória (Slider, Curve, Sweeper) |
| `--pitch-offspeed`| `#10B981` | Destaque Mudança | Desaceleração, quebra de tempo (Changeup, Splitter) |
| `--base-occupied` | `#F59E0B` | Corredor em base | Alerta de pressão no montinho (âmbar vivo com glow) |
| `--base-empty`    | `#334155` | Base vazia | Estado passivo com borda suave |
| `--strike-zone`   | `#38BDF8` | Strike Zone Box | Zona oficial de strike (borda ciano nítida) |
| `--ball-zone`     | `#64748B` | Chase / Waste Zone | Zonas periféricas (11 a 14) fora da zona de strike |

---

## 3. SVG Component Construction

### 3.1. Baseball Diamond Component
A geometric representation of the infield:
```
           [ 2B ] (x=100, y=25)
           /    \
 [ 3B ]   /      \   [ 1B ]
(x=25,   /        \ (x=175, y=100)
 y=100)  \        /
          \      /
           [ HP ] (x=100, y=175)
```
- **Base Paths**: Linhas conectando Home Plate -> 1B -> 2B -> 3B -> Home Plate com stroke `#334155` tracejado ou sólido.
- **Bases (1B, 2B, 3B)**: Quadrados rotacionados em 45 graus (`transform="rotate(45, cx, cy)"`).
- **Home Plate**: Polígono de 5 pontas apontando para baixo.
- **Mound (Montinho)**: Círculo sutil centralizado no centro do diamante (`cx=100, cy=100`).

### 3.2. Statcast Strike Zone Matrix
Representação da perspectiva do receptor (Catcher's perspective):
- **Zonas 1 a 9**: Grade 3x3 dividida em células proporcionais de 50x50 pixels.
  - Linha 1 (High): Zonas 1 (esq), 2 (centro), 3 (dir)
  - Linha 2 (Mid): Zonas 4 (esq), 5 (coração da zona), 6 (dir)
  - Linha 3 (Low): Zonas 7 (esq), 8 (centro), 9 (dir)
- **Zonas 11 a 14 (Waste / Chase / Shadow)**:
  - 11: Superior Esquerdo (Shadow Alto/Esq)
  - 12: Superior Direito (Shadow Alto/Dir)
  - 13: Inferior Esquerdo (Shadow Baixo/Esq)
  - 14: Inferior Direito (Shadow Baixo/Dir)
- **Feedback Interativo**: Zona selecionada recebe preenchimento vibrante, sombra e indicador numérico central.

---

## 4. Tipografia e Micro-Hierarquia

1. **Cabeçalhos Principais**: Fonte sem serifa pesada (`font-weight: 700`), caixa alta com tracking suave (`letter-spacing: 0.05em`).
2. **Badges de Predição**:
   - Cantos arredondados (`border-radius: 8px`).
   - Sombra projetada colorida (`box-shadow: 0 4px 14px rgba(..., 0.35)`).
   - Ícone identificador visual de alto impacto (⚡, 🌀, 🛑).
3. **Métricas Rápidas**: Placar e contagem exibidos em caixas digitais simulando o placar eletrônico do estádio (*scoreboard*).

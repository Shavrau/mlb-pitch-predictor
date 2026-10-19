---
name: game-design
description: Design system, UI patterns, and gameplay mechanics for browser-based arcade, retro 8-bit/16-bit, and gamified sports applications. Covers pixel-art aesthetics, retro HUD construction, Web Audio API 8-bit sound synthesis, and turn-based duel resolution loops.
license: Apache-2.0
metadata:
  version: "1.0"
  last-reviewed: "2026-10-17"
  skill-author: Antigravity Agentic Team
---

# Game Design & Arcade UI: Retro Sports & Gamified Mechanics

## 1. Retro Arcade Visual Aesthetic (8-bit / 16-bit Fliperama)
- **Typography**: Google Font `'Press Start 2P', monospace` for arcade headers, scores, and status messages; high-contrast monospace for stats.
- **Color Palette**:
  - Arcade Cabinet Dark: `#070A12` with subtle CRT scanline gradient.
  - Neon Cyan / 1P: `#00FFCC` (Player 1, High Score, Clean Strikes).
  - Neon Pink / Magenta: `#FF007F` (Boss Batter, Danger, High Leverage).
  - Arcade Amber / Coin: `#FFCC00` (Credits, Combo Multiplier, Insert Coin).
  - CRT Glow: `text-shadow: 0 0 8px rgba(..., 0.8); box-shadow: 0 0 15px rgba(..., 0.6)`.
- **Pixel-Art HUD Layout**:
  - Top Bar: `1UP SCORE`, `HIGH SCORE`, `LIVES / CORRIDAS RESTANTES (❤️❤️❤️)`, `COMBO MULTIPLIER`.
  - Duel Arena: Matchup card with animated retro borders, Strike Zone 3x3 pixel grid, and pitch commit button.

---

## 2. Web Audio API 8-bit Chiptune Synthesis (Zero External Dependencies)
Streamlit can trigger lightweight browser-native synthesized sound effects using inline JavaScript Web Audio API without needing external MP3 files:

```javascript
function playArcadeSound(type) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.connect(gain);
  gain.connect(ctx.destination);

  if (type === 'strike') {
    // Punchy high-to-low beep
    osc.type = 'square';
    osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5
    osc.frequency.exponentialRampToValueAtTime(220, ctx.currentTime + 0.15);
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 0.15);
    osc.start();
    osc.stop(ctx.currentTime + 0.15);
  } else if (type === 'hit') {
    // Noise burst / crash sound
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(150, ctx.currentTime);
    osc.frequency.linearRampToValueAtTime(80, ctx.currentTime + 0.25);
    gain.gain.setValueAtTime(0.4, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 0.25);
    osc.start();
    osc.stop(ctx.currentTime + 0.25);
  } else if (type === 'strikeout') {
    // Fanfare ascending 3-tone arpeggio
    // ...
  }
}
```

---

## 3. RPG Attribute Distribution & Priors Mapping
- **Player Stat Budget**: 100 points allocated across attributes:
  - `VEL (Velocidade)`: Dictates Fastball preference.
  - `EFE (Efeito/Spin)`: Dictates Breaking Ball preference.
  - `TIM (Timing/Decepção)`: Dictates Offspeed preference.
- **Normalization to Databricks Priors**:
  ```python
  total = max(1.0, float(vel + efe + tim))
  pitcher_fb_prior = round(vel / total, 4)
  pitcher_bb_prior = round(efe / total, 4)
  pitcher_os_prior = round(tim / total, 4)
  ```
- **Control / Stamina Attribute**: If player stamina drops, pitch deviation can occur.

---

## 4. Turn-Based Arcade Game Loop
1. **User Action**: Pitcher selects pitch category and location (Strike vs. Chase).
2. **AI Action**: Databricks Serverless Model Serving predicts user choice given game context.
3. **Duel Resolution**:
   - Compares user selection vs. AI prediction + zone location.
   - Updates combo counter, score, balls, strikes, outs, runners, and lives.
4. **Game Over & High Score**:
   - When player runs allowed reaches 3 (Game Over), displays retro initials entry modal (`[A][A][A]`), saving session high scores.

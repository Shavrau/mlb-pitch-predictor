"""
Retro Arcade Baseball Survival Minigame Engine.

Manages pure game logic for the User vs. AI duel on the mound:
- MLB Star Pitcher Tier-based arsenals (Tier S, Tier A, Tier B).
- Official MLB Star Batters with tactical traits.
- RPG attribute points distribution into normalized pitcher priors.
- Arcade game state tracking (score, lives, outs, count, combo, bases, history, timing meter).
- Probabilistic pitch duel resolution matrix:
  - Whiff, Foul ball, Weak contact out, Base hit, Home run.
  - Batter Timing Meter (Anti-Spam): repeating pitches warms up the batter's timing,
    while changing pitches breaks timing and resets the meter.
- High score leaderboard and batter generation.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

LEAGUE_AVG_FB = 0.5543
LEAGUE_AVG_BB = 0.3053
LEAGUE_AVG_OS = 0.1404

# ---------------------------------------------------------------------------
# 1. Registro Oficial de Rebatedores Astros da MLB
# ---------------------------------------------------------------------------

MLB_BATTERS: list[dict[str, Any]] = [
    {
        "name": "Shohei Ohtani",
        "team": "Los Angeles Dodgers",
        "stand": "L",
        "archetype": "Fenômeno da Força (Power Slugger)",
        "description": "Clube 50/50. Esmaga retas no coração da zona; vulnerável a bolas de efeito no canto externo baixo.",
        "trait": "Devorador de Retas: +25% de chance de rebatida forte contra bolas rápidas.",
        "trait_id": "anti_fastball",
    },
    {
        "name": "Aaron Judge",
        "team": "New York Yankees",
        "stand": "R",
        "archetype": "Gigante da Força (Power Giant)",
        "description": "Alcance massivo e saída de bola de 115+ mph. Não perdoa arremessos no topo da zona.",
        "trait": "Castigo no Topo: Perigo extremo contra bolas altas na zona de strike.",
        "trait_id": "high_zone_punisher",
    },
    {
        "name": "Juan Soto",
        "team": "New York Mets",
        "stand": "L",
        "archetype": "Mestre da Disciplina (Plate Discipline)",
        "description": "Olho de águia no bastão. Dificilmente é enganado por arremessos fora da zona.",
        "trait": "Olho de Águia: Raramente vai atrás de bolas fora da zona.",
        "trait_id": "eye_of_the_hawk",
    },
    {
        "name": "Mookie Betts",
        "team": "Los Angeles Dodgers",
        "stand": "R",
        "archetype": "Mestre do Contato & Precisão",
        "description": "Controle cirúrgico do taco. Dificilmente sofre strikeout e coloca a bola em jogo.",
        "trait": "Rei do Contato: Bate quase tudo para foul e raramente sofre strikeout.",
        "trait_id": "contact_king",
    },
    {
        "name": "Bryce Harper",
        "team": "Philadelphia Phillies",
        "stand": "L",
        "archetype": "Slugger Decisivo (Clutch)",
        "description": "Rebatedor agressivo em contagens iniciais com poder brutal para o campo oposto.",
        "trait": "Emboscada Inicial: Ataca agressivamente nos primeiros arremessos (0-0 e 1-0).",
        "trait_id": "aggressive_ambush",
    },
]

# Aliases para compatibilidade retrô
RETRO_BATTERS: list[dict[str, str]] = [
    {
        "name": b["name"],
        "archetype": b["archetype"],
        "description": b["description"],
    }
    for b in MLB_BATTERS
]

# ---------------------------------------------------------------------------
# 2. Arsenais Oficiais por Tier dos 5 Arremessadores da MLB
# ---------------------------------------------------------------------------

PITCHER_TIERS: dict[str, dict[str, Any]] = {
    "Paul Skenes": {
        "pitcher_id": 694973,
        "name": "Paul Skenes",
        "team": "Pittsburgh Pirates",
        "p_throws": "R",
        "tags": ["🚀 102 mph", "⚡ Topo da Zona", "🔥 RHP", "💪 Alta Potência"],
        "description": "Potência pura e velocidade terminal inigualável no topo da zona de strike.",
        "priors": (0.64, 0.28, 0.08),
        "pitches": [
            {
                "tier": "S",
                "tier_label": "⭐ TIER S",
                "name": "4-Seam Fastball",
                "code": "FF",
                "category": "Fastball",
                "speed": "102 mph",
                "desc": "Foguete no topo da zona (Quase intocável)",
                "whiff_rate": 0.40,
                "foul_rate": 0.35,
            },
            {
                "tier": "A",
                "tier_label": "⚡ TIER A",
                "name": "Splinker",
                "code": "FS",
                "category": "Offspeed",
                "speed": "94 mph",
                "desc": "Híbrido violento de sinker com splitter",
                "whiff_rate": 0.32,
                "foul_rate": 0.25,
            },
            {
                "tier": "B",
                "tier_label": "🎯 TIER B",
                "name": "Sweeper",
                "code": "ST",
                "category": "Breaking Ball",
                "speed": "88 mph",
                "desc": "Varredura horizontal para desequilibrar",
                "whiff_rate": 0.25,
                "foul_rate": 0.20,
            },
        ],
    },
    "Corbin Burnes": {
        "pitcher_id": 669203,
        "name": "Corbin Burnes",
        "team": "Baltimore Orioles",
        "p_throws": "R",
        "tags": ["🌪️ 2800+ RPM", "🌀 Quebra Lateral", "🔥 RHP", "🏆 Cy Young"],
        "description": "Especialista em rotação extrema com Cutter cortante e curvas devastadoras.",
        "priors": (0.42, 0.46, 0.12),
        "pitches": [
            {
                "tier": "S",
                "tier_label": "⭐ TIER S",
                "name": "Cutter",
                "code": "FC",
                "category": "Fastball",
                "speed": "95 mph",
                "desc": "Corte seco e pesado na linha da cintura",
                "whiff_rate": 0.38,
                "foul_rate": 0.33,
            },
            {
                "tier": "A",
                "tier_label": "⚡ TIER A",
                "name": "Curveball",
                "code": "CU",
                "category": "Breaking Ball",
                "speed": "84 mph",
                "desc": "Arco 12-6 de rotação altíssima (2800 RPM)",
                "whiff_rate": 0.34,
                "foul_rate": 0.26,
            },
            {
                "tier": "B",
                "tier_label": "🎯 TIER B",
                "name": "Changeup",
                "code": "CH",
                "category": "Offspeed",
                "speed": "89 mph",
                "desc": "Desaceleração furtiva na borda inferior",
                "whiff_rate": 0.24,
                "foul_rate": 0.18,
            },
        ],
    },
    "Logan Webb": {
        "pitcher_id": 657277,
        "name": "Logan Webb",
        "team": "San Francisco Giants",
        "p_throws": "R",
        "tags": ["🎭 Disfarce de Trajetória", "📉 Bolas Rasteiras", "🔥 RHP", "🎯 Controle Cirúrgico"],
        "description": "Mestre no disfarce de trajetória e especialista em induzir contatos fracos rasteiros.",
        "priors": (0.46, 0.22, 0.32),
        "pitches": [
            {
                "tier": "S",
                "tier_label": "⭐ TIER S",
                "name": "Changeup",
                "code": "CH",
                "category": "Offspeed",
                "speed": "87 mph",
                "desc": "Cai da mesa na borda inferior da zona",
                "whiff_rate": 0.39,
                "foul_rate": 0.32,
            },
            {
                "tier": "A",
                "tier_label": "⚡ TIER A",
                "name": "Sinker",
                "code": "SI",
                "category": "Fastball",
                "speed": "93 mph",
                "desc": "Afundamento pesado gerador de rolamento",
                "whiff_rate": 0.30,
                "foul_rate": 0.25,
            },
            {
                "tier": "B",
                "tier_label": "🎯 TIER B",
                "name": "Sweeper",
                "code": "ST",
                "category": "Breaking Ball",
                "speed": "83 mph",
                "desc": "Desvio lateral amplo fora do alcance do bastão",
                "whiff_rate": 0.25,
                "foul_rate": 0.20,
            },
        ],
    },
    "Chris Sale": {
        "pitcher_id": 519242,
        "name": "Chris Sale",
        "team": "Atlanta Braves",
        "p_throws": "L",
        "tags": ["🏹 Canhoto LHP", "📐 Soltura Baixa (25°)", "❄️ Strikeout Fulminante", "🏆 Cy Young NL"],
        "description": "Canhoto lendário com ângulo de soltura lateral cruzado e slider letal.",
        "priors": (0.48, 0.36, 0.16),
        "pitches": [
            {
                "tier": "S",
                "tier_label": "⭐ TIER S",
                "name": "Sweeping Slider",
                "code": "SL",
                "category": "Breaking Ball",
                "speed": "81 mph",
                "desc": "Slider fulminante com quebra horizontal devastadora",
                "whiff_rate": 0.42,
                "foul_rate": 0.35,
            },
            {
                "tier": "A",
                "tier_label": "⚡ TIER A",
                "name": "4-Seam Fastball",
                "code": "FF",
                "category": "Fastball",
                "speed": "95 mph",
                "desc": "Reta furiosa de ângulo cruzado",
                "whiff_rate": 0.31,
                "foul_rate": 0.25,
            },
            {
                "tier": "B",
                "tier_label": "🎯 TIER B",
                "name": "Changeup",
                "code": "CH",
                "category": "Offspeed",
                "speed": "86 mph",
                "desc": "Flutua enganando destros",
                "whiff_rate": 0.23,
                "foul_rate": 0.18,
            },
        ],
    },
    "Yoshinobu Yamamoto": {
        "pitcher_id": 808967,
        "name": "Yoshinobu Yamamoto",
        "team": "Los Angeles Dodgers",
        "p_throws": "R",
        "tags": ["🎌 Dardo Japonês", "👻 Bola Fantasma", "🌈 Arco-Íris", "🔥 RHP"],
        "description": "Superestrela japonesa com Splitter fantasma e mecânica compacta.",
        "priors": (0.44, 0.32, 0.24),
        "pitches": [
            {
                "tier": "S",
                "tier_label": "⭐ TIER S",
                "name": "Splitter",
                "code": "FS",
                "category": "Offspeed",
                "speed": "90 mph",
                "desc": "Bola fantasma que desaparece do campo de visão",
                "whiff_rate": 0.41,
                "foul_rate": 0.34,
            },
            {
                "tier": "A",
                "tier_label": "⚡ TIER A",
                "name": "Rainbow Curve",
                "code": "CU",
                "category": "Breaking Ball",
                "speed": "78 mph",
                "desc": "Curva arco-íris de queda vertical vertiginosa",
                "whiff_rate": 0.32,
                "foul_rate": 0.26,
            },
            {
                "tier": "B",
                "tier_label": "🎯 TIER B",
                "name": "4-Seam Fastball",
                "code": "FF",
                "category": "Fastball",
                "speed": "96 mph",
                "desc": "Reta veloz com efeito ascendente perceptível",
                "whiff_rate": 0.26,
                "foul_rate": 0.20,
            },
        ],
    },
}

# Mapeamento de mira rápida simplificada (3 botões em vez de 13)
QUICK_AIM_MAP: dict[str, dict[str, Any]] = {
    "Topo da Zona": {
        "zone": 2,
        "label": "⬆️ Topo da Zona",
        "desc": "Reta no alto / Zona de Strike 2",
        "is_in_zone": True,
    },
    "Fundo da Zona": {
        "zone": 8,
        "label": "⬇️ Fundo da Zona",
        "desc": "Bola de efeito no pé / Zona de Strike 8",
        "is_in_zone": True,
    },
    "Fora da Zona (Isca)": {
        "zone": 14,
        "label": "🎣 Fora da Zona (Isca)",
        "desc": "Bola no chão para induzir swing no vazio / Zona 14",
        "is_in_zone": False,
    },
}


def get_pitcher_tier_info(pitcher_name: str) -> dict[str, Any]:
    """Retorna o registro de tiers e arsenal para o arremessador especificado."""
    return PITCHER_TIERS.get(pitcher_name, PITCHER_TIERS["Paul Skenes"])


def calculate_priors_from_stats(vel: int, efe: int, tim: int) -> tuple[float, float, float]:
    """
    Converte pontos de atributos em priors normalizados somando 1.0.

    Args:
        vel: Pontos de Velocidade (Fastball).
        efe: Pontos de Efeito (Breaking Ball).
        tim: Pontos de Variação/Timing (Offspeed).

    Returns:
        Tupla (pitcher_fb_prior, pitcher_bb_prior, pitcher_os_prior) somando 1.0.
    """
    total = vel + efe + tim
    if total <= 0:
        return (LEAGUE_AVG_FB, LEAGUE_AVG_BB, LEAGUE_AVG_OS)

    fb = float(vel) / total
    bb = float(efe) / total
    os_val = float(tim) / total

    sum_priors = fb + bb + os_val
    if abs(sum_priors - 1.0) > 1e-6:
        fb = fb / sum_priors
        bb = bb / sum_priors
        os_val = 1.0 - fb - bb

    return (fb, bb, os_val)


def get_random_retro_batter() -> dict[str, Any]:
    """Retorna um perfil de astro da MLB para o duelo no montinho."""
    return copy.deepcopy(random.choice(MLB_BATTERS))


@dataclass
class ArcadeGameState:
    """Estado do jogo de sobrevivência arcade (Você vs. IA)."""

    score: int = 0
    high_score: int = 0
    lives: int = 3
    outs: int = 0
    balls: int = 0
    strikes: int = 0
    combo_multiplier: int = 1
    consecutive_k: int = 0
    on_1b: int = 0
    on_2b: int = 0
    on_3b: int = 0
    runs_allowed: int = 0
    game_over: bool = False
    current_batter_name: str = ""
    pitcher_name: str = "Paul Skenes"
    timing_meter: int = 0  # 0: Frio, 1: Alerta, 2: Quente, 3: Na Espera!
    consecutive_same_pitch: int = 0
    last_pitch_code: str = ""
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.current_batter_name:
            self.current_batter_name = get_random_retro_batter()["name"]


def resolve_arcade_pitch(
    state: ArcadeGameState,
    user_category: str,
    user_type: str,
    user_zone: int,
    ai_prediction: str,
    pitch_tier: str = "A",
    probabilistic: bool = False,
    rng_seed: float | None = None,
) -> tuple[ArcadeGameState, dict]:
    """
    Resolve o duelo de arremesso entre Jogador (Arremessador) e IA (Rebatedor).

    Suporta:
    - Modo determinístico (probabilistic=False): preserva 100% dos contratos originais de teste.
    - Modo probabilístico de jogo (probabilistic=True): calcula whiff, foul ball, contato fraco
      e rebatida baseado no Tier do arremesso (S, A, B) e no Medidor de Timing do Rebatedor (anti-spam).

    Args:
        state: Estado atual do jogo arcade.
        user_category: Categoria escolhida ("Fastball", "Breaking Ball", "Offspeed").
        user_type: Código do arremesso ("FF", "SL", "CH", etc.).
        user_zone: Zona Statcast (1..9 na zona, 11..14 fora da zona).
        ai_prediction: Predição feita pelo modelo Databricks.
        pitch_tier: Nível do arremesso ("S", "A", "B").
        probabilistic: Se True, aplica chances de whiff/foul/out/hit e penalidade de timing.
        rng_seed: Valor opcional entre 0.0 e 1.0 para testes reprodutíveis do modo probabilístico.

    Returns:
        Tupla (novo_estado, resultado_do_arremesso).
    """
    next_state = copy.deepcopy(state)
    is_in_zone = 1 <= user_zone <= 9
    ai_guessed = user_category.strip().lower() == ai_prediction.strip().lower()

    # Atualização do Medidor de Timing (Anti-Spam)
    timing_broken = False
    if next_state.last_pitch_code and user_type == next_state.last_pitch_code:
        next_state.consecutive_same_pitch += 1
        if next_state.consecutive_same_pitch >= 2:
            next_state.timing_meter = min(3, next_state.timing_meter + 1)
    else:
        if next_state.timing_meter > 0:
            timing_broken = True
        next_state.timing_meter = 0
        next_state.consecutive_same_pitch = 1

    next_state.last_pitch_code = user_type

    outcome: dict = {
        "result": "",
        "user_category": user_category,
        "user_type": user_type,
        "user_zone": user_zone,
        "ai_prediction": ai_prediction,
        "pitch_tier": pitch_tier,
        "runs": 0,
        "message": "",
        "inning_cleared": False,
        "timing_meter": next_state.timing_meter,
        "timing_broken": timing_broken,
    }

    rand_val = rng_seed if rng_seed is not None else random.random()

    if is_in_zone:
        if ai_guessed:
            # IA ADIVINHOU O ARREMESSO NA ZONA
            # 1. Se o medidor de timing estiver cheio (>= 2), o rebatedor estava na espera -> CRUSH!
            if probabilistic and next_state.timing_meter >= 2:
                # O rebatedor cronometrou perfeitamente o arremesso repetido!
                runs = 1 + next_state.on_1b + next_state.on_2b + next_state.on_3b
                next_state.on_1b = 0
                next_state.on_2b = 0
                next_state.on_3b = 0
                next_state.lives -= runs
                next_state.runs_allowed += runs
                next_state.combo_multiplier = 1
                next_state.consecutive_k = 0
                next_state.balls = 0
                next_state.strikes = 0
                next_state.current_batter_name = get_random_retro_batter()["name"]

                outcome["result"] = "HOME_RUN"
                outcome["runs"] = runs
                outcome["message"] = (
                    f"💥 HOME RUN DEVASTADOR! O rebatedor cronometrou seu arremesso repetido ({next_state.consecutive_same_pitch}x {user_type})! {runs} corrida(s)!"
                )
            elif user_zone == 5:
                # Coração da zona (Meatball) -> Home Run certo
                runs = 1 + next_state.on_1b + next_state.on_2b + next_state.on_3b
                next_state.on_1b = 0
                next_state.on_2b = 0
                next_state.on_3b = 0
                next_state.lives -= runs
                next_state.runs_allowed += runs
                next_state.combo_multiplier = 1
                next_state.consecutive_k = 0
                next_state.balls = 0
                next_state.strikes = 0
                next_state.current_batter_name = get_random_retro_batter()["name"]

                outcome["result"] = "HOME_RUN"
                outcome["runs"] = runs
                outcome["message"] = (
                    f"💥 HOME RUN! A IA leu a mente na Zona 5 (Meatball)! {runs} corrida(s) anotada(s)!"
                )
            else:
                # Zonas periféricas 1..4, 6..9
                if probabilistic:
                    # Chance probabilística baseada no Tier do Arremesso:
                    # Tier S: 35% Foul (salva como strike se strikes < 2), 25% Contato Fraco (Out), 40% Hit
                    # Tier A: 20% Foul, 20% Contato Fraco (Out), 60% Hit
                    # Tier B: 10% Foul, 10% Contato Fraco, 80% Hit
                    foul_thresh = 0.35 if pitch_tier == "S" else (0.20 if pitch_tier == "A" else 0.10)
                    out_thresh = foul_thresh + (0.25 if pitch_tier == "S" else (0.20 if pitch_tier == "A" else 0.10))

                    if rand_val < foul_thresh:
                        # FOUL BALL: O rebatedor adivinhou mas a qualidade do arremesso espirrou no taco
                        if next_state.strikes < 2:
                            next_state.strikes += 1
                        next_state.score += 50
                        outcome["result"] = "FOUL"
                        outcome["message"] = (
                            f"⚡ BOLA EM FOUL! A IA adivinhou {ai_prediction}, mas o arremesso Tier {pitch_tier} espirrou no taco!"
                        )
                    elif rand_val < out_thresh:
                        # CONTATO FRACO (GROUNDOUT / FLYOUT) -> OUT!
                        next_state.outs += 1
                        next_state.balls = 0
                        next_state.strikes = 0
                        next_state.score += 200
                        next_state.current_batter_name = get_random_retro_batter()["name"]
                        outcome["result"] = "OUT"
                        outcome["message"] = (
                            f"🧤 CONTATO FRACO (OUT)! A IA esperava {ai_prediction}, mas a qualidade Tier {pitch_tier} gerou um rolamento fácil!"
                        )

                        if next_state.outs >= 3:
                            next_state.score += 1000
                            next_state.outs = 0
                            next_state.on_1b = 0
                            next_state.on_2b = 0
                            next_state.on_3b = 0
                            outcome["result"] = "INNING_CLEARED"
                            outcome["inning_cleared"] = True
                            outcome["message"] = (
                                f"🏁 ENTRADA LIMPA! 3 OUTS! Bônus de +1.000 pts! (Total: {next_state.score} pts)"
                            )
                    else:
                        # REBATIDA VÁLIDA
                        runs = next_state.on_3b
                        next_state.on_3b = next_state.on_2b
                        next_state.on_2b = next_state.on_1b
                        next_state.on_1b = 1
                        next_state.lives -= runs
                        next_state.runs_allowed += runs
                        next_state.combo_multiplier = 1
                        next_state.consecutive_k = 0
                        next_state.balls = 0
                        next_state.strikes = 0
                        next_state.current_batter_name = get_random_retro_batter()["name"]

                        outcome["result"] = "HIT"
                        outcome["runs"] = runs
                        outcome["message"] = (
                            f"⚡ REBATIDA VÁLIDA! A IA adivinhou {ai_prediction} e bateu firme!"
                            + (f" ({runs} corrida anotada!)" if runs > 0 else "")
                        )
                else:
                    # Modo determinístico puro
                    runs = next_state.on_3b
                    next_state.on_3b = next_state.on_2b
                    next_state.on_2b = next_state.on_1b
                    next_state.on_1b = 1
                    next_state.lives -= runs
                    next_state.runs_allowed += runs
                    next_state.combo_multiplier = 1
                    next_state.consecutive_k = 0
                    next_state.balls = 0
                    next_state.strikes = 0
                    next_state.current_batter_name = get_random_retro_batter()["name"]

                    outcome["result"] = "HIT"
                    outcome["runs"] = runs
                    outcome["message"] = (
                        f"⚡ REBATIDA VÁLIDA! A IA adivinhou {ai_prediction}!"
                        + (f" ({runs} corrida anotada!)" if runs > 0 else "")
                    )

            if next_state.lives <= 0:
                next_state.lives = 0
                next_state.game_over = True
                outcome["message"] += " 💀 FIM DE JOGO!"
        else:
            # IA ERROU O ARREMESSO NA ZONA -> STRIKE!
            if next_state.strikes < 2:
                next_state.strikes += 1
                next_state.score += 100
                outcome["result"] = "STRIKE"
                outcome["message"] = (
                    f"🎯 STRIKE! A IA esperava {ai_prediction}, mas veio {user_category}! (+100 pts)"
                )
            else:
                # 3º Strike -> STRIKEOUT!
                points = 500 * next_state.combo_multiplier
                next_state.score += points
                next_state.combo_multiplier += 1
                next_state.consecutive_k += 1
                next_state.outs += 1
                next_state.balls = 0
                next_state.strikes = 0
                next_state.current_batter_name = get_random_retro_batter()["name"]

                outcome["result"] = "STRIKEOUT"
                outcome["message"] = (
                    f"🔥 STRIKEOUT (K)! IA enganada na zona! (+{points} pts | Combo x{next_state.combo_multiplier})"
                )

                if next_state.outs >= 3:
                    next_state.score += 1000
                    next_state.outs = 0
                    next_state.on_1b = 0
                    next_state.on_2b = 0
                    next_state.on_3b = 0
                    outcome["result"] = "INNING_CLEARED"
                    outcome["inning_cleared"] = True
                    outcome["message"] = (
                        f"🏁 ENTRADA LIMPA! 3 OUTS! Bônus de +1.000 pts! (Total: {next_state.score} pts)"
                    )
    else:
        # FORA DA ZONA (ZONAS 11..14 / CHASE)
        if ai_guessed:
            # IA tem disciplina e não vai na bola fora -> Ball / Walk
            if next_state.balls < 3:
                next_state.balls += 1
                outcome["result"] = "BALL"
                outcome["message"] = (
                    f"👀 BOLA! A IA previu {ai_prediction} fora da zona e não foi atrás."
                )
            else:
                # 4ª Bola -> WALK!
                next_state.balls = 0
                next_state.strikes = 0
                next_state.current_batter_name = get_random_retro_batter()["name"]

                runs = 0
                if next_state.on_1b == 1:
                    if next_state.on_2b == 1:
                        if next_state.on_3b == 1:
                            runs = 1
                            next_state.lives -= 1
                            next_state.runs_allowed += 1
                        else:
                            next_state.on_3b = 1
                    else:
                        next_state.on_2b = 1
                next_state.on_1b = 1

                outcome["result"] = "WALK"
                outcome["runs"] = runs
                outcome["message"] = (
                    "🚶 BASE POR BOLAS (WALK)! A IA teve disciplina com 4 bolas."
                    + (f" ({runs} corrida anotada!)" if runs > 0 else "")
                )
                if next_state.lives <= 0:
                    next_state.lives = 0
                    next_state.game_over = True
                    outcome["message"] += " 💀 FIM DE JOGO!"
        else:
            # IA foi enganada fora da zona -> WHIFF!
            if next_state.strikes < 2:
                next_state.strikes += 1
                next_state.score += 150
                outcome["result"] = "WHIFF"
                outcome["message"] = (
                    "🌪️ SWING NO VAZIO (WHIFF)! A IA foi atrás da bola fora! (+150 pts)"
                )
            else:
                # 3º Strike -> STRIKEOUT por Whiff!
                points = 500 * next_state.combo_multiplier
                next_state.score += points
                next_state.combo_multiplier += 1
                next_state.consecutive_k += 1
                next_state.outs += 1
                next_state.balls = 0
                next_state.strikes = 0
                next_state.current_batter_name = get_random_retro_batter()["name"]

                outcome["result"] = "STRIKEOUT"
                outcome["message"] = (
                    f"🔥 STRIKEOUT (K)! IA abanou no vazio! (+{points} pts | Combo x{next_state.combo_multiplier})"
                )

                if next_state.outs >= 3:
                    next_state.score += 1000
                    next_state.outs = 0
                    next_state.on_1b = 0
                    next_state.on_2b = 0
                    next_state.on_3b = 0
                    outcome["result"] = "INNING_CLEARED"
                    outcome["inning_cleared"] = True
                    outcome["message"] = (
                        f"🏁 ENTRADA LIMPA! 3 OUTS! Bônus de +1.000 pts! (Total: {next_state.score} pts)"
                    )

    if timing_broken:
        outcome["message"] = "🧊 Quebrou o timing do rebatedor! " + outcome["message"]

    next_state.high_score = max(next_state.high_score, next_state.score)

    next_state.history.append(
        {
            "pitch_category": user_category,
            "pitch_type": user_type,
            "pitch_zone": user_zone,
            "pitch_tier": pitch_tier,
            "ai_prediction": ai_prediction,
            "result": outcome["result"],
            "score": next_state.score,
            "combo": next_state.combo_multiplier,
            "timing_meter": next_state.timing_meter,
        }
    )

    return next_state, outcome


def register_high_score(
    initials: str,
    score: int,
    leaderboard: list[dict] | None = None,
) -> list[dict]:
    """
    Registra uma nova pontuação no ranking local retrô.

    Args:
        initials: 3 iniciais do jogador (ex: "AAA", "RHI").
        score: Pontuação obtida.
        leaderboard: Lista existente de recordes.

    Returns:
        Lista com as 10 melhores pontuações ordenadas de forma decrescente.
    """
    clean_initials = (initials or "AAA").strip().upper()[:3]
    if len(clean_initials) < 3:
        clean_initials = (clean_initials + "AAA")[:3]

    board = list(leaderboard) if leaderboard is not None else []
    board.append({"initials": clean_initials, "score": int(score)})
    board.sort(key=lambda entry: entry["score"], reverse=True)
    return board[:10]

"""
Retro Arcade Baseball Survival Minigame Engine.

Manages pure game logic for the User vs. AI duel on the mound:
- RPG attribute points distribution into normalized pitcher priors.
- Arcade game state tracking (score, lives, outs, count, combo, bases, history).
- Pitch duel resolution matrix (In-zone vs Chase zone, AI hit/miss, strikeouts, walks, inning clears).
- High score leaderboard and retro batter generation.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

LEAGUE_AVG_FB = 0.5543
LEAGUE_AVG_BB = 0.3053
LEAGUE_AVG_OS = 0.1404

RETRO_BATTERS: list[dict[str, str]] = [
    {
        "name": "Babe Byte",
        "archetype": "Slugger de 8-bits",
        "description": "Rebatedor de força lendário com taco pixelado.",
    },
    {
        "name": "Pixel Paul",
        "archetype": "Rebatedor de Contato Rápido",
        "description": "Excelente visão de zona e velocidade nos pés.",
    },
    {
        "name": "Mega Manny",
        "archetype": "Batedor Clássico de Fliperama",
        "description": "Não perdoa erros no meio da zona (Meatball).",
    },
    {
        "name": "Slugger 64",
        "archetype": "Rebatedor da Era 16-Bit",
        "description": "Especialista em rebater bolas rápidas no alto.",
    },
    {
        "name": "Ninten Nolan",
        "archetype": "Veterano do fliperama",
        "description": "Caçador de sliders fora da zona.",
    },
    {
        "name": "Cyber Cabrera",
        "archetype": "Campeão da Coroa Tripla Digital",
        "description": "Disciplina perfeita: não vai em bolas fora.",
    },
]


def calculate_priors_from_stats(vel: int, efe: int, tim: int) -> tuple[float, float, float]:
    """
    Converte os 100 pontos de atributos RPG em priors normalizados somando 1.0.

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

    # Ajuste fino de arredondamento para garantir soma estrita de 1.0
    sum_priors = fb + bb + os_val
    if abs(sum_priors - 1.0) > 1e-6:
        fb = fb / sum_priors
        bb = bb / sum_priors
        os_val = 1.0 - fb - bb

    return (fb, bb, os_val)


def get_random_retro_batter() -> dict[str, str]:
    """Retorna um perfil aleatório de rebatedor retrô para exibição no HUD."""
    return copy.deepcopy(random.choice(RETRO_BATTERS))


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
) -> tuple[ArcadeGameState, dict]:
    """
    Resolve o duelo de arremesso entre Jogador (Arremessador) e IA (Rebatedor).

    Args:
        state: Estado atual do jogo arcade.
        user_category: Categoria escolhida ("Fastball", "Breaking Ball", "Offspeed").
        user_type: Código do arremesso ("FF", "SL", "CH", etc.).
        user_zone: Zona Statcast (1..9 na zona, 11..14 fora da zona).
        ai_prediction: Predição feita pelo modelo Databricks ("Fastball", "Breaking Ball", "Offspeed").

    Returns:
        Tupla (novo_estado, resultado_do_arremesso).
    """
    next_state = copy.deepcopy(state)
    is_in_zone = 1 <= user_zone <= 9
    ai_guessed = user_category.strip().lower() == ai_prediction.strip().lower()

    outcome: dict = {
        "result": "",
        "user_category": user_category,
        "user_type": user_type,
        "user_zone": user_zone,
        "ai_prediction": ai_prediction,
        "runs": 0,
        "message": "",
        "inning_cleared": False,
    }

    if is_in_zone:
        if ai_guessed:
            # IA adivinhou e a bola está na zona -> Rebatida!
            if user_zone == 5:
                # Coração da zona (Meatball) -> Home Run!
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
                    f"💥 HOME RUN! A IA leu a mente na Zona 5! {runs} corrida(s) anotada(s)!"
                )
            else:
                # Zonas periféricas 1..4, 6..9 -> Rebatida Simples com avanço
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
            # IA errou o arremesso na zona -> Strike Looking / Foul!
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
        # Fora da Zona (Zonas 11..14)
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

                # Avanço forçado
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

    # Atualiza High Score
    next_state.high_score = max(next_state.high_score, next_state.score)

    # Registra no histórico do duelo
    next_state.history.append(
        {
            "pitch_category": user_category,
            "pitch_type": user_type,
            "pitch_zone": user_zone,
            "ai_prediction": ai_prediction,
            "result": outcome["result"],
            "score": next_state.score,
            "combo": next_state.combo_multiplier,
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

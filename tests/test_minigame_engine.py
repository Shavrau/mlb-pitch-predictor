"""
Unit tests for the Retro Arcade Survival Minigame Engine (`pitch_predictor.minigame_engine`) — Red Phase TDD.

Covers:
1. `calculate_priors_from_stats(vel, efe, tim)`:
   - Normalizes 100 attribute points into `(pitcher_fb_prior, pitcher_bb_prior, pitcher_os_prior)` summing to 1.0.
   - Falls back to MLB default priors `(0.5543, 0.3053, 0.1404)` when sum <= 0.
   - Handles arbitrary integer distributions while guaranteeing float sum == 1.0.
2. `ArcadeGameState` dataclass defaults and state tracking fields.
3. `resolve_arcade_pitch(state, user_category, user_type, user_zone, ai_prediction)`:
   - In-Zone (1..9) + AI Guessed (`user_category == ai_prediction`):
     - Zone 5 (Heart) -> `"HOME_RUN"`, clears bases, deducts `lives` by runs scored, resets combo to 1.
     - Other Strike Zones (1..4, 6..9) -> `"HIT"`, advances runners, deducts `lives` if runs score, resets combo to 1.
     - Triggers `game_over = True` when `lives <= 0`.
   - In-Zone (1..9) + AI Missed (`user_category != ai_prediction`):
     - `strikes < 2` -> `"STRIKE"`, `strikes += 1`, `score += 100`.
     - `strikes == 2` -> `"STRIKEOUT"`, `outs += 1`, `score += 500 * combo_multiplier`,
       `combo_multiplier += 1`, `consecutive_k += 1`, resets count to 0-0.
   - Chase Zone (11..14) + AI Guessed (`user_category == ai_prediction`):
     - `balls < 3` -> `"BALL"`, `balls += 1`.
     - `balls == 3` -> `"WALK"`, forced runner advancement, resets count to 0-0.
   - Chase Zone (11..14) + AI Missed (`user_category != ai_prediction`):
     - `strikes < 2` -> `"WHIFF"`, `strikes += 1`, `score += 150`.
     - `strikes == 2` -> `"STRIKEOUT"`, `score += 500 * combo_multiplier`, `combo_multiplier += 1`, resets count to 0-0.
   - 3rd Out (`outs == 2` + Strikeout):
     - Clears the inning (`"INNING_CLEARED"`), awards `+1000` bonus points, clears bases (`on_1b=0, on_2b=0, on_3b=0`),
       and resets `outs = 0`.
4. `register_high_score(initials, score, leaderboard)`:
   - Uppercases 3-letter initials, sorts descending by `score`, and caps leaderboard at top 10.
5. `get_random_retro_batter()`:
   - Returns a valid retro batter dictionary with HUD display attributes.
"""

from __future__ import annotations

import pytest

from pitch_predictor.minigame_engine import (
    ArcadeGameState,
    calculate_priors_from_stats,
    get_random_retro_batter,
    register_high_score,
    resolve_arcade_pitch,
)

# ---------------------------------------------------------------------------
# 1. Attribute Points -> Pitcher Priors Normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vel", "efe", "tim", "expected"),
    [
        (50, 30, 20, (0.50, 0.30, 0.20)),
        (70, 20, 10, (0.70, 0.20, 0.10)),
        (25, 50, 25, (0.25, 0.50, 0.25)),
        (33, 33, 34, (0.33, 0.33, 0.34)),
    ],
)
def test_calculate_priors_from_stats_exact_distributions(
    vel: int, efe: int, tim: int, expected: tuple[float, float, float]
) -> None:
    """Verify calculate_priors_from_stats normalizes integer stats into priors summing to 1.0."""
    fb, bb, os_val = calculate_priors_from_stats(vel, efe, tim)
    assert fb == pytest.approx(expected[0], abs=1e-4)
    assert bb == pytest.approx(expected[1], abs=1e-4)
    assert os_val == pytest.approx(expected[2], abs=1e-4)
    assert (fb + bb + os_val) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    ("vel", "efe", "tim"),
    [
        (0, 0, 0),
        (-10, -5, 0),
    ],
)
def test_calculate_priors_from_stats_zero_or_negative_returns_mlb_defaults(
    vel: int, efe: int
    , tim: int
) -> None:
    """Verify zero or negative stat totals fall back to MLB baseline priors (0.5543, 0.3053, 0.1404)."""
    fb, bb, os_val = calculate_priors_from_stats(vel, efe, tim)
    assert (fb, bb, os_val) == pytest.approx((0.5543, 0.3053, 0.1404), abs=1e-4)
    assert (fb + bb + os_val) == pytest.approx(1.0, abs=1e-4)


def test_calculate_priors_from_stats_arbitrary_integers_sum_to_one() -> None:
    """Verify arbitrary non-100 positive integer combinations still normalize to sum == 1.0."""
    fb, bb, os_val = calculate_priors_from_stats(17, 29, 11)
    assert fb > 0 and bb > 0 and os_val > 0
    assert (fb + bb + os_val) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. ArcadeGameState Dataclass Contract
# ---------------------------------------------------------------------------


def test_arcade_game_state_default_initialization() -> None:
    """Verify ArcadeGameState initializes with the expected arcade survival default attributes."""
    state = ArcadeGameState()
    assert state.score == 0
    assert state.high_score == 0
    assert state.lives == 3
    assert state.outs == 0
    assert state.balls == 0
    assert state.strikes == 0
    assert state.combo_multiplier == 1
    assert state.consecutive_k == 0
    assert state.on_1b == 0
    assert state.on_2b == 0
    assert state.on_3b == 0
    assert state.runs_allowed == 0
    assert state.game_over is False
    assert isinstance(state.current_batter_name, str)
    assert len(state.current_batter_name) > 0
    assert isinstance(state.history, list)
    assert len(state.history) == 0


# ---------------------------------------------------------------------------
# 3. Duel Resolution Matrix (`resolve_arcade_pitch`)
# ---------------------------------------------------------------------------


def test_resolve_arcade_pitch_in_zone_ai_misses_adds_strike_and_100_pts() -> None:
    """In-Zone (1..9) + AI missed with strikes < 2 -> STRIKE, strikes += 1, score += 100."""
    state = ArcadeGameState(balls=1, strikes=0, score=200)
    next_state, outcome = resolve_arcade_pitch(
        state=state,
        user_category="Fastball",
        user_type="FF",
        user_zone=2,
        ai_prediction="Breaking Ball",
    )

    assert outcome["result"] == "STRIKE"
    assert next_state.strikes == 1
    assert next_state.balls == 1
    assert next_state.score == 300
    assert len(next_state.history) == 1


def test_resolve_arcade_pitch_in_zone_ai_misses_on_two_strikes_records_strikeout_and_combo() -> None:
    """In-Zone (1..9) + AI missed with strikes == 2 -> STRIKEOUT, +500 * combo, combo += 1, resets count."""
    state = ArcadeGameState(
        balls=2,
        strikes=2,
        outs=0,
        score=300,
        combo_multiplier=2,
        consecutive_k=1,
    )
    next_state, outcome = resolve_arcade_pitch(
        state=state,
        user_category="Offspeed",
        user_type="CH",
        user_zone=8,
        ai_prediction="Fastball",
    )

    assert outcome["result"] == "STRIKEOUT"
    assert next_state.outs == 1
    assert next_state.balls == 0
    assert next_state.strikes == 0
    assert next_state.score == 300 + (500 * 2)
    assert next_state.combo_multiplier == 3
    assert next_state.consecutive_k == 2


def test_resolve_arcade_pitch_chase_zone_ai_misses_whiff_adds_150_pts() -> None:
    """Chase Zone (11..14) + AI missed with strikes < 2 -> WHIFF, strikes += 1, score += 150."""
    state = ArcadeGameState(balls=0, strikes=1, score=0)
    next_state, outcome = resolve_arcade_pitch(
        state=state,
        user_category="Breaking Ball",
        user_type="SL",
        user_zone=14,
        ai_prediction="Fastball",
    )

    assert outcome["result"] == "WHIFF"
    assert next_state.strikes == 2
    assert next_state.score == 150


def test_resolve_arcade_pitch_chase_zone_ai_misses_on_two_strikes_records_strikeout() -> None:
    """Chase Zone (11..14) + AI missed on 2 strikes -> STRIKEOUT, +500 * combo, combo += 1, resets count."""
    state = ArcadeGameState(balls=1, strikes=2, outs=1, score=100, combo_multiplier=1)
    next_state, outcome = resolve_arcade_pitch(
        state=state,
        user_category="Breaking Ball",
        user_type="ST",
        user_zone=13,
        ai_prediction="Fastball",
    )

    assert outcome["result"] == "STRIKEOUT"
    assert next_state.outs == 2
    assert next_state.balls == 0
    assert next_state.strikes == 0
    assert next_state.score >= 600
    assert next_state.combo_multiplier == 2
    assert next_state.consecutive_k == 1


def test_resolve_arcade_pitch_chase_zone_ai_guesses_takes_ball_and_walk_on_ball_four() -> None:
    """Chase Zone (11..14) + AI guessed -> BALL when balls < 3, and WALK with forced advancement on balls == 3."""
    state = ArcadeGameState(balls=2, strikes=1, on_1b=1, on_2b=0, on_3b=0)
    state_3b, outcome_ball = resolve_arcade_pitch(
        state=state,
        user_category="Breaking Ball",
        user_type="CU",
        user_zone=14,
        ai_prediction="Breaking Ball",
    )
    assert outcome_ball["result"] == "BALL"
    assert state_3b.balls == 3
    assert state_3b.strikes == 1

    # Next pitch: 4th ball -> WALK, forces runner from 1B to 2B and puts batter on 1B
    state_after_walk, outcome_walk = resolve_arcade_pitch(
        state=state_3b,
        user_category="Offspeed",
        user_type="CH",
        user_zone=13,
        ai_prediction="Offspeed",
    )
    assert outcome_walk["result"] == "WALK"
    assert state_after_walk.balls == 0
    assert state_after_walk.strikes == 0
    assert state_after_walk.on_1b == 1
    assert state_after_walk.on_2b == 1


def test_resolve_arcade_pitch_in_zone_ai_guesses_hit_and_zone_5_home_run_deducts_lives() -> None:
    """In-Zone + AI guessed -> HIT on edges (1..4, 6..9) and HOME_RUN on Zone 5, deducting lives and resetting combo."""
    # 1. Edge zone hit with runner on 3B -> scores run, loses 1 life, resets combo to 1
    state = ArcadeGameState(
        lives=3,
        combo_multiplier=4,
        consecutive_k=3,
        on_1b=0,
        on_2b=0,
        on_3b=1,
        balls=1,
        strikes=1,
    )
    state_after_hit, outcome_hit = resolve_arcade_pitch(
        state=state,
        user_category="Fastball",
        user_type="FF",
        user_zone=4,
        ai_prediction="Fastball",
    )
    assert outcome_hit["result"] == "HIT"
    assert state_after_hit.combo_multiplier == 1
    assert state_after_hit.consecutive_k == 0
    assert state_after_hit.lives == 2
    assert state_after_hit.balls == 0
    assert state_after_hit.strikes == 0

    # 2. Zone 5 (Heart) meatball guessed by AI with runner on 1B -> 2-run HOME_RUN -> lives == 0 -> game_over = True
    state_hr_setup = ArcadeGameState(
        lives=2,
        combo_multiplier=2,
        on_1b=1,
        on_2b=0,
        on_3b=0,
    )
    state_game_over, outcome_hr = resolve_arcade_pitch(
        state=state_hr_setup,
        user_category="Fastball",
        user_type="FF",
        user_zone=5,
        ai_prediction="Fastball",
    )
    assert outcome_hr["result"] == "HOME_RUN"
    assert state_game_over.combo_multiplier == 1
    assert state_game_over.on_1b == 0
    assert state_game_over.on_2b == 0
    assert state_game_over.on_3b == 0
    assert state_game_over.lives <= 0
    assert state_game_over.game_over is True


def test_resolve_arcade_pitch_third_out_clears_inning_and_awards_1000_bonus() -> None:
    """Recording the 3rd out triggers INNING_CLEARED, +1000 bonus, clears bases, and resets outs to 0."""
    state = ArcadeGameState(
        score=1000,
        outs=2,
        balls=0,
        strikes=2,
        combo_multiplier=2,
        on_1b=1,
        on_2b=1,
        on_3b=0,
    )
    next_state, outcome = resolve_arcade_pitch(
        state=state,
        user_category="Fastball",
        user_type="FF",
        user_zone=1,
        ai_prediction="Offspeed",
    )

    assert (
        outcome["result"] == "INNING_CLEARED"
        or outcome.get("inning_cleared") is True
    )
    assert next_state.outs == 0
    assert next_state.balls == 0
    assert next_state.strikes == 0
    assert next_state.on_1b == 0
    assert next_state.on_2b == 0
    assert next_state.on_3b == 0
    # 1000 base + (500 * 2 strikeout) + 1000 inning cleared bonus = 3000
    assert next_state.score == 3000
    assert next_state.combo_multiplier == 3


# ---------------------------------------------------------------------------
# 4. Retro High-Score Leaderboard & Random Batter Generator
# ---------------------------------------------------------------------------


def test_register_high_score_sorts_descending_normalizes_initials_and_caps_at_10() -> None:
    """Verify register_high_score uppercases 3-char initials, sorts descending, and keeps top 10."""
    existing = [{"initials": f"P{i:02d}", "score": i * 100} for i in range(1, 11)]
    updated = register_high_score(initials="rhi", score=2500, leaderboard=existing)

    assert len(updated) == 10
    assert updated[0]["initials"] == "RHI"
    assert updated[0]["score"] == 2500
    scores = [entry["score"] for entry in updated]
    assert scores == sorted(scores, reverse=True)


def test_get_random_retro_batter_returns_valid_hud_profile() -> None:
    """Verify get_random_retro_batter returns a dictionary with retro batter name and HUD metadata."""
    batter = get_random_retro_batter()
    assert isinstance(batter, dict)
    assert "name" in batter
    assert isinstance(batter["name"], str)
    assert len(batter["name"]) > 0


# ---------------------------------------------------------------------------
# 5. MLB Star Batters, Pitcher Tiers & Probabilistic Engine Tests
# ---------------------------------------------------------------------------


def test_mlb_batters_registry_contains_five_official_sluggers() -> None:
    """Verify MLB_BATTERS contains all 5 star sluggers with full tactical metadata."""
    from pitch_predictor.minigame_engine import MLB_BATTERS

    assert len(MLB_BATTERS) == 5
    batter_names = [b["name"] for b in MLB_BATTERS]
    assert "Shohei Ohtani" in batter_names
    assert "Aaron Judge" in batter_names
    assert "Juan Soto" in batter_names
    assert "Mookie Betts" in batter_names
    assert "Bryce Harper" in batter_names

    for b in MLB_BATTERS:
        assert "stand" in b and b["stand"] in ("R", "L")
        assert "trait" in b and len(b["trait"]) > 0
        assert "trait_id" in b and len(b["trait_id"]) > 0
        assert "archetype" in b and len(b["archetype"]) > 0


def test_pitcher_tiers_registry_covers_five_pitchers_with_three_tiers() -> None:
    """Verify PITCHER_TIERS registers all 5 MLB stars with S, A, and B tier pitches."""
    from pitch_predictor.minigame_engine import PITCHER_TIERS, get_pitcher_tier_info

    assert len(PITCHER_TIERS) == 5
    pitchers = ["Paul Skenes", "Corbin Burnes", "Logan Webb", "Chris Sale", "Yoshinobu Yamamoto"]
    for p_name in pitchers:
        info = get_pitcher_tier_info(p_name)
        assert info["name"] == p_name
        assert "tags" in info and len(info["tags"]) >= 3
        assert "pitches" in info and len(info["pitches"]) == 3

        tiers = [p["tier"] for p in info["pitches"]]
        assert tiers == ["S", "A", "B"]

        for p in info["pitches"]:
            assert "name" in p and len(p["name"]) > 0
            assert "code" in p and len(p["code"]) >= 2
            assert "category" in p and p["category"] in ("Fastball", "Breaking Ball", "Offspeed")
            assert "speed" in p and "mph" in p["speed"]


def test_probabilistic_resolve_tier_s_foul_and_contact_out() -> None:
    """Verify Tier S pitch induces FOUL or weak OUT when AI guesses right with low rng_seed."""
    state = ArcadeGameState(balls=0, strikes=0, outs=0, lives=3)

    # 1. Low rng_seed (e.g. 0.10 < 0.35 foul_thresh for Tier S) -> FOUL
    state_foul, outcome_foul = resolve_arcade_pitch(
        state=state,
        user_category="Fastball",
        user_type="FF",
        user_zone=2,
        ai_prediction="Fastball",
        pitch_tier="S",
        probabilistic=True,
        rng_seed=0.10,
    )
    assert outcome_foul["result"] == "FOUL"
    assert state_foul.strikes == 1
    assert state_foul.lives == 3

    # 2. Mid rng_seed (0.45, between 0.35 and 0.60 out_thresh for Tier S) -> OUT
    state_out, outcome_out = resolve_arcade_pitch(
        state=state,
        user_category="Fastball",
        user_type="FF",
        user_zone=2,
        ai_prediction="Fastball",
        pitch_tier="S",
        probabilistic=True,
        rng_seed=0.45,
    )
    assert outcome_out["result"] == "OUT"
    assert state_out.outs == 1
    assert state_out.lives == 3

    # 3. High rng_seed (0.80 >= 0.60) -> HIT
    state_hit, outcome_hit = resolve_arcade_pitch(
        state=state,
        user_category="Fastball",
        user_type="FF",
        user_zone=2,
        ai_prediction="Fastball",
        pitch_tier="S",
        probabilistic=True,
        rng_seed=0.80,
    )
    assert outcome_hit["result"] == "HIT"
    assert state_hit.on_1b == 1


def test_timing_meter_heat_anti_spam_and_timing_break() -> None:
    """Verify repeated same pitches heat up the timing meter, and switching pitches breaks timing."""
    state = ArcadeGameState(lives=3, balls=0, strikes=0)

    # Pitch 1: 4-Seam Fastball (FF)
    s1, _o1 = resolve_arcade_pitch(
        state=state,
        user_category="Fastball",
        user_type="FF",
        user_zone=2,
        ai_prediction="Offspeed",
        pitch_tier="S",
        probabilistic=True,
    )
    assert s1.consecutive_same_pitch == 1
    assert s1.timing_meter == 0

    # Pitch 2: 4-Seam Fastball again -> timing_meter increments to 1
    s2, _o2 = resolve_arcade_pitch(
        state=s1,
        user_category="Fastball",
        user_type="FF",
        user_zone=2,
        ai_prediction="Offspeed",
        pitch_tier="S",
        probabilistic=True,
    )
    assert s2.consecutive_same_pitch == 2
    assert s2.timing_meter == 1

    # Pitch 3: 4-Seam Fastball 3rd time -> timing_meter reaches 2 (Heated!)
    s3, _o3 = resolve_arcade_pitch(
        state=s2,
        user_category="Fastball",
        user_type="FF",
        user_zone=2,
        ai_prediction="Offspeed",
        pitch_tier="S",
        probabilistic=True,
    )
    assert s3.consecutive_same_pitch == 3
    assert s3.timing_meter == 2

    # Pitch 4: Fastball again and AI guesses right while timing_meter >= 2 -> CRUSH (HOME_RUN)!
    s4, o4 = resolve_arcade_pitch(
        state=s3,
        user_category="Fastball",
        user_type="FF",
        user_zone=2,
        ai_prediction="Fastball",
        pitch_tier="S",
        probabilistic=True,
    )
    assert o4["result"] == "HOME_RUN"
    assert "cronometrou seu arremesso repetido" in o4["message"]
    assert s4.lives < 3

    # Pitch 5: Switching to Splinker (FS) breaks the timing and resets meter to 0!
    s5, o5 = resolve_arcade_pitch(
        state=s4,
        user_category="Offspeed",
        user_type="FS",
        user_zone=8,
        ai_prediction="Breaking Ball",
        pitch_tier="A",
        probabilistic=True,
    )
    assert s5.timing_meter == 0
    assert s5.consecutive_same_pitch == 1
    assert o5["timing_broken"] is True
    assert "Quebrou o timing do rebatedor" in o5["message"]


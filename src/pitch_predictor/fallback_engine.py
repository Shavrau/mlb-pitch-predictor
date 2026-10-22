"""
Unified Fallback & Offline Inference Engine for MLB Pitch Predictor.

Provides a single, deterministic, baseball-authentic inference engine used across
all UI pages and client wrappers when the Databricks Serverless Model Serving
endpoint is scaling from zero, unreachable, or in offline mode.
"""

from __future__ import annotations

from typing import Any

from pitch_predictor.serving_client import (
    MLB_LEAGUE_PRIORS,
    derive_tactical_features,
    normalize_priors,
)


def predict_tactical_fallback(raw_inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Compute a deterministic, baseball-authentic tactical prediction when offline.

    Factors in:
    - Pitcher historical repertoire priors (Fastball, Breaking Ball, Offspeed).
    - Count leverage & count state (3-0 hitter count vs 0-2 pitcher ahead).
    - Platoon matchup (same_hand).
    - Baserunner scoring position leverage (RISP).

    Returns:
        Structured response dictionary matching `PitchServingClient.predict` contract.
    """
    features = derive_tactical_features(raw_inputs)
    count_str = str(features.get("count_state", "0-0"))
    same_hand = int(features.get("same_hand", 1))
    risp = int(features.get("scoring_position", 0))

    fb_prior = float(features.get("pitcher_fb_prior", MLB_LEAGUE_PRIORS[0]))
    bb_prior = float(features.get("pitcher_bb_prior", MLB_LEAGUE_PRIORS[1]))
    os_prior = float(features.get("pitcher_os_prior", MLB_LEAGUE_PRIORS[2]))

    fb_norm, bb_norm, os_norm = normalize_priors(fb_prior, bb_prior, os_prior)

    w_fb = fb_norm
    w_bb = bb_norm
    w_os = os_norm

    # Situational weighting based on MLB tactical counts
    if count_str in {"3-0", "3-1"}:
        # Behind in count: pitcher must throw strikes -> Fastball heavy
        w_fb += 0.40
    elif count_str in {"2-0", "1-0"}:
        w_fb += 0.25
    elif count_str == "0-0":
        # First pitch: establish zone with fastball
        w_fb += 0.15
    elif count_str in {"0-2", "1-2"}:
        # Pitcher ahead: putaway pitch with breaking ball / chase
        w_bb += 0.45
    elif count_str == "0-1":
        if same_hand == 1:
            w_bb += 0.35
        else:
            w_os += 0.25
    elif count_str in {"1-1", "2-2"}:
        if same_hand == 0:
            w_os += 0.40
        else:
            w_bb += 0.35
    elif count_str == "2-1":
        w_fb += 0.20

    if risp == 1:
        # Runners in scoring position: favor groundball-inducing offspeed/breaking
        w_os += 0.10
        w_bb += 0.05

    # Normalize weights to probabilities
    total_w = w_fb + w_bb + w_os
    if total_w <= 0:
        total_w = 1.0

    p_fb = round(max(0.01, w_fb / total_w), 4)
    p_bb = round(max(0.01, w_bb / total_w), 4)
    p_os = round(max(0.01, w_os / total_w), 4)

    # Re-normalize to exactly 1.0
    sum_p = p_fb + p_bb + p_os
    p_fb = round(p_fb / sum_p, 4)
    p_bb = round(p_bb / sum_p, 4)
    p_os = round(1.0 - p_fb - p_bb, 4)

    # Determine highest probability class
    if p_fb >= p_bb and p_fb >= p_os:
        pred = "Fastball"
    elif p_bb >= p_os:
        pred = "Breaking Ball"
    else:
        pred = "Offspeed"

    return {
        "status": "fallback",
        "prediction": pred,
        "probabilities": {
            "Fastball": p_fb,
            "Breaking Ball": p_bb,
            "Offspeed": p_os,
        },
        "latency_ms": 1.2,
        "features": features,
        "endpoint_name": "offline_fallback_engine",
        "attempts": 1,
        "is_fallback": True,
    }


def predict_multimodal_fallback(raw_inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Compute a deterministic multimodal prediction combining Statcast and biomechanics.
    """
    tac_res = predict_tactical_fallback(raw_inputs)
    sc_pred = tac_res["prediction"]
    sc_probs = tac_res["probabilities"]

    # Biomechanical pose tipping signals
    arm_slot = float(raw_inputs.get("arm_slot_angle", 48.0))
    elbow_flex = float(raw_inputs.get("elbow_flexion_angle", 135.0))
    rel_ht = float(raw_inputs.get("release_height", 0.70))

    # Tipping heuristic:
    # 1. Elbow flexion < 120° or arm slot < 40° -> strong Breaking Ball indicator
    # 2. Arm slot > 62° or release height < 0.66 -> Offspeed indicator
    # 3. High elbow extension >= 140° -> Fastball indicator
    vis_scores = {"Fastball": 0.33, "Breaking Ball": 0.33, "Offspeed": 0.34}
    if elbow_flex < 120.0 or arm_slot < 40.0:
        vis_scores["Breaking Ball"] += 0.50
    elif arm_slot > 60.0 or rel_ht < 0.66:
        vis_scores["Offspeed"] += 0.45
    elif elbow_flex >= 138.0:
        vis_scores["Fastball"] += 0.45

    tot_v = sum(vis_scores.values())
    vis_probs = {k: round(v / tot_v, 4) for k, v in vis_scores.items()}
    vis_pred = max(vis_probs, key=vis_probs.get)  # type: ignore[arg-type]

    # Multimodal fusion: 55% Statcast situational, 45% Vision tipping
    multi_probs = {}
    for cat in ("Fastball", "Breaking Ball", "Offspeed"):
        multi_probs[cat] = round(0.55 * sc_probs[cat] + 0.45 * vis_probs[cat], 4)

    tot_m = sum(multi_probs.values())
    multi_probs = {k: round(v / tot_m, 4) for k, v in multi_probs.items()}
    multi_pred = max(multi_probs, key=multi_probs.get)  # type: ignore[arg-type]

    if sc_pred == vis_pred:
        alignment = "AGREEMENT"
        summary = (
            f"Alinhamento total: tanto o contexto tático ({sc_pred}) quanto a postura "
            f"biomecânica ({vis_pred}) convergem para {multi_pred}."
        )
    elif vis_pred == multi_pred:
        alignment = "TIPPING_OVERRIDE"
        summary = (
            f"Alerta de Pitch Tipping: o modelo tático projetava {sc_pred}, mas a mecânica "
            f"corporal ({vis_pred}) denunciou {multi_pred}."
        )
    else:
        alignment = "DIVERGENT"
        summary = (
            f"Sinais divergentes entre situação tática ({sc_pred}) e postura ({vis_pred}). "
            f"A IA calculou maior probabilidade integrada para {multi_pred}."
        )

    return {
        "status": "fallback",
        "prediction": multi_pred,
        "probabilities": multi_probs,
        "statcast_prediction": sc_pred,
        "vision_prediction": vis_pred,
        "alignment_status": alignment,
        "summary_text": summary,
        "latency_ms": 1.5,
        "features": tac_res["features"],
        "endpoint_name": "offline_fallback_engine",
        "attempts": 1,
        "is_fallback": True,
    }


def simulate_12_counts_fallback(batch_records: list[dict[str, Any]]) -> dict[str, str]:
    """
    Generate 12-count predictions using the unified tactical fallback engine.
    """
    predictions_map: dict[str, str] = {}
    for rec in batch_records:
        res = predict_tactical_fallback(rec)
        count_key = str(rec.get("count_state", "0-0"))
        predictions_map[count_key] = res["prediction"]
    return predictions_map

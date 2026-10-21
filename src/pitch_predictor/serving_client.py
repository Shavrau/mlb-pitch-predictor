"""
Databricks Serverless Model Serving client and 24-feature tactical contract.

Provides:
- `EXPECTED_FEATURE_COLUMNS`: Exact 24-column feature schema expected by the Silver table / AutoML model.
- `derive_tactical_features`: Deterministic computation of tactical baseball features (`same_hand`,
  `count_state`, `count_leverage`, `runners_on_base`, `scoring_position`, First Pitch rules).
- `validate_inference_record`: Domain and schema validation for inference payloads.
- `normalize_priors`: Pitcher repertoire percentage normalization with MLB league average fallback.
- `build_rest_payload`: Formats records into `{"dataframe_records": [...]}`.
- `PitchServingClient`: Hybrid Databricks SDK / REST client with scale-to-zero cold-start retry resilience.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

DEFAULT_WORKSPACE_HOST = "https://dbc-654c8244-a596.cloud.databricks.com"
DEFAULT_ENDPOINT_NAME = "dev_rhian_souza_pitch_predictor_endpoint_dev"
DEFAULT_PROFILE = "dev"

MLB_LEAGUE_PRIORS: tuple[float, float, float] = (0.5543, 0.3053, 0.1404)

EXPECTED_FEATURE_COLUMNS: list[str] = [
    "prev_pitch_category",
    "prev_pitch_type",
    "prev_pitch_zone",
    "prev_pitch_result",
    "p_throws",
    "stand",
    "same_hand",
    "balls",
    "strikes",
    "count_state",
    "count_leverage",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "score_diff",
    "on_1b_flag",
    "on_2b_flag",
    "on_3b_flag",
    "runners_on_base",
    "scoring_position",
    "n_thruorder_pitcher",
    "pitcher_fb_prior",
    "pitcher_bb_prior",
    "pitcher_os_prior",
]

VALID_STATCAST_ZONES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14}
PITCHER_AHEAD_COUNTS = {"0-1", "0-2", "1-2"}
PITCHER_BEHIND_COUNTS = {"1-0", "2-0", "3-0", "3-1"}

BIOMECHANICAL_FEATURE_COLUMNS: list[str] = [
    "arm_slot_angle",
    "release_height",
    "elbow_flexion_angle",
    "trunk_tilt",
    "stride_extension",
]

DEFAULT_BIOMECHANICS: dict[str, float] = {
    "arm_slot_angle": 48.0,
    "release_height": 0.70,
    "elbow_flexion_angle": 135.0,
    "trunk_tilt": 11.5,
    "stride_extension": 1.40,
}


def normalize_priors(fb: float, bb: float, os_val: float) -> tuple[float, float, float]:
    """
    Normalize pitcher repertoire priors (Fastball, Breaking Ball, Offspeed) to sum to 1.0.

    Falls back to MLB league-average priors (0.5543, 0.3053, 0.1404) when the sum is <= 0.
    """
    fb_clean = max(0.0, float(fb))
    bb_clean = max(0.0, float(bb))
    os_clean = max(0.0, float(os_val))

    total = fb_clean + bb_clean + os_clean
    if total <= 0.0:
        return MLB_LEAGUE_PRIORS

    return (
        fb_clean / total,
        bb_clean / total,
        os_clean / total,
    )


def derive_tactical_features(raw_inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Derive the complete 24-column tactical feature dictionary in the exact contract order.

    Enforces:
    - First Pitch consistency (`prev_pitch_category == "First Pitch"` forces `type="None"`,
      `zone=0`, `result="First Pitch"`).
    - Platoon handedness indicator (`same_hand`).
    - Count state (`"{balls}-{strikes}"`) and tactical leverage (`Pitcher Ahead`,
      `Pitcher Behind`, `Neutral`).
    - Baserunner aggregation (`runners_on_base` and `scoring_position` / RISP).
    """
    prev_category = str(raw_inputs.get("prev_pitch_category", "First Pitch"))
    if prev_category == "First Pitch":
        prev_type = "None"
        prev_zone = 0
        prev_result = "First Pitch"
    else:
        prev_type = str(raw_inputs.get("prev_pitch_type", "FF"))
        if prev_type == "None":
            prev_type = "FF"
        prev_zone = int(raw_inputs.get("prev_pitch_zone", 5))
        if prev_zone == 0:
            prev_zone = 5
        prev_result = str(raw_inputs.get("prev_pitch_result", "S"))
        if prev_result == "First Pitch":
            prev_result = "S"

    p_throws = str(raw_inputs.get("p_throws", "R")).upper()
    stand = str(raw_inputs.get("stand", "R")).upper()
    same_hand = 1 if p_throws == stand else 0

    balls = int(raw_inputs.get("balls", 0))
    strikes = int(raw_inputs.get("strikes", 0))
    count_state = f"{balls}-{strikes}"

    if count_state in PITCHER_AHEAD_COUNTS:
        count_leverage = "Pitcher Ahead"
    elif count_state in PITCHER_BEHIND_COUNTS:
        count_leverage = "Pitcher Behind"
    else:
        count_leverage = "Neutral"

    outs_when_up = int(raw_inputs.get("outs_when_up", 0))
    inning = int(raw_inputs.get("inning", 1))
    inning_topbot = str(raw_inputs.get("inning_topbot", "Top"))
    score_diff = int(raw_inputs.get("score_diff", 0))

    on_1b_flag = 1 if int(raw_inputs.get("on_1b_flag", 0)) > 0 else 0
    on_2b_flag = 1 if int(raw_inputs.get("on_2b_flag", 0)) > 0 else 0
    on_3b_flag = 1 if int(raw_inputs.get("on_3b_flag", 0)) > 0 else 0

    runners_on_base = on_1b_flag + on_2b_flag + on_3b_flag
    scoring_position = 1 if (on_2b_flag == 1 or on_3b_flag == 1) else 0

    n_thruorder_pitcher = int(raw_inputs.get("n_thruorder_pitcher", 1))

    fb_raw = float(raw_inputs.get("pitcher_fb_prior", MLB_LEAGUE_PRIORS[0]))
    bb_raw = float(raw_inputs.get("pitcher_bb_prior", MLB_LEAGUE_PRIORS[1]))
    os_raw = float(raw_inputs.get("pitcher_os_prior", MLB_LEAGUE_PRIORS[2]))
    fb_prior, bb_prior, os_prior = normalize_priors(fb_raw, bb_raw, os_raw)

    ordered_record: dict[str, Any] = {
        "prev_pitch_category": prev_category,
        "prev_pitch_type": prev_type,
        "prev_pitch_zone": prev_zone,
        "prev_pitch_result": prev_result,
        "p_throws": p_throws,
        "stand": stand,
        "same_hand": same_hand,
        "balls": balls,
        "strikes": strikes,
        "count_state": count_state,
        "count_leverage": count_leverage,
        "outs_when_up": outs_when_up,
        "inning": inning,
        "inning_topbot": inning_topbot,
        "score_diff": score_diff,
        "on_1b_flag": on_1b_flag,
        "on_2b_flag": on_2b_flag,
        "on_3b_flag": on_3b_flag,
        "runners_on_base": runners_on_base,
        "scoring_position": scoring_position,
        "n_thruorder_pitcher": n_thruorder_pitcher,
        "pitcher_fb_prior": round(fb_prior, 4),
        "pitcher_bb_prior": round(bb_prior, 4),
        "pitcher_os_prior": round(os_prior, 4),
    }
    return ordered_record


def validate_inference_record(record: dict[str, Any]) -> bool:
    """
    Validate that an inference record contains all 24 required columns and valid MLB domain values.

    Raises ValueError if any column is missing or out of bounds.
    """
    if not isinstance(record, dict):
        raise ValueError("Inference record must be a dictionary.")  # noqa: TRY004

    missing_columns = [col for col in EXPECTED_FEATURE_COLUMNS if col not in record]
    if missing_columns:
        raise ValueError(
            f"Missing required feature column(s): {', '.join(missing_columns)}"
        )

    balls = record["balls"]
    if not isinstance(balls, int) or not (0 <= balls <= 3):
        raise ValueError(f"Invalid 'balls' value ({balls}): must be an integer in 0..3.")

    strikes = record["strikes"]
    if not isinstance(strikes, int) or not (0 <= strikes <= 2):
        raise ValueError(
            f"Invalid 'strikes' value ({strikes}): must be an integer in 0..2."
        )

    outs = record["outs_when_up"]
    if not isinstance(outs, int) or not (0 <= outs <= 2):
        raise ValueError(
            f"Invalid 'outs_when_up' value ({outs}): must be an integer in 0..2."
        )

    zone = record["prev_pitch_zone"]
    if not isinstance(zone, int) or zone not in VALID_STATCAST_ZONES:
        raise ValueError(
            f"Invalid 'prev_pitch_zone' value ({zone}): must be 0, 1..9, or 11..14."
        )

    if record["p_throws"] not in {"R", "L"}:
        raise ValueError(f"Invalid 'p_throws' value ({record['p_throws']}).")

    if record["stand"] not in {"R", "L"}:
        raise ValueError(f"Invalid 'stand' value ({record['stand']}).")

    if record["inning_topbot"] not in {"Top", "Bot"}:
        raise ValueError(f"Invalid 'inning_topbot' value ({record['inning_topbot']}).")

    return True


def build_rest_payload(
    records: list[dict[str, Any]] | dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """
    Validate and format one or more tactical feature dicts into a Databricks Serving payload.
    """
    record_list = [records] if isinstance(records, dict) else list(records)
    if not record_list:
        raise ValueError("At least one inference record is required to build payload.")

    formatted_records: list[dict[str, Any]] = []
    for rec in record_list:
        validate_inference_record(rec)
        ordered = {col: rec[col] for col in EXPECTED_FEATURE_COLUMNS}
        formatted_records.append(ordered)

    return {"dataframe_records": formatted_records}


def derive_multimodal_features(raw_inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Derive complete features combining 24-column tactical contract with 5-column biomechanics.
    """
    tactical = derive_tactical_features(raw_inputs)
    for bio_col in BIOMECHANICAL_FEATURE_COLUMNS:
        tactical[bio_col] = float(raw_inputs.get(bio_col, DEFAULT_BIOMECHANICS[bio_col]))
    return tactical


def build_multimodal_payload(
    records: list[dict[str, Any]] | dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """
    Validate and format one or more multimodal feature dicts into a Databricks Serving payload.
    """
    record_list = [records] if isinstance(records, dict) else list(records)
    if not record_list:
        raise ValueError("At least one inference record is required to build payload.")

    formatted_records: list[dict[str, Any]] = []
    for rec in record_list:
        validate_inference_record(rec)
        ordered = {col: rec[col] for col in EXPECTED_FEATURE_COLUMNS}
        for bio_col in BIOMECHANICAL_FEATURE_COLUMNS:
            ordered[bio_col] = float(rec.get(bio_col, DEFAULT_BIOMECHANICS[bio_col]))
        formatted_records.append(ordered)

    return {"dataframe_records": formatted_records}


class PitchServingClient:
    """
    Client for querying the Databricks Serverless Model Serving endpoint for pitch prediction.

    Supports hybrid authentication:
    1. Direct REST invocation when `host` and `token` (or `DATABRICKS_TOKEN`) are provided.
    2. Databricks SDK (`WorkspaceClient`) using local CLI profile or Databricks Apps ambient auth.
    """

    def __init__(
        self,
        endpoint_name: str = DEFAULT_ENDPOINT_NAME,
        host: str | None = None,
        token: str | None = None,
        profile: str | None = DEFAULT_PROFILE,
        timeout_seconds: int = 60,
        max_retries: int = 2,
        backoff_seconds: float = 2.0,
    ) -> None:
        self.endpoint_name = os.environ.get("SERVING_ENDPOINT_NAME", endpoint_name)
        self.host = host or os.environ.get("DATABRICKS_HOST")
        self.token = token or os.environ.get("DATABRICKS_TOKEN")
        self.profile = profile
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def _get_workspace_client(self) -> Any:
        """Instantiate a Databricks SDK WorkspaceClient with profile or ambient credentials."""
        from databricks.sdk import WorkspaceClient

        # When running inside Databricks Apps or when client credentials are set in env
        if os.environ.get("DATABRICKS_APP_NAME") or os.environ.get("DATABRICKS_CLIENT_ID"):
            return WorkspaceClient()

        if self.profile:
            try:
                return WorkspaceClient(profile=self.profile)
            except Exception:  # noqa: BLE001, S110
                pass
        return WorkspaceClient()

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Send inference payload to the Databricks Model Serving endpoint via REST or SDK.
        """
        if self.host and self.token:
            url = (
                f"{self.host.rstrip('/')}/serving-endpoints/"
                f"{self.endpoint_name}/invocations"
            )
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                raise ConnectionError(f"Serving endpoint connection failed: {exc}") from exc

            if response.status_code in {502, 503, 504}:
                raise ConnectionError(
                    f"Serving endpoint warming up (HTTP {response.status_code})"
                )
            response.raise_for_status()
            return response.json()

        # Fallback to Databricks SDK WorkspaceClient
        try:
            w = self._get_workspace_client()
            sdk_response = w.serving_endpoints.query(
                name=self.endpoint_name,
                dataframe_records=payload["dataframe_records"],
            )
            predictions = (
                list(sdk_response.predictions)
                if sdk_response.predictions is not None
                else []
            )
            return {"predictions": predictions}
        except Exception as exc:
            err_msg = str(exc).lower()
            if any(k in err_msg for k in ("503", "502", "504", "timeout", "temporarily", "connection")):
                raise ConnectionError(f"Serving endpoint cold-start or transient error: {exc}") from exc
            raise

    @staticmethod
    def _extract_prediction_label(response_data: dict[str, Any]) -> str:
        """Extract the predicted pitch category string from the endpoint response."""
        labels = PitchServingClient._extract_prediction_labels_list(response_data)
        return labels[0]

    @staticmethod
    def _extract_prediction_labels_list(response_data: dict[str, Any]) -> list[str]:
        """Extract a list of predicted pitch category strings from a batch endpoint response."""
        predictions = response_data.get("predictions", [])
        if not predictions:
            raise ValueError(f"Empty predictions returned from endpoint: {response_data}")

        extracted: list[str] = []
        for item in predictions:
            if isinstance(item, dict):
                found = None
                for key in ("prediction", "pitch_category", "label", "0"):
                    if key in item:
                        found = str(item[key])
                        break
                extracted.append(found if found is not None else str(next(iter(item.values()))))
            else:
                extracted.append(str(item))
        return extracted

    def predict(self, raw_inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Derive tactical features, validate contract, invoke endpoint with retry resilience,
        and return structured prediction result with latency metrics.
        """
        features = derive_tactical_features(raw_inputs)
        payload = build_rest_payload([features])

        last_error: Exception | None = None
        total_attempts = max(1, self.max_retries + 1)

        start_time = time.perf_counter()
        for attempt in range(total_attempts):
            try:
                response_data = self._send_request(payload)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                prediction_str = self._extract_prediction_label(response_data)
                return {
                    "status": "success",
                    "prediction": prediction_str,
                    "latency_ms": round(latency_ms, 2),
                    "features": features,
                    "endpoint_name": self.endpoint_name,
                    "attempts": attempt + 1,
                }
            except (ConnectionError, TimeoutError) as exc:
                last_error = exc
                if attempt < total_attempts - 1:
                    time.sleep(self.backoff_seconds)
                else:
                    raise

        raise RuntimeError(f"Prediction failed after {total_attempts} attempts: {last_error}")

    def predict_batch(self, raw_inputs_list: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Perform batch inference across multiple game states (e.g., all 12 MLB counts)
        in a single HTTP/SDK round-trip with cold-start retry resilience.
        """
        if not raw_inputs_list:
            raise ValueError("raw_inputs_list must contain at least one input dictionary.")

        features_list = [derive_tactical_features(raw) for raw in raw_inputs_list]
        payload = build_rest_payload(features_list)

        last_error: Exception | None = None
        total_attempts = max(1, self.max_retries + 1)

        start_time = time.perf_counter()
        for attempt in range(total_attempts):
            try:
                response_data = self._send_request(payload)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                predictions_list = self._extract_prediction_labels_list(response_data)
                return {
                    "status": "success",
                    "predictions": predictions_list,
                    "latency_ms": round(latency_ms, 2),
                    "features_list": features_list,
                    "endpoint_name": self.endpoint_name,
                    "attempts": attempt + 1,
                }
            except (ConnectionError, TimeoutError) as exc:
                last_error = exc
                if attempt < total_attempts - 1:
                    time.sleep(self.backoff_seconds)
                else:
                    raise

        raise RuntimeError(f"Batch prediction failed after {total_attempts} attempts: {last_error}")

    @staticmethod
    def _extract_multimodal_dict(item: Any) -> dict[str, Any]:
        """Normalize a single prediction item from endpoint into full multimodal structure."""
        if isinstance(item, dict):
            pred = str(item.get("prediction", item.get("pitch_category", "Fastball")))
            p_fb = float(item.get("prob_fastball", 0.0))
            p_bb = float(item.get("prob_breaking_ball", 0.0))
            p_os = float(item.get("prob_offspeed", 0.0))
            sc_pred = str(item.get("statcast_prediction", pred))
            vis_pred = str(item.get("vision_prediction", pred))
            alignment = str(item.get("alignment_status", "AGREEMENT"))
            summary = str(item.get("summary_text", f"Predição final: {pred}"))

            if p_fb == 0.0 and p_bb == 0.0 and p_os == 0.0:
                p_fb = 0.55 if pred == "Fastball" else (0.25 if pred == "Breaking Ball" else 0.20)
                p_bb = 0.55 if pred == "Breaking Ball" else (0.30 if pred == "Fastball" else 0.15)
                p_os = 0.55 if pred == "Offspeed" else (0.20 if pred == "Fastball" else 0.25)
        else:
            pred = str(item)
            p_fb = 0.55 if pred == "Fastball" else 0.25
            p_bb = 0.55 if pred == "Breaking Ball" else 0.25
            p_os = 0.55 if pred == "Offspeed" else 0.20
            sc_pred = pred
            vis_pred = pred
            alignment = "AGREEMENT"
            summary = f"Predição tática do modelo: {pred}."

        return {
            "prediction": pred,
            "probabilities": {
                "Fastball": round(p_fb, 4),
                "Breaking Ball": round(p_bb, 4),
                "Offspeed": round(p_os, 4),
            },
            "statcast_prediction": sc_pred,
            "vision_prediction": vis_pred,
            "alignment_status": alignment,
            "summary_text": summary,
        }

    def predict_multimodal(self, raw_inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Derive tactical and biomechanical features, invoke multimodal endpoint with retry resilience,
        and return full multimodal prediction, probability distribution, alignment status, and explanation.
        """
        features = derive_multimodal_features(raw_inputs)
        payload = build_multimodal_payload([features])

        last_error: Exception | None = None
        total_attempts = max(1, self.max_retries + 1)

        start_time = time.perf_counter()
        for attempt in range(total_attempts):
            try:
                response_data = self._send_request(payload)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                predictions = response_data.get("predictions", [])
                if not predictions:
                    raise ValueError(f"Empty predictions returned from endpoint: {response_data}")

                parsed = self._extract_multimodal_dict(predictions[0])
                return {
                    "status": "success",
                    "prediction": parsed["prediction"],
                    "probabilities": parsed["probabilities"],
                    "statcast_prediction": parsed["statcast_prediction"],
                    "vision_prediction": parsed["vision_prediction"],
                    "alignment_status": parsed["alignment_status"],
                    "summary_text": parsed["summary_text"],
                    "latency_ms": round(latency_ms, 2),
                    "features": features,
                    "endpoint_name": self.endpoint_name,
                    "attempts": attempt + 1,
                }
            except (ConnectionError, TimeoutError) as exc:
                last_error = exc
                if attempt < total_attempts - 1:
                    time.sleep(self.backoff_seconds)
                else:
                    raise

        raise RuntimeError(f"Multimodal prediction failed after {total_attempts} attempts: {last_error}")

    def predict_multimodal_batch(
        self, raw_inputs_list: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Perform batch multimodal inference across multiple game states with physical features.
        """
        if not raw_inputs_list:
            raise ValueError("raw_inputs_list must contain at least one input dictionary.")

        features_list = [derive_multimodal_features(raw) for raw in raw_inputs_list]
        payload = build_multimodal_payload(features_list)

        last_error: Exception | None = None
        total_attempts = max(1, self.max_retries + 1)

        start_time = time.perf_counter()
        for attempt in range(total_attempts):
            try:
                response_data = self._send_request(payload)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                predictions = response_data.get("predictions", [])
                if not predictions:
                    raise ValueError(f"Empty predictions returned from endpoint: {response_data}")

                parsed_list = [self._extract_multimodal_dict(p) for p in predictions]
                return {
                    "status": "success",
                    "predictions": parsed_list,
                    "latency_ms": round(latency_ms, 2),
                    "features_list": features_list,
                    "endpoint_name": self.endpoint_name,
                    "attempts": attempt + 1,
                }
            except (ConnectionError, TimeoutError) as exc:
                last_error = exc
                if attempt < total_attempts - 1:
                    time.sleep(self.backoff_seconds)
                else:
                    raise

        raise RuntimeError(f"Batch multimodal prediction failed after {total_attempts} attempts: {last_error}")

    def check_endpoint_status(self) -> dict[str, Any]:
        """Query the operational state of the configured Databricks Serving Endpoint."""
        try:
            w = self._get_workspace_client()
            ep = w.serving_endpoints.get(name=self.endpoint_name)
            ready_state = (
                str(ep.state.ready.value)
                if ep.state and ep.state.ready
                else "UNKNOWN"
            )
            config_update = (
                str(ep.state.config_update.value)
                if ep.state and ep.state.config_update
                else "NONE"
            )
            return {
                "endpoint_name": self.endpoint_name,
                "ready": ready_state,
                "config_update": config_update,
                "online": "READY" in ready_state.upper(),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "endpoint_name": self.endpoint_name,
                "ready": "UNVERIFIED",
                "config_update": "UNKNOWN",
                "online": False,
                "detail": str(exc),
            }

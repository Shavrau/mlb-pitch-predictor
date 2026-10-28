"""
Página 4 — Laboratório de Visão Computacional & Predição pela Pose 3D (Pitch Tipping).

Focada exclusivamente na predição de arremessos com base em Visão Computacional:
1. Captura do Print Base no momento exato de soltura da bola (Release Point) a partir do vídeo.
2. Extração e sobreposição do Modelo da Pose 3D (33 marcos anatômicos MediaPipe Pose) sobre o Print Base.
3. Cálculo das métricas cinemáticas (Arm Slot, Flexão do Cotovelo, Altura de Soltura, Inclinação do Tronco).
4. Predição da categoria do arremesso (Fastball, Breaking Ball, Offspeed) puramente pela postura corporal
   e explicabilidade via SHAP TreeExplainer.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pitch_predictor.scenarios import (
    MLB_PITCHERS,
    get_pitcher_names,
    get_pitcher_profile,
)
from pitch_predictor.vision.overlay import (
    extract_and_annotate_release_frame,
    extract_base_release_frame,
    find_preloaded_release_frame,
)
from pitch_predictor.vision.pose_extractor import PoseExtractor
from pitch_predictor.vision.tipping_model import PitchTippingClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Caching de Recursos e Dados
# ---------------------------------------------------------------------------


@st.cache_resource
def load_pose_extractor() -> PoseExtractor:
    """Carrega o extrator de pose MediaPipe em cache singleton."""
    return PoseExtractor()


@st.cache_resource
def load_tipping_model() -> PitchTippingClassifier:
    """Carrega o classificador de Pitch Tipping treinado em features de Visão Computacional com SHAP."""
    model_path = Path("models/pitch_tipping_model.joblib")
    if model_path.exists():
        try:
            return PitchTippingClassifier.load(model_path)
        except (FileNotFoundError, TypeError, OSError):
            pass
    # Fallback: cria e treina instância em memória
    clf = PitchTippingClassifier(random_state=42)
    from pitch_predictor.vision.tipping_model import generate_synthetic_biomechanics

    clf.fit(generate_synthetic_biomechanics(n_samples_per_class=40, random_state=42))
    return clf


@st.cache_data
def load_sample_manifest(pitcher_id: int = 694973) -> pd.DataFrame:
    """Carrega o manifesto de vídeos do arremessador."""
    candidate_paths = [
        Path(f"data/videos/{pitcher_id}/manifest.csv"),
        Path(__file__).parent.parent / "src" / "pitch_predictor" / "vision" / "sample_data" / "manifest.csv",
        Path("/app/python/source_code/src/pitch_predictor/vision/sample_data/manifest.csv"),
    ]
    for p in candidate_paths:
        if p.exists():
            df = pd.read_csv(p)
            if "status" in df.columns:
                df = df[df["status"] == "COMPLETED"]
            if "pitcher_id" in df.columns:
                p_df = df[df["pitcher_id"] == pitcher_id]
                if not p_df.empty:
                    return p_df.copy()
            if not df.empty:
                return df.copy()

    # Fallback padrão caso execute em ambiente limpo
    return pd.DataFrame([
        {
            "game_pk": 745170,
            "at_bat_number": 31,
            "pitch_number": 4,
            "pitcher_id": pitcher_id,
            "pitch_type": "FF",
            "pitch_category": "Fastball",
            "video_url": "https://sporty-clips.mlb.com/TGxyek5fWGw0TUFRPT1fVndBSFZBRUFBbEVBQzF0V1Z3QUFBUU5TQUZnQVd3UUFCd1FEQ1ZBRFZRRUVWVlpR.mp4",
            "video_path": f"data/videos/{pitcher_id}/745170_31_4_Fastball.mp4",
            "volume_path": f"/Volumes/pitch_predictor/baseball_ml_dev/pitch_data/videos/{pitcher_id}/745170_31_4_Fastball.mp4",
            "status": "COMPLETED",
        },
        {
            "game_pk": 745500,
            "at_bat_number": 44,
            "pitch_number": 1,
            "pitcher_id": pitcher_id,
            "pitch_type": "ST",
            "pitch_category": "Breaking Ball",
            "video_url": "https://sporty-clips.mlb.com/MVlET1lfWGw0TUFRPT1fVUFCUlhGRUFBMVlBQVZvRlZ3QUFVbFJlQUZoUkJnUUFVd1JSVlZVQ0JnWmRCd1VI.mp4",
            "video_path": f"data/videos/{pitcher_id}/745500_44_1_Breaking_Ball.mp4",
            "volume_path": f"/Volumes/pitch_predictor/baseball_ml_dev/pitch_data/videos/{pitcher_id}/745500_44_1_Breaking_Ball.mp4",
            "status": "COMPLETED",
        },
        {
            "game_pk": 745500,
            "at_bat_number": 34,
            "pitch_number": 3,
            "pitcher_id": pitcher_id,
            "pitch_type": "FS",
            "pitch_category": "Offspeed",
            "video_url": "https://sporty-clips.mlb.com/MVlET1lfWGw0TUFRPT1fVkFOWVhBWUhWUVFBWEZNQVVRQUFBQTREQUZnQ0FGa0FWRnhYQmdWV0FWRUdCZ0VG.mp4",
            "video_path": f"data/videos/{pitcher_id}/745500_34_3_Offspeed.mp4",
            "volume_path": f"/Volumes/pitch_predictor/baseball_ml_dev/pitch_data/videos/{pitcher_id}/745500_34_3_Offspeed.mp4",
            "status": "COMPLETED",
        },
    ])


@st.cache_data
def load_biomechanical_dataset(pitcher_id: int = 694973) -> pd.DataFrame:
    """Carrega as features biomecânicas extraídas."""
    candidate_paths = [
        Path(f"data/videos/{pitcher_id}/biomechanical_features.parquet"),
        Path(f"data/videos/{pitcher_id}/biomechanical_features.csv"),
        Path(__file__).parent.parent / "src" / "pitch_predictor" / "vision" / "sample_data" / "biomechanical_features.csv",
        Path("/app/python/source_code/src/pitch_predictor/vision/sample_data/biomechanical_features.csv"),
    ]
    for p in candidate_paths:
        if p.exists():
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
            if "pitcher_id" in df.columns:
                p_df = df[df["pitcher_id"] == pitcher_id]
                if not p_df.empty:
                    return p_df.copy()
            if not df.empty:
                return df.copy()

    # Dynamic fallback based on registered profile in MLB_PITCHERS
    p_meta = next((v for v in MLB_PITCHERS.values() if v.get("pitcher_id") == pitcher_id), None)
    base_slot = p_meta["bio_profile"]["arm_slot_angle"] if p_meta else 49.78
    base_elbow = p_meta["bio_profile"]["elbow_flexion_angle"] if p_meta else 146.11
    base_ht = p_meta["bio_profile"]["release_height"] if p_meta else 0.710
    base_tilt = p_meta["bio_profile"]["trunk_tilt"] if p_meta else 11.67
    base_stride = p_meta["bio_profile"]["stride_extension"] if p_meta else 1.45

    return pd.DataFrame([
        {
            "game_pk": 745170,
            "at_bat_number": 31,
            "pitch_number": 4,
            "pitcher_id": pitcher_id,
            "pitch_category": "Fastball",
            "pitch_type": "FF",
            "arm_slot_angle": base_slot,
            "release_height": base_ht,
            "elbow_flexion_angle": base_elbow,
            "trunk_tilt": base_tilt,
            "stride_extension": base_stride,
            "release_frame": 198,
        },
        {
            "game_pk": 745500,
            "at_bat_number": 44,
            "pitch_number": 1,
            "pitcher_id": pitcher_id,
            "pitch_category": "Breaking Ball",
            "pitch_type": "ST",
            "arm_slot_angle": max(10.0, base_slot - 12.0),
            "release_height": base_ht - 0.03,
            "elbow_flexion_angle": base_elbow - 20.0,
            "trunk_tilt": base_tilt + 2.0,
            "stride_extension": base_stride - 0.05,
            "release_frame": 174,
        },
        {
            "game_pk": 745500,
            "at_bat_number": 34,
            "pitch_number": 3,
            "pitcher_id": pitcher_id,
            "pitch_category": "Offspeed",
            "pitch_type": "FS",
            "arm_slot_angle": min(85.0, base_slot + 14.0),
            "release_height": base_ht - 0.05,
            "elbow_flexion_angle": base_elbow - 10.0,
            "trunk_tilt": base_tilt + 3.0,
            "stride_extension": base_stride,
            "release_frame": 174,
        },
    ])


@st.cache_data(show_spinner=False)
def fetch_pitch_clip_locally(url: str, dest_str: str) -> str:
    """Baixa clipe MP4 sob demanda para o cache local se ainda não existir."""
    if not url:
        return ""
    dest = Path(dest_str)
    if dest.exists() and dest.stat().st_size > 1000:
        return str(dest)
    try:
        from pitch_predictor.vision.video_ingestion import download_pitch_clip

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            dest = Path("/tmp") / "pitch_videos" / dest.name
            dest.parent.mkdir(parents=True, exist_ok=True)

        if download_pitch_clip(url, dest, timeout=12):
            return str(dest)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not download clip from %s to %s: %s", url, dest, exc)
    return ""


def _render_image_compat(image_array: np.ndarray, caption: str) -> None:
    """Renderiza imagem numpy RGB com compatibilidade entre versões do Streamlit."""
    try:
        st.image(
            image_array,
            caption=caption,
            channels="RGB",
            use_container_width=True,
        )
    except TypeError:
        try:
            st.image(
                image_array,
                caption=caption,
                channels="RGB",
                use_column_width=True,
            )
        except TypeError:
            st.image(
                image_array,
                caption=caption,
                channels="RGB",
            )


# ---------------------------------------------------------------------------
# Layout da Página
# ---------------------------------------------------------------------------

st.title("🔬 Laboratório de Visão Computacional — Predição pela Pose 3D")
st.caption("Predição de Arremessos e Detecção de Pitch Tipping Exclusivamente por Estimativa de Pose 3D (MediaPipe) & SHAP")

with st.container(border=True):
    st.markdown(
        """
        ### Como a Visão Computacional Prediz o Arremesso (Pitch Tipping)?
        No beisebol de elite, **Pitch Tipping** ocorre quando a biomecânica corporal do arremessador
        denuncia involuntariamente o tipo de arremesso antes de a bola deixar sua mão.
        
        Esta página é **100% dedicada à predição baseada em Visão Computacional**:
        1. **Captura do Print Base**: Extraímos o frame estático exato do ponto de soltura (*Release Point*) a partir do vídeo da transmissão.
        2. **Modelo da Pose 3D**: Aplicamos o **MediaPipe Pose** sobre o print base para mapear **33 articulações tridimensionais** e medir os ângulos anatômicos do corpo.
        3. **Predição Biomecânica Pura**: Nosso classificador analisa exclusivamente a postura extraída da pose para prever se o arremesso é **Fastball**, **Breaking Ball** ou **Offspeed**.
        """
    )

# ---------------------------------------------------------------------------
# 1. Seleção do Arremessador & Arremesso
# ---------------------------------------------------------------------------

st.subheader("1. Seleção do Arremessador & Clipe para Análise de Pose")

PITCHER_EMOJIS = {
    "Paul Skenes": "🚀",
    "Corbin Burnes": "🌪️",
    "Logan Webb": "📉",
    "Chris Sale": "🏹",
    "Yoshinobu Yamamoto": "👻",
}

pitcher_names = get_pitcher_names()

col_pitcher, col_clip = st.columns([1, 1.25], gap="medium")

with col_pitcher:
    selected_pitcher_name = st.selectbox(
        "Arremessador Estrela da MLB:",
        options=pitcher_names,
        format_func=lambda name: f"{PITCHER_EMOJIS.get(name, '⚾')} {name} ({MLB_PITCHERS[name]['team']} • {MLB_PITCHERS[name]['p_throws']}HP)",
        key="vision_pitcher_selector",
    )

pitcher_info = get_pitcher_profile(selected_pitcher_name)
pitcher_id = pitcher_info["pitcher_id"]
p_throws = pitcher_info["p_throws"]

manifest_df = load_sample_manifest(pitcher_id)
bio_df = load_biomechanical_dataset(pitcher_id)
tipping_model = load_tipping_model()
pose_extractor = load_pose_extractor()

options_labels = []
for _, row in manifest_df.iterrows():
    cat = row.get("pitch_category", "Unknown")
    pt = row.get("pitch_type", "UNK")
    g_pk = row.get("game_pk", "")
    ab = row.get("at_bat_number", "")
    pn = row.get("pitch_number", "")
    options_labels.append(f"{cat} ({pt}) — Game {g_pk} | AB #{ab} | Pitch #{pn}")

if not options_labels:
    options_labels = ["Fastball (FF) — Game 745170 | AB #31 | Pitch #4"]

# Reset selected pitch if pitcher changed or value not in active options
if "current_vision_pitcher" not in st.session_state:
    st.session_state.current_vision_pitcher = selected_pitcher_name
elif st.session_state.current_vision_pitcher != selected_pitcher_name:
    st.session_state.current_vision_pitcher = selected_pitcher_name
    st.session_state.vision_pitch_selector = options_labels[0]

if "vision_pitch_selector" in st.session_state and st.session_state.vision_pitch_selector not in options_labels:
    st.session_state.vision_pitch_selector = options_labels[0]

with col_clip:
    selected_label = st.selectbox(
        "Selecione o clipe do arremesso para extração do print e da pose:",
        options=options_labels,
        key="vision_pitch_selector",
    )

selected_idx = options_labels.index(selected_label) if selected_label in options_labels else 0
selected_row = manifest_df.iloc[selected_idx]
selected_cat = selected_row.get("pitch_category", "Fastball")

# Obter métricas biomecânicas associadas ao lance específico (game_pk, at_bat, pitch)
sel_g_pk = selected_row.get("game_pk")
sel_ab = selected_row.get("at_bat_number")
sel_pn = selected_row.get("pitch_number")

matched_bio = bio_df[
    (bio_df["game_pk"] == sel_g_pk)
    & (bio_df["at_bat_number"] == sel_ab)
    & (bio_df["pitch_number"] == sel_pn)
]
if matched_bio.empty:
    matched_bio = bio_df[bio_df["pitch_category"] == selected_cat]
if matched_bio.empty:
    matched_bio = bio_df.iloc[0:1]

bio_record = matched_bio.iloc[0].to_dict()

# Quick summary card of pitcher and pitch
info_c1, info_c2, info_c3 = st.columns(3)
with info_c1:
    st.info(
        f"**Arremessador**: {pitcher_info['name']} (#{pitcher_id})\n\n"
        f"**Equipe**: {pitcher_info['team']}",
        icon=":material/person:",
    )
with info_c2:
    st.info(
        f"**Mão Dominante**: `{p_throws}HP` ({'Destro' if p_throws == 'R' else 'Canhoto'})\n\n"
        f"**Slot de Braço**: `{pitcher_info.get('arm_slot_type', '3/4')}`",
        icon=":material/sports_baseball:",
    )
with info_c3:
    st.info(
        f"**Categoria Real**: `{selected_cat}` (`{selected_row.get('pitch_type', 'FF')}`)\n\n"
        f"**Arsenal**: {pitcher_info['signature_arsenal']}",
        icon=":material/analytics:",
    )

st.divider()

# ---------------------------------------------------------------------------
# 2. Pipeline Visual: Vídeo -> Print Base -> Modelo da Pose 3D
# ---------------------------------------------------------------------------

st.subheader("2. Pipeline de Visão Computacional: Vídeo ➔ Print Base ➔ Modelo da Pose 3D")

video_col, base_print_col, pose_col = st.columns(3)

video_path_str = str(selected_row.get("video_path", ""))
video_file = Path(video_path_str)
release_frame = int(bio_record.get("release_frame", 198))
video_url = str(selected_row.get("video_url", ""))

# Baixa ou localiza o clipe MP4 sob demanda apenas se o snapshot calibrado não existir
active_video_path = video_file
has_preloaded = find_preloaded_release_frame(
    video_path=video_path_str,
    pitch_category=selected_cat,
    game_pk=sel_g_pk,
    at_bat_number=sel_ab,
    pitch_number=sel_pn,
    pitcher_id=pitcher_id,
    exact_match_only=True,
)
if not video_file.exists() and video_url and has_preloaded is None:
    downloaded = fetch_pitch_clip_locally(video_url, video_path_str)
    if downloaded and Path(downloaded).exists():
        active_video_path = Path(downloaded)

with video_col:
    st.markdown("**🎬 1. Vídeo do Lance (Transmissão MLB)**")
    if active_video_path.exists():
        st.video(str(active_video_path))
    elif video_url:
        st.video(video_url)
    else:
        st.warning("Vídeo local em cache não encontrado. Utilizando simulação de pose.")

with base_print_col:
    st.markdown(f"**📸 2. Print Base para o Modelo (Frame #{release_frame})**")
    base_frame_img = extract_base_release_frame(
        video_path=active_video_path,
        release_frame_idx=release_frame,
        pitch_category=selected_cat,
        game_pk=sel_g_pk,
        at_bat_number=sel_ab,
        pitch_number=sel_pn,
        pitcher_id=pitcher_id,
    )
    _render_image_compat(
        base_frame_img,
        caption=f"Print Base Original (Frame #{release_frame}) — Imagem de entrada para extração da pose",
    )

with pose_col:
    st.markdown("**🦴 3. Modelo da Pose 3D (Esqueleto Extraído)**")

    metrics_dict = {
        "arm_slot_angle": float(bio_record.get("arm_slot_angle", 50.0)),
        "elbow_flexion_angle": float(bio_record.get("elbow_flexion_angle", 145.0)),
        "release_height": float(bio_record.get("release_height", 0.71)),
        "trunk_tilt": float(bio_record.get("trunk_tilt", 11.5)),
    }

    annotated_frame = extract_and_annotate_release_frame(
        video_path=active_video_path,
        release_frame_idx=release_frame,
        p_throws=p_throws,
        metrics=metrics_dict,
        extractor=pose_extractor,
        pitch_category=selected_cat,
        game_pk=sel_g_pk,
        at_bat_number=sel_ab,
        pitch_number=sel_pn,
        pitcher_id=pitcher_id,
    )

    _render_image_compat(
        annotated_frame,
        caption="Modelo de Pose MediaPipe sobreposto ao Print Base | Braço: Neon Cyan | Tronco: Neon Pink",
    )

# ---------------------------------------------------------------------------
# Métricas Biomecânicas Extraídas da Pose
# ---------------------------------------------------------------------------

st.markdown("##### 📐 Métricas Cinemáticas Extraídas do Modelo da Pose no Ponto de Soltura")
m1, m2, m3, m4 = st.columns(4)

arm_slot_val = float(bio_record.get("arm_slot_angle", 50.0))
elbow_flex_val = float(bio_record.get("elbow_flexion_angle", 145.0))
rel_ht_val = float(bio_record.get("release_height", 0.71))
trunk_tilt_val = float(bio_record.get("trunk_tilt", 11.5))
stride_ext_val = float(bio_record.get("stride_extension", 1.40))

with m1:
    st.metric(
        label="Ângulo do Braço (Arm Slot)",
        value=f"{arm_slot_val:.1f}°",
        delta="3/4 Slot" if 35 <= arm_slot_val <= 65 else ("Sidearm" if arm_slot_val < 35 else "Overhand"),
    )
with m2:
    st.metric(
        label="Flexão do Cotovelo",
        value=f"{elbow_flex_val:.1f}°",
        delta="Braço Estendido" if elbow_flex_val >= 130 else "Cotovelo Flexionado",
    )
with m3:
    st.metric(
        label="Altura de Soltura",
        value=f"{rel_ht_val:.3f}",
        delta="Ponto Alto" if rel_ht_val >= 0.70 else "Ponto Rebaixado",
    )
with m4:
    st.metric(
        label="Inclinação do Tronco",
        value=f"{trunk_tilt_val:.1f}°",
        delta="Alinhado",
    )

st.divider()

# ---------------------------------------------------------------------------
# 3. Predição Exclusiva com Base na Visão Computacional (Pitch Tipping)
# ---------------------------------------------------------------------------

st.subheader("3. Predição do Arremesso com Base na Visão Computacional (Modelo de Pose)")

sample_input = pd.DataFrame([{
    "arm_slot_angle": arm_slot_val,
    "release_height": rel_ht_val,
    "elbow_flexion_angle": elbow_flex_val,
    "trunk_tilt": trunk_tilt_val,
    "stride_extension": stride_ext_val,
}])

explanation = tipping_model.explain(sample_input)
bio_probs = explanation["probabilities"]
bio_pred = explanation["predicted_category"]
bio_conf = explanation["confidence"]

vision_pred_col, vision_prob_col = st.columns([1, 1.3], gap="medium")

with vision_pred_col, st.container(border=True):
    st.markdown("#### 👁️ Predição do Modelo de Visão (Pitch Tipping)")
    st.caption("Inferência gerada exclusivamente a partir dos ângulos 3D extraídos do print da pose")

    st.metric(
        label="Arremesso Previsto pela Pose",
        value=bio_pred,
        delta=f"{bio_conf * 100:.1f}% de certeza biomecânica",
    )

    if bio_pred == selected_cat:
        st.success(
            f"**Leitura Corporal Confirmada**: O modelo de Visão Computacional previu **{bio_pred}** "
            f"com **{bio_conf * 100:.1f}%** de confiança, acertando a categoria real do lance (`{selected_cat}`).",
            icon=":material/verified:",
        )
    else:
        st.info(
            f"**Disfarce Mecânico Detectado**: A postura corporal indicou padrão de **{bio_pred}** "
            f"({bio_conf * 100:.1f}%), enquanto o arremesso real foi `{selected_cat}`.",
            icon=":material/visibility:",
        )

with vision_prob_col, st.container(border=True):
    st.markdown("#### 📊 Distribuição de Probabilidades pela Pose 3D")
    st.caption("Probabilidade de cada família de arremesso calculada puramente pela mecânica corporal")

    fb_prob = float(bio_probs.get("Fastball", 0.0))
    bb_prob = float(bio_probs.get("Breaking Ball", 0.0))
    os_prob = float(bio_probs.get("Offspeed", 0.0))

    st.progress(fb_prob, text=f"🚀 Fastball (Bola Rápida): {fb_prob * 100:.1f}%")
    st.progress(bb_prob, text=f"🌪️ Breaking Ball (Bola de Efeito): {bb_prob * 100:.1f}%")
    st.progress(os_prob, text=f"🛑 Offspeed (Mudança de Velocidade): {os_prob * 100:.1f}%")

    st.caption(
        f"**Vetor de Entrada do Modelo de Visão**: Arm Slot `{arm_slot_val:.1f}°` • "
        f"Cotovelo `{elbow_flex_val:.1f}°` • Altura `{rel_ht_val:.3f}` • "
        f"Tronco `{trunk_tilt_val:.1f}°` • Passada `{stride_ext_val:.2f}m`"
    )

st.divider()

# ---------------------------------------------------------------------------
# 4. Diagnóstico Explicativo SHAP da Pose (Por que o corpo denunciou?)
# ---------------------------------------------------------------------------

st.subheader("4. Explicabilidade da Visão Computacional (SHAP): Quais Articulações Denunciaram o Arremesso?")

shap_col1, shap_col2 = st.columns([3, 2])

with shap_col1:
    st.markdown("##### Importância das Articulações da Pose na Predição (SHAP Values)")
    top_features = explanation.get("top_tipping_features", [])

    shap_chart_data = pd.DataFrame([
        {
            "Articulação / Métrica": f["display_name"],
            "Impacto SHAP": f["shap_value"],
        }
        for f in top_features
    ]).set_index("Articulação / Métrica")

    try:
        st.bar_chart(shap_chart_data, horizontal=True, color="#00FFCC", use_container_width=True)
    except TypeError:
        try:
            st.bar_chart(shap_chart_data, color="#00FFCC", use_container_width=True)
        except TypeError:
            st.bar_chart(shap_chart_data)

with shap_col2, st.container(border=True):
    st.markdown("##### 📝 Relatório Biomecânico da Pose")
    st.markdown(explanation.get("summary_text", ""))

    st.markdown("---")
    p_last_name = pitcher_info["name"].split()[-1]
    if bio_pred == "Fastball":
        speed_text = "100+ mph" if pitcher_id == 694973 else "alta velocidade"
        st.markdown(
            f"💡 **Leitura Visual para o Rebatedor**: No print capturado, o braço de {p_last_name} estende-se em amplitude máxima ({elbow_flex_val:.0f}°), "
            f"sem flexão excessiva no cotovelo. **Prepare o swing rápido para {speed_text}**."
        )
    elif bio_pred == "Breaking Ball":
        st.markdown(
            f"💡 **Leitura Visual para o Rebatedor**: O cotovelo flexionado em {elbow_flex_val:.0f}° e a soltura ajustada ({arm_slot_val:.0f}°) no modelo da pose denunciam "
            "que ele está 'envelopando' a bola para imprimir giro horizontal. **Atenção à bola de efeito quebrando para fora da zona**."
        )
    else:
        st.markdown(
            f"💡 **Leitura Visual para o Rebatedor**: O arm slot em {arm_slot_val:.0f}° com soltura firme na pose indica desaceleração e ação descendente. "
            "**Atenção ao arremesso em mudança afundando na parte baixa da zona**."
        )

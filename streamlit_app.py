"""Interactive app for speech-to-text transcription."""

from __future__ import annotations

import csv
import tempfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from digit_recognition.exports import build_json, build_srt, build_txt, build_vtt
from digit_recognition.media_ingest import (
    SUPPORTED_VIDEO_EXTENSIONS,
    MediaIngestError,
    extract_audio_from_video,
    fetch_youtube_audio,
    is_valid_youtube_url,
)

if TYPE_CHECKING:
    from digit_recognition.transcriber import SpeechTranscriber, TranscriptionResult


st.set_page_config(page_title="Speech-to-Text Transcription", page_icon="🎤", layout="wide")

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
AUTHOR_IMAGE = ASSETS_DIR / "pic1.png"
REPO_URL = "https://github.com/okonp07/Speech-to-text-generation-engine"
FUTURE_DEVELOPMENT_URL = f"{REPO_URL}/blob/main/future-development.md"
FEEDBACK_FILE = Path(__file__).resolve().parent / "feedback_responses.csv"
HeroPill = tuple[str, str]


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
        }
        .stApp {
            color: #102a26;
            background:
                radial-gradient(circle at top left, rgba(242, 201, 76, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(15, 118, 110, 0.20), transparent 28%),
                linear-gradient(180deg, #f7f3ea 0%, #f4efe4 45%, #efe7d8 100%);
        }
        [data-testid="stAppViewContainer"] {
            color: #102a26;
        }
        [data-testid="stHeader"] {
            background: rgba(247, 243, 234, 0.82);
            backdrop-filter: blur(8px);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        label[data-testid="stWidgetLabel"],
        label[data-testid="stWidgetLabel"] p,
        .stCaption,
        .stCaption p,
        .stRadio p,
        .stRadio label,
        .stFileUploader label,
        .stAudioInput label,
        .stSelectbox label,
        .stTextInput label,
        .stTextArea label,
        .stSlider label,
        .stMarkdown,
        .stMarkdown p:not(.hero-copy),
        .stMarkdown li,
        .stMarkdown span:not(.hero-kicker):not(.hero-title):not(.hero-copy) {
            color: #22313f !important;
        }
        .stRadio [role="radiogroup"] label,
        .stRadio [role="radiogroup"] p {
            color: #22313f !important;
        }
        .stAudioInput,
        .stFileUploader,
        .stSelectbox,
        .stTextInput,
        .stTextArea,
        .stSlider,
        .stRadio {
            color: #22313f;
        }
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {
            color: #102a26 !important;
            background: rgba(255, 252, 245, 0.9) !important;
        }
        [data-baseweb="select"] > div {
            color: #102a26 !important;
            background: rgba(255, 252, 245, 0.9) !important;
        }
        .stFileUploader section {
            background: rgba(255, 252, 245, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 18px;
        }
        .stFileUploader small,
        .stFileUploader span,
        .stFileUploader p {
            color: #22313f !important;
        }
        .stRadio [role="radiogroup"] {
            gap: 0.55rem;
        }
        .stRadio [role="radiogroup"] > label {
            background: rgba(255, 252, 245, 0.64);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 0.2rem 0.75rem;
        }
        .stButton > button,
        .stDownloadButton > button {
            background: rgba(255, 252, 245, 0.92);
            color: #102a26;
            border: 1px solid rgba(148, 163, 184, 0.24);
            min-height: 3rem;
            font-size: 0.98rem;
            font-weight: 600;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            color: #102a26;
            border-color: rgba(15, 118, 110, 0.32);
            background: rgba(255, 252, 245, 1);
        }
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {
            background: #0f766e;
            color: #fff9ed;
            border-color: #0f766e;
        }
        .stButton > button[kind="primary"]:hover,
        .stDownloadButton > button[kind="primary"]:hover {
            background: #115e59;
            color: #fff9ed;
            border-color: #115e59;
        }
        .stAlert {
            color: #15312d;
        }
        [data-testid="stAlertContainer"] * {
            color: inherit;
        }
        .hero-shell {
            position: relative;
            overflow: hidden;
            padding: 2rem 2rem 1.6rem 2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, #0f172a 0%, #134e4a 52%, #f59e0b 140%);
            color: #fff9ed;
            box-shadow: 0 24px 70px rgba(15, 23, 42, 0.18);
            margin-bottom: 1.4rem;
        }
        .hero-shell:after {
            content: "";
            position: absolute;
            inset: auto -40px -55px auto;
            width: 220px;
            height: 220px;
            background: rgba(255, 248, 220, 0.12);
            border-radius: 999px;
            filter: blur(4px);
        }
        .hero-kicker {
            display: inline-block;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 248, 220, 0.14);
            border: 1px solid rgba(255, 248, 220, 0.22);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.7rem;
            line-height: 1.02;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 0 0 0.65rem 0;
        }
        .hero-copy {
            max-width: 700px;
            font-size: 1.04rem;
            line-height: 1.6;
            color: rgba(255, 249, 237, 0.88);
            margin-bottom: 1.2rem;
        }
        .hero-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
        }
        a.hero-pill {
            display: block;
            background: rgba(255, 249, 237, 0.12);
            border: 1px solid rgba(255, 249, 237, 0.2);
            border-radius: 999px;
            padding: 0.5rem 0.9rem;
            font-size: 0.9rem;
            color: #fff9ed;
            text-decoration: none;
            transition: transform 0.18s ease, background 0.18s ease, border-color 0.18s ease;
        }
        a.hero-pill:hover {
            transform: translateY(-1px);
            background: rgba(255, 249, 237, 0.18);
            border-color: rgba(255, 249, 237, 0.32);
        }
        a.hero-pill:focus,
        a.hero-pill:focus-visible {
            outline: 2px solid rgba(255, 249, 237, 0.8);
            outline-offset: 2px;
        }
        .section-card {
            background: rgba(255, 252, 245, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 1.15rem 1.15rem 1rem 1.15rem;
            box-shadow: 0 12px 34px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
        }
        .section-card,
        .detail-card,
        .results-banner {
            scroll-margin-top: 1.2rem;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #102a26;
            margin-bottom: 0.25rem;
        }
        .section-copy {
            color: #4b5563;
            font-size: 0.95rem;
            margin-bottom: 0.5rem;
        }
        .mini-note {
            color: #5b5f68;
            font-size: 0.88rem;
            margin-top: 0.4rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 252, 245, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetricLabel"] {
            color: #5b5f68;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 800;
        }
        .results-banner {
            margin: 0.25rem 0 0.8rem 0;
            padding: 0.85rem 1rem;
            border-radius: 18px;
            background: linear-gradient(90deg, rgba(15, 118, 110, 0.12), rgba(245, 158, 11, 0.12));
            border: 1px solid rgba(15, 118, 110, 0.12);
            color: #15312d;
            font-weight: 600;
        }
        .transcript-shell {
            background: rgba(255, 252, 245, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 12px 34px rgba(15, 23, 42, 0.06);
            min-height: 220px;
        }
        .transcript-label {
            color: #0f766e;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 0.8rem;
            font-weight: 700;
        }
        .transcript-body {
            color: #102a26;
            font-size: 1.08rem;
            line-height: 1.85;
            white-space: pre-wrap;
        }
        .history-caption {
            color: #4b5563;
            font-size: 0.92rem;
            margin-top: -0.4rem;
            margin-bottom: 0.8rem;
        }
        .detail-card {
            background: rgba(255, 252, 245, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 1.35rem 1.35rem 1.1rem 1.35rem;
            box-shadow: 0 12px 34px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
        }
        .detail-card h3 {
            margin: 0 0 0.85rem 0;
            color: #102a26;
            font-size: 1.2rem;
            letter-spacing: -0.02em;
        }
        .detail-card p {
            color: #374151;
            line-height: 1.72;
            font-size: 0.98rem;
            margin-bottom: 0.8rem;
        }
        .detail-card ul {
            color: #374151;
            line-height: 1.72;
            font-size: 0.98rem;
            padding-left: 1.2rem;
            margin-top: 0.35rem;
        }
        .about-kicker {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.1);
            color: #0f766e;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 0.9rem;
        }
        .footer-shell {
            margin-top: 2.2rem;
            padding: 1.1rem 1rem 1.3rem 1rem;
            text-align: center;
            color: #4b5563;
            font-size: 0.93rem;
            line-height: 1.8;
        }
        @media (max-width: 768px) {
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(242, 201, 76, 0.12), transparent 30%),
                    linear-gradient(180deg, #f8f4eb 0%, #f5efe2 54%, #efe7d8 100%);
            }
            [data-testid="stHeader"] {
                background: rgba(248, 244, 235, 0.92);
            }
            .block-container {
                padding-top: 1.25rem;
                padding-bottom: 2rem;
            }
            .hero-shell {
                padding: 1.4rem 1.2rem 1.2rem 1.2rem;
                border-radius: 22px;
            }
            .hero-shell:after {
                display: none;
            }
            .hero-title {
                font-size: 2.05rem;
                line-height: 1.06;
            }
            .hero-copy {
                font-size: 0.98rem;
                line-height: 1.62;
            }
            .hero-pills {
                gap: 0.45rem;
            }
            a.hero-pill {
                font-size: 0.82rem;
                padding: 0.45rem 0.72rem;
            }
            .section-card,
            .detail-card,
            .transcript-shell {
                border-radius: 18px;
                padding: 1rem 0.95rem 0.95rem 0.95rem;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            }
            .section-title,
            .detail-card h3 {
                font-size: 1.02rem;
            }
            .section-copy,
            .detail-card p,
            .detail-card ul,
            .history-caption,
            .footer-shell {
                font-size: 0.95rem;
                line-height: 1.68;
            }
            .transcript-body {
                font-size: 0.99rem;
                line-height: 1.72;
            }
            div[data-testid="stMetric"] {
                border-radius: 18px;
                padding: 0.85rem;
            }
            .stButton > button,
            .stDownloadButton > button {
                min-height: 3.1rem;
                font-size: 0.95rem;
            }
            .stRadio [role="radiogroup"] > label {
                padding: 0.15rem 0.65rem;
            }
            .stTextInput input,
            .stTextArea textarea,
            [data-baseweb="select"] > div {
                font-size: 0.98rem !important;
            }
        }
        @media (max-width: 560px) {
            .hero-title {
                font-size: 1.82rem;
            }
            .hero-copy,
            .section-copy,
            .detail-card p,
            .detail-card ul,
            .transcript-body,
            .footer-shell {
                font-size: 0.93rem;
            }
            .hero-pills {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.45rem;
            }
            a.hero-pill {
                text-align: center;
                width: 100%;
            }
            .stButton > button,
            .stDownloadButton > button {
                font-size: 0.92rem;
            }
            .stRadio [role="radiogroup"] {
                align-items: stretch;
            }
            .stRadio [role="radiogroup"] > label {
                width: 100%;
            }
            [data-testid="stSidebar"] .stButton > button,
            [data-testid="stSidebar"] .stLinkButton > a {
                min-height: 2.9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(
    kicker: str = "Live Speech-to-Text",
    title: str = "Speech-to-Text Transcription",
    copy: str = (
        "Record directly in the browser or upload a spoken clip, then turn it into written text "
        "with a confidence score, detected language, and audio diagnostics."
    ),
    pills: list[HeroPill] | None = None,
) -> None:
    pills = pills or [
        ("Microphone recording", "#input-section"),
        ("Transcript output", "#results-section"),
        ("Confidence score", "#results-section"),
        ("Audio quality checks", "#quality-checks-section"),
    ]
    pills_html = "".join(
        f'<a class="hero-pill" href="{escape(href, quote=True)}">{escape(label)}</a>'
        for label, href in pills
    )
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">{escape(kicker)}</div>
            <div class="hero-title">{escape(title)}</div>
            <div class="hero-copy">{escape(copy)}</div>
            <div class="hero-pills">
                {pills_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_intro(title: str, copy: str, anchor_id: str | None = None) -> None:
    anchor_attr = f' id="{escape(anchor_id, quote=True)}"' if anchor_id else ""
    st.markdown(
        f"""
        <div class="section-card"{anchor_attr}>
            <div class="section-title">{escape(title)}</div>
            <div class="section-copy">{escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _detail_card(
    title: str,
    body_html: str,
    kicker: str | None = None,
    anchor_id: str | None = None,
) -> None:
    anchor_attr = f' id="{escape(anchor_id, quote=True)}"' if anchor_id else ""
    title_html = f"<h3>{escape(title)}</h3>" if title else ""
    kicker_html = f'<div class="about-kicker">{escape(kicker)}</div>' if kicker else ""
    card_html = f'<div class="detail-card"{anchor_attr}>{title_html}{kicker_html}{body_html}</div>'
    st.markdown(card_html, unsafe_allow_html=True)


def _render_footer() -> None:
    st.markdown(
        """
        <div class="footer-shell">
            <div>&copy; Okon Prince, 2026</div>
            <div>
                This project is meant for Educational and research purposes only.
                It is not to be used for commercial purposes.
            </div>
            <div>enquiries; okonp07@gmail.com</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_audio(audio: np.ndarray, sample_rate: int):
    import librosa

    figure, axes = plt.subplots(2, 1, figsize=(10, 6))
    times = np.arange(len(audio)) / sample_rate
    axes[0].plot(times, audio, color="#0f766e", linewidth=1)
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=64, n_fft=1024, hop_length=256)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    image = axes[1].imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Mel Spectrogram")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Mel Bin")
    figure.colorbar(image, ax=axes[1], shrink=0.8)
    figure.tight_layout()
    return figure


def _html_paragraphs(paragraphs: list[str]) -> str:
    return "\n".join(f"<p>{escape(paragraph)}</p>" for paragraph in paragraphs)


def _html_bullets(items: list[tuple[str, str]]) -> str:
    bullet_html = "".join(
        f"<li><strong>{escape(label)}:</strong> {escape(description)}</li>"
        for label, description in items
    )
    return f"<ul>{bullet_html}</ul>"


def _author_profile_html() -> str:
    paragraphs = [
        (
            "Okon Prince\n"
            "Senior Data Scientist at MIVA Open University | AI Engineer & Data Scientist"
        ),
        "I design and deploy end-to-end data systems that turn raw data into production-ready intelligence.",
        (
            "My core stack includes Python, Streamlit, BigQuery, Deep Learning, Supabase, "
            "Hugging Face, PySpark, SQL, Machine Learning, LLMs, and Transformers."
        ),
        (
            "My work spans risk scoring systems, A/B testing, Traditional and AI-powered "
            "dashboards, RAG pipelines, predictive analytics, Image processing and analytics, "
            "LLM-based solutions and AI research."
        ),
        (
            "Currently, I work as a Senior Data Scientist in the department of Research and "
            "Development at MIVA Open University, where I carry out AI / ML Research and "
            "build intelligent systems that drive analytics, decision support and scalable AI innovation."
        ),
    ]
    return "\n".join(
        [
            _html_paragraphs(paragraphs),
            (
                f"<p><strong>{escape('I believe:')}</strong> "
                f"{escape('models are trained, systems are engineered and impact is delivered.')}</p>"
            ),
        ]
    )


def _render_sidebar_navigation() -> str:
    if "page" not in st.session_state:
        st.session_state.page = "App"

    with st.sidebar:
        nav_app, nav_about, nav_feedback = st.columns(3)
        if nav_app.button(
            "App",
            use_container_width=True,
            type="primary" if st.session_state.page == "App" else "secondary",
        ):
            st.session_state.page = "App"
            st.rerun()
        if nav_about.button(
            "About",
            use_container_width=True,
            type="primary" if st.session_state.page == "About" else "secondary",
        ):
            st.session_state.page = "About"
            st.rerun()
        if nav_feedback.button(
            "Feedback",
            use_container_width=True,
            type="primary" if st.session_state.page == "Feedback" else "secondary",
        ):
            st.session_state.page = "Feedback"
            st.rerun()
        link_future, link_codebase = st.columns(2)
        link_future.link_button(
            "Future dev",
            FUTURE_DEVELOPMENT_URL,
            use_container_width=True,
        )
        link_codebase.link_button(
            "Codebase",
            REPO_URL,
            use_container_width=True,
        )
        st.markdown("---")

    return st.session_state.page


def _append_feedback_row(payload: dict[str, object]) -> None:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "submitted_at_utc",
        "name",
        "email",
        "use_case",
        "overall_rating",
        "transcription_quality",
        "ease_of_use",
        "design_clarity",
        "would_recommend",
        "favorite_part",
        "improvement_suggestion",
    ]
    file_exists = FEEDBACK_FILE.exists()
    with FEEDBACK_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(payload)


def _render_feedback_page() -> None:
    _render_hero(
        kicker="User Feedback",
        title="Feedback",
        copy=(
            "Share your experience with the app so we can understand what works well, "
            "what feels unclear, and where the product should improve next."
        ),
        pills=[
            ("User experience", "#feedback-form"),
            ("Transcription quality", "#feedback-form"),
            ("Design clarity", "#feedback-form"),
            ("Improvement ideas", "#feedback-form"),
        ],
    )

    _section_intro(
        "Why feedback matters",
        (
            "This form helps us assess the real user experience of the speech-to-text app, "
            "including transcription quality, ease of use, interface clarity, and priorities for future improvement."
        ),
    )

    _detail_card(
        "Feedback form",
        _html_paragraphs(
            [
                "Your responses are stored locally with the app so they can be reviewed later.",
                "You can keep your feedback anonymous, or include contact details if you want a follow-up.",
            ]
        ),
        anchor_id="feedback-form",
    )

    with st.form("feedback-form-fields", clear_on_submit=True):
        identity_left, identity_right = st.columns(2)
        with identity_left:
            name = st.text_input("Name (optional)")
        with identity_right:
            email = st.text_input("Email (optional)")

        use_case = st.selectbox(
            "What best describes your use case?",
            [
                "Demo / exploration",
                "Research / experimentation",
                "Education / learning",
                "Product prototype",
                "Personal transcription",
                "Other",
            ],
        )

        overall_rating = st.slider("Overall experience", min_value=1, max_value=5, value=4)
        transcription_quality = st.slider("Transcription quality", min_value=1, max_value=5, value=4)
        ease_of_use = st.slider("Ease of use", min_value=1, max_value=5, value=4)
        design_clarity = st.slider("Interface clarity", min_value=1, max_value=5, value=4)
        would_recommend = st.radio(
            "Would you recommend this app to someone else?",
            ["Yes", "Maybe", "No"],
            horizontal=True,
        )
        favorite_part = st.text_area(
            "What did you like most?",
            placeholder="Tell us what felt strongest or most useful in the experience.",
            height=120,
        )
        improvement_suggestion = st.text_area(
            "What should we improve?",
            placeholder="Share anything that felt unclear, inaccurate, or missing.",
            height=140,
        )

        submitted = st.form_submit_button("Submit feedback", use_container_width=True, type="primary")

    if submitted:
        payload = {
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "name": name.strip(),
            "email": email.strip(),
            "use_case": use_case,
            "overall_rating": overall_rating,
            "transcription_quality": transcription_quality,
            "ease_of_use": ease_of_use,
            "design_clarity": design_clarity,
            "would_recommend": would_recommend,
            "favorite_part": favorite_part.strip(),
            "improvement_suggestion": improvement_suggestion.strip(),
        }
        _append_feedback_row(payload)
        st.success("Thanks for the feedback. Your response has been recorded.")


@st.cache_resource
def _load_transcriber(model_size: str) -> "SpeechTranscriber":
    try:
        from digit_recognition import SpeechTranscriber
    except ImportError as exc:
        raise RuntimeError(
            "Speech transcription dependencies are unavailable. Install requirements.txt with Python 3.11 "
            "or 3.12 before using the app."
        ) from exc
    return SpeechTranscriber(model_size=model_size)


def _transcript_html(result: "TranscriptionResult") -> str:
    transcript = escape(result.text).replace("\n", "<br>")
    return (
        '<div class="transcript-shell">'
        '<div class="transcript-label">Transcript</div>'
        f'<div class="transcript-body">{transcript}</div>'
        "</div>"
    )


def _transcribe(uploaded_file, transcriber: "SpeechTranscriber", language: str | None):
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    try:
        return _transcribe_local_path(temp_path, transcriber, language)
    finally:
        temp_path.unlink(missing_ok=True)


def _transcribe_local_path(
    audio_path: Path,
    transcriber: "SpeechTranscriber",
    language: str | None,
):
    """Shared transcription helper for files already on disk.

    Used by the YouTube URL and video file tabs, where ingestion produces a
    local path we want to keep alive across reruns (cached in session
    state) rather than deleting straight away.
    """

    result = transcriber.transcribe_file(audio_path, language=language)
    audio = transcriber.processor.load_audio(audio_path)
    report = transcriber.processor.quality_report(audio)
    return audio, result, report


def _safe_basename(name: str) -> str:
    stem = Path(name).stem or "transcript"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)[:80] or "transcript"


def _render_download_buttons(result: "TranscriptionResult", base_name: str) -> None:
    """Four download buttons matching the reference notebook output set."""

    base = _safe_basename(base_name)
    col_txt, col_srt, col_vtt, col_json = st.columns(4)
    with col_txt:
        st.download_button(
            "Download .txt",
            data=build_txt(result),
            file_name=f"{base}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col_srt:
        st.download_button(
            "Download .srt",
            data=build_srt(result),
            file_name=f"{base}.srt",
            mime="application/x-subrip",
            use_container_width=True,
        )
    with col_vtt:
        st.download_button(
            "Download .vtt",
            data=build_vtt(result),
            file_name=f"{base}.vtt",
            mime="text/vtt",
            use_container_width=True,
        )
    with col_json:
        st.download_button(
            "Download .json",
            data=build_json(result),
            file_name=f"{base}.json",
            mime="application/json",
            use_container_width=True,
        )


def _render_about_page() -> None:
    _render_hero(
        kicker="Project Overview",
        title="About",
        copy=(
            "This page explains what the speech-to-text application does, why it was built "
            "and how the solution works end to end."
        ),
        pills=[
            ("Project summary", "#about-project"),
            ("System workflow", "#system-workflow"),
            ("About the Author", "#author-profile"),
            ("Future development", "#future-development"),
        ],
    )

    _detail_card(
        "About the project",
        _html_paragraphs(
            [
                (
                    "This project is an interactive speech-to-text application that converts "
                    "spoken audio into written words. A user can either speak directly into the "
                    "browser using a microphone or upload an existing audio file, and the app "
                    "returns a transcript together with a confidence score and supporting diagnostics."
                ),
                (
                    "The goal of the project is to make speech AI practical, understandable, and "
                    "easy to explore. Instead of limiting the user to a narrow classification task, "
                    "the app now handles free-form speech and presents the output in a way that both "
                    "technical and non-technical users can interpret."
                ),
                (
                    "This makes the solution useful for educational demonstrations, AI research, "
                    "speech-interface prototyping, and general experimentation with audio-to-text workflows."
                ),
            ]
        ),
        anchor_id="about-project",
    )

    _detail_card(
        "How the solution works",
        _html_paragraphs(
            [
                (
                    "The solution begins with audio capture. The user either records speech in the "
                    "browser or uploads a supported audio file such as WAV, MP3, M4A, FLAC, or OGG. "
                    "The app loads the audio at a consistent sample rate so the rest of the pipeline "
                    "can work on a stable representation of the signal."
                ),
                (
                    "Once the audio is available, the transcription engine processes the speech and "
                    "decodes it into natural-language text. The underlying model is designed for "
                    "automatic speech recognition, which means it does not try to choose from a small "
                    "set of labels. Instead, it generates actual written language from what it hears."
                ),
                (
                    "After transcription, the app presents the result in a structured way. The main "
                    "transcript shows the recognized text, the confidence score summarizes how reliable "
                    "the transcription appears to be, and language metadata helps the user understand "
                    "what language the system believes it detected."
                ),
                (
                    "The app also exposes segment-level timing, which makes it easier to inspect how "
                    "different parts of the recording were interpreted. In addition, waveform and "
                    "spectrogram views help reveal silence, clipping, or noise, while audio-quality "
                    "checks help explain why a result may be strong, weak, or partially inaccurate."
                ),
                (
                    "Taken together, the solution is not just a transcript generator. It is a clear, "
                    "inspectable speech-processing workflow that helps the user understand both the text "
                    "output and the quality of the underlying audio."
                ),
            ]
        ),
        anchor_id="system-workflow",
    )

    _detail_card(
        "Future development",
        _html_paragraphs(
            [
                (
                    "A dedicated future-development guide is maintained in the repository for anyone "
                    "who wants to extend this work. It outlines realistic enhancement directions, "
                    "implementation considerations, and contributor expectations."
                ),
                (
                    "That document also makes it clear that future work built on this foundation "
                    "must acknowledge the original project originator."
                ),
            ]
        ),
        anchor_id="future-development",
    )
    st.link_button("future development", FUTURE_DEVELOPMENT_URL)

    author_text_col, author_image_col = st.columns([1.45, 1], gap="large")
    with author_text_col:
        _detail_card(
            "About the Author",
            _author_profile_html(),
            anchor_id="author-profile",
        )

    with author_image_col:
        st.markdown("<div style='height: 3.35rem;'></div>", unsafe_allow_html=True)
        image_left, image_center, image_right = st.columns([0.12, 0.76, 0.12])
        with image_center:
            st.image(AUTHOR_IMAGE, use_container_width=True)


def _render_results_panel(
    audio: np.ndarray,
    result: "TranscriptionResult",
    report,
    display_name: str,
    transcriber: "SpeechTranscriber",
) -> None:
    """Shared rendering used by every input source."""

    st.markdown(
        """
        <div class="results-banner" id="results-section">
            Transcript ready. Review the written output, confidence score, and audio diagnostics below.
        </div>
        """,
        unsafe_allow_html=True,
    )

    transcript_col, metrics_col = st.columns([1.7, 1], gap="large")
    with transcript_col:
        st.markdown(_transcript_html(result), unsafe_allow_html=True)

    with metrics_col:
        st.metric("Confidence", f"{result.confidence:.1%}")
        st.metric("Detected language", result.language.upper())
        st.metric(
            "Language confidence",
            "N/A" if result.language_confidence is None else f"{result.language_confidence:.1%}",
        )
        st.metric("Transcript duration", f"{result.duration_seconds:.1f}s")

    _section_intro(
        "Download transcript",
        "Save the transcript alongside time-coded SRT/VTT subtitles and a full JSON dump.",
    )
    _render_download_buttons(result, display_name)

    if result.segments:
        _section_intro(
            "Transcript timeline",
            "Each row shows a transcribed segment with timing and confidence.",
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "start_seconds": round(segment.start_seconds, 2),
                        "end_seconds": round(segment.end_seconds, 2),
                        "confidence": round(segment.confidence, 3),
                        "text": segment.text,
                    }
                    for segment in result.segments
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    viz_col, qa_col = st.columns([1.5, 1])
    with viz_col:
        _section_intro(
            "Signal view",
            (
                "The waveform and mel spectrogram help you see whether the recording is clear, "
                "clipped, or dominated by silence."
            ),
        )
        st.pyplot(_plot_audio(audio, transcriber.processor.sample_rate))

    with qa_col:
        _section_intro(
            "Quality checks",
            "These quick diagnostics flag common recording issues that can reduce prediction quality.",
            anchor_id="quality-checks-section",
        )
        st.write(
            {
                "duration_seconds": round(report.duration_seconds, 3),
                "peak_amplitude": round(report.peak_amplitude, 4),
                "rms_amplitude": round(report.rms_amplitude, 4),
                "issues": list(report.issues) or ["No obvious issues detected."],
            }
        )

    st.session_state.history.append(
        {
            "file": display_name,
            "transcript_preview": result.text[:80] + ("..." if len(result.text) > 80 else ""),
            "confidence": round(result.confidence, 4),
            "language": result.language,
        }
    )
    _section_intro(
        "Transcription history",
        (
            "Track how the app responds across multiple takes so you can compare speaking styles, "
            "background conditions, and transcript confidence."
        ),
        anchor_id="history-section",
    )
    st.markdown(
        '<div class="history-caption">Latest transcripts from this session appear below.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)


def _run_and_render(
    source_kind: str,
    audio_source,
    audio_path: Path | None,
    display_name: str,
    model_size: str,
    language: str | None,
) -> None:
    """Load the transcriber, transcribe, then render the shared results panel.

    Exactly one of *audio_source* (an UploadedFile / AudioInput result) and
    *audio_path* (a Path on disk produced by media_ingest) must be provided.
    """

    try:
        transcriber = _load_transcriber(model_size)
        if audio_source is not None:
            audio, result, report = _transcribe(audio_source, transcriber, language=language)
        elif audio_path is not None:
            audio, result, report = _transcribe_local_path(audio_path, transcriber, language=language)
        else:  # pragma: no cover - defensive
            st.error("No audio source was provided to the transcriber.")
            return
    except RuntimeError as exc:
        st.error(str(exc))
        st.caption(
            "If you deploy on Streamlit Community Cloud, choose Python 3.11 or 3.12 in Advanced settings."
        )
        return
    except Exception as exc:
        st.error(f"Transcription failed ({source_kind}): {exc}")
        st.caption(
            "If this is the first run, confirm that model downloads are allowed in the deployment environment."
        )
        return

    _render_results_panel(audio, result, report, display_name, transcriber)


def _render_app_page() -> None:
    _render_hero()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "audio_input_key" not in st.session_state:
        st.session_state.audio_input_key = 0
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0
    if "video_uploader_key" not in st.session_state:
        st.session_state.video_uploader_key = 0
    if "youtube_url_value" not in st.session_state:
        st.session_state.youtube_url_value = ""
    # Caches of ingested local audio paths, keyed so we don't re-download /
    # re-extract on every Streamlit rerun.
    if "youtube_cache" not in st.session_state:
        st.session_state.youtube_cache = {}  # {url: (Path, display_name)}
    if "video_cache" not in st.session_state:
        st.session_state.video_cache = {}  # {cache_key: (Path, display_name)}

    with st.sidebar:
        st.markdown("### Control Panel")
        model_label = st.selectbox(
            "Transcription model",
            ["Fast (tiny)", "Balanced (base)", "Detailed (small)"],
            index=1,
        )
        model_size = {
            "Fast (tiny)": "tiny",
            "Balanced (base)": "base",
            "Detailed (small)": "small",
        }[model_label]
        language_label = st.selectbox(
            "Language hint",
            ["Auto detect", "English"],
            index=0,
        )
        language = None if language_label == "Auto detect" else "en"
        st.markdown("Supported audio: WAV, MP3, M4A, FLAC, OGG")
        st.markdown("Supported video: MP4, MOV, MKV, WEBM, AVI, M4V")
        st.markdown("YouTube: paste any public watch / shorts / youtu.be link")
        st.markdown("---")
        st.caption("The first run may take longer while the speech model downloads.")
        st.caption("Best results come from clear speech and low background noise.")
        st.caption("YouTube + video flows require ffmpeg on the host.")

    _section_intro(
        "Input",
        (
            "Pick how you want to feed audio into the app. You can record in the browser, "
            "upload an audio or video file, or paste a YouTube link."
        ),
        anchor_id="input-section",
    )

    mic_tab, audio_tab, video_tab, youtube_tab = st.tabs(
        ["🎤 Microphone", "📁 Audio file", "🎞️ Video file", "▶️ YouTube URL"]
    )

    # --- Microphone tab ------------------------------------------------------
    with mic_tab:
        st.caption(
            "Click record, allow microphone access in your browser, speak naturally, "
            "then stop recording."
        )
        mic_audio = st.audio_input(
            "Record speech",
            sample_rate=22050,
            key=f"audio-input-{st.session_state.audio_input_key}",
        )
        if st.button("Clear recording", key="clear-mic"):
            st.session_state.audio_input_key += 1
            st.rerun()

        if mic_audio:
            st.audio(mic_audio)
            _run_and_render(
                source_kind="microphone",
                audio_source=mic_audio,
                audio_path=None,
                display_name=getattr(mic_audio, "name", "microphone.wav"),
                model_size=model_size,
                language=language,
            )
        else:
            st.info("Record a clip to start.")

    # --- Audio file tab ------------------------------------------------------
    with audio_tab:
        uploaded_audio = st.file_uploader(
            "Upload an audio file containing speech",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            key=f"file-uploader-{st.session_state.file_uploader_key}",
        )
        if st.button("Clear uploaded audio", key="clear-audio"):
            st.session_state.file_uploader_key += 1
            st.rerun()

        if uploaded_audio:
            st.audio(uploaded_audio)
            _run_and_render(
                source_kind="audio file",
                audio_source=uploaded_audio,
                audio_path=None,
                display_name=uploaded_audio.name,
                model_size=model_size,
                language=language,
            )
        else:
            st.info("Upload an audio file to start.")

    # --- Video file tab ------------------------------------------------------
    with video_tab:
        st.caption(
            "Upload a video; the app extracts the audio track with ffmpeg and then transcribes it."
        )
        uploaded_video = st.file_uploader(
            "Upload a video file",
            type=[ext.lstrip(".") for ext in SUPPORTED_VIDEO_EXTENSIONS],
            key=f"video-uploader-{st.session_state.video_uploader_key}",
        )
        col_go, col_clear = st.columns([1, 1])
        with col_go:
            run_video = st.button(
                "Extract audio and transcribe",
                key="run-video",
                type="primary",
                disabled=uploaded_video is None,
                use_container_width=True,
            )
        with col_clear:
            if st.button("Clear video", key="clear-video", use_container_width=True):
                st.session_state.video_uploader_key += 1
                st.session_state.video_cache = {}
                st.rerun()

        if uploaded_video is not None:
            cache_key = f"{uploaded_video.name}:{uploaded_video.size}"
            cached = st.session_state.video_cache.get(cache_key)

            if run_video and cached is None:
                with st.spinner("Extracting audio from video…"):
                    try:
                        video_tmp_dir = Path(tempfile.mkdtemp(prefix="video_src_"))
                        video_tmp_path = video_tmp_dir / uploaded_video.name
                        video_tmp_path.write_bytes(uploaded_video.getbuffer())
                        ingested = extract_audio_from_video(
                            video_tmp_path,
                            display_name=uploaded_video.name,
                        )
                        st.session_state.video_cache[cache_key] = (
                            ingested.audio_path,
                            ingested.display_name,
                        )
                        cached = st.session_state.video_cache[cache_key]
                    except MediaIngestError as exc:
                        st.error(str(exc))
                        cached = None

            if cached is not None:
                audio_path, display_name = cached
                st.audio(str(audio_path))
                _run_and_render(
                    source_kind="video file",
                    audio_source=None,
                    audio_path=audio_path,
                    display_name=display_name,
                    model_size=model_size,
                    language=language,
                )
            elif not run_video:
                st.info("Upload a video, then click “Extract audio and transcribe”.")
        else:
            st.info("Upload a video file to start.")

    # --- YouTube URL tab -----------------------------------------------------
    with youtube_tab:
        st.caption(
            "Paste a YouTube link (watch, shorts, or youtu.be). The app uses yt-dlp to pull "
            "the audio, then transcribes it exactly like any other source."
        )
        youtube_url = st.text_input(
            "YouTube URL",
            value=st.session_state.youtube_url_value,
            placeholder="https://www.youtube.com/watch?v=…",
            key="youtube-url-input",
        )
        st.session_state.youtube_url_value = youtube_url

        with st.expander(
            "Bot-check workaround: upload a YouTube cookies.txt",
            expanded=False,
        ):
            st.markdown(
                "On hosted deployments (Streamlit Community Cloud, Render, Fly.io, "
                "etc.) YouTube frequently demands a signed-in cookie before it will "
                "serve audio — the error looks like *“Sign in to confirm you're not "
                "a bot”*. Upload a `cookies.txt` exported from your browser to bypass it."
            )
            st.markdown(
                "**How to get `cookies.txt`:** install the "
                "[Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) "
                "Chrome extension, open [youtube.com](https://www.youtube.com) while "
                "signed in, click the extension, and save the file. **Use a throwaway "
                "Google account** — anyone with this file can use your YouTube session."
            )
            uploaded_cookies = st.file_uploader(
                "cookies.txt",
                type=["txt"],
                key="youtube-cookies",
                help="Netscape-format cookies exported from your browser while signed into YouTube.",
            )
            if uploaded_cookies is not None:
                st.session_state["youtube_cookies_bytes"] = uploaded_cookies.getvalue()
                st.success("Cookies loaded for this session.")
            elif "youtube_cookies_bytes" in st.session_state:
                st.caption("Cookies from an earlier upload are still active in this session.")
            if st.session_state.get("youtube_cookies_bytes") and st.button(
                "Clear stored cookies", key="clear-youtube-cookies"
            ):
                st.session_state.pop("youtube_cookies_bytes", None)
                st.rerun()

        col_go_yt, col_clear_yt = st.columns([1, 1])
        with col_go_yt:
            run_youtube = st.button(
                "Download and transcribe",
                key="run-youtube",
                type="primary",
                disabled=not youtube_url.strip(),
                use_container_width=True,
            )
        with col_clear_yt:
            if st.button("Clear URL and cache", key="clear-youtube", use_container_width=True):
                st.session_state.youtube_url_value = ""
                st.session_state.youtube_cache = {}
                st.rerun()

        url_key = youtube_url.strip()
        cached_yt = st.session_state.youtube_cache.get(url_key)

        if run_youtube and url_key:
            if not is_valid_youtube_url(url_key):
                st.error(
                    "That does not look like a YouTube URL. Paste a link that starts "
                    "with https://www.youtube.com/ or https://youtu.be/."
                )
            elif cached_yt is None:
                with st.spinner("Downloading audio from YouTube…"):
                    cookies_bytes = st.session_state.get("youtube_cookies_bytes")
                    cookie_tmp: Path | None = None
                    try:
                        if cookies_bytes:
                            cookie_tmp = Path(
                                tempfile.NamedTemporaryFile(
                                    delete=False, suffix=".txt"
                                ).name
                            )
                            cookie_tmp.write_bytes(cookies_bytes)
                        ingested = fetch_youtube_audio(
                            url_key,
                            cookiefile=cookie_tmp,
                        )
                        st.session_state.youtube_cache[url_key] = (
                            ingested.audio_path,
                            ingested.display_name,
                        )
                        cached_yt = st.session_state.youtube_cache[url_key]
                    except MediaIngestError as exc:
                        st.error(str(exc))
                        cached_yt = None
                    finally:
                        if cookie_tmp is not None:
                            cookie_tmp.unlink(missing_ok=True)

        if cached_yt is not None:
            audio_path, display_name = cached_yt
            st.audio(str(audio_path))
            _run_and_render(
                source_kind="YouTube URL",
                audio_source=None,
                audio_path=audio_path,
                display_name=display_name,
                model_size=model_size,
                language=language,
            )
        elif not run_youtube:
            st.info("Paste a YouTube URL, then click “Download and transcribe”.")


def main() -> None:
    _inject_styles()
    page = _render_sidebar_navigation()

    if page == "About":
        _render_about_page()
    elif page == "Feedback":
        _render_feedback_page()
    else:
        _render_app_page()

    _render_footer()


if __name__ == "__main__":
    main()

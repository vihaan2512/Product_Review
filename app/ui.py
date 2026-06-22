import io
import base64
import time
import json
from pathlib import Path

import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image

# ── Page config — must be first Streamlit call ────────────────
st.set_page_config(
    page_title="Multimodal E-commerce Product Review Analysis Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────

def grade_color(grade: str) -> str:
    if grade.startswith("A"): return "#1D9E75"
    if grade.startswith("B"): return "#4A9EE0"
    if grade.startswith("C"): return "#EF9F27"
    if grade.startswith("D"): return "#E06B4A"
    return "#D85A30"


def score_color(score: float) -> str:
    if score >= 80: return "#1D9E75"
    if score >= 65: return "#4A9EE0"
    if score >= 50: return "#EF9F27"
    return "#D85A30"


# ── Visualisation helpers ─────────────────────────────────────

def render_gauge(score: float) -> go.Figure:
    color = score_color(score)
    fig   = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 48, "color": color}, "suffix": "/100"},
        gauge={
            "axis":     {"range": [0, 100], "tickwidth": 1},
            "bar":      {"color": color, "thickness": 0.3},
            "bgcolor":  "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  50], "color": "#FDECEA"},
                {"range": [50, 65], "color": "#FEF3E2"},
                {"range": [65, 80], "color": "#E3F4F1"},
                {"range": [80, 100],"color": "#D6EEF8"},
            ],
            "threshold": {
                "line":  {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_breakdown_bars(breakdown: dict) -> go.Figure:
    dim_labels = {
        "sentiment":    "Customer Sentiment",
        "defect":       "Product Condition",
        "authenticity": "Review Authenticity",
        "aspect":       "Aspect Quality",
    }
    dims   = list(breakdown.keys())
    scores = list(breakdown.values())
    colors = [score_color(s) for s in scores]
    labels = [dim_labels.get(d, d.title()) for d in dims]

    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}" for s in scores],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}/100<extra></extra>",
    ))
    fig.update_layout(
        height=max(200, len(dims) * 60),
        xaxis=dict(range=[0, 115], showgrid=True, title="Score /100"),
        yaxis=dict(title=""),
        margin=dict(l=10, r=60, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_aspect_radar(absa_results: dict) -> go.Figure:
    if not absa_results:
        return None

    sorted_aspects = sorted(
        absa_results.items(),
        key=lambda x: abs(x[1].get("mean_score", 0)),
        reverse=True,
    )[:8]

    aspects = [a for a, _ in sorted_aspects]
    scores  = [d.get("mean_score", 0) for _, d in sorted_aspects]
    radar   = [(s + 1) / 2 for s in scores]
    radar_c = radar + [radar[0]]
    asp_c   = aspects + [aspects[0]]

    avg   = np.mean(scores)
    color = "#1D9E75" if avg > 0.1 else "#D85A30" if avg < -0.1 else "#EF9F27"
    fill_color = "rgba(29, 158, 117, 0.2)" if avg > 0.1 else "rgba(216, 90, 48, 0.2)" if avg < -0.1 else "rgba(239, 159, 39, 0.2)"

    fig = go.Figure(go.Scatterpolar(
        r=radar_c, theta=asp_c,
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=color, width=2),
        marker=dict(size=6),
        hovertemplate="<b>%{theta}</b><br>Score: %{customdata:.2f}<extra></extra>",
        customdata=scores + [scores[0]],
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["Very Neg", "Neg", "Neutral", "Pos", "Very Pos"],
                tickfont=dict(size=9),
            )
        ),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_sentiment_donut(sentiment_results: list) -> go.Figure:
    from collections import Counter
    counts = Counter(r.get("label", "neutral") for r in sentiment_results)
    labels = ["positive", "neutral", "negative"]
    values = [counts.get(l, 0) for l in labels]
    colors = ["#1D9E75", "#EF9F27", "#D85A30"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=colors, hole=0.5,
        hovertemplate="<b>%{label}</b><br>%{value} reviews (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        height=280,
        showlegend=True,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_fake_review_table(reviews: list, fake_results: list) -> pd.DataFrame:
    rows = []
    for i, (review, fake) in enumerate(zip(reviews, fake_results)):
        risk  = fake.get("risk_level", "low")
        score = fake.get("fake_score", 0)
        emoji = "🔴" if risk == "high" else "🟡" if risk == "medium" else "🟢"
        
        # Format flags into clean reasons list
        flags = fake.get("flags", [])
        explanation = ", ".join(flags) if flags else "Linguistic patterns indicate a genuine customer review."
        
        rows.append({
            "#":          i + 1,
            "Risk":       f"{emoji} {risk.upper()}",
            "Score":      f"{score:.2f}",
            "Explanation": explanation,
            "Review":     review[:100] + "..." if len(review) > 100 else review,
        })
    if not rows:
        return pd.DataFrame(columns=["#", "Risk", "Score", "Explanation", "Review"])
    df = pd.DataFrame(rows)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df


# ── Main app ──────────────────────────────────────────────────

def main():
    # Header
    st.title("🛍️ Multimodal E-commerce Product Review Analysis Platform")
    st.markdown(
        "AI-powered product quality analysis using sentiment, "
        "defect detection, fake review detection, and aspect analysis."
    )
    st.divider()

    # Check if a history result is currently loaded
    result = st.session_state.get("history_load")
    has_history_result = result is not None

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_url = "http://localhost:8000"

        mode = st.radio(
            "Input mode",
            ["📝 Enter reviews manually", "🔗 Paste Product URL"],
            key="sidebar_mode_select"
        )

        if "prev_mode" in st.session_state and st.session_state["prev_mode"] != mode:
            if "history_load" in st.session_state:
                del st.session_state["history_load"]
                has_history_result = False
                result = None
                st.rerun()
        st.session_state["prev_mode"] = mode

        run_absa = True
        run_fake = True

        st.divider()
        st.subheader("📜 Recent Analysis History")
        try:
            hist_resp = requests.get(f"{api_url}/history", params={"limit": 5}, timeout=3)
            if hist_resp.status_code == 200:
                history_data = hist_resp.json()
                if history_data:
                    for entry in history_data:
                        score = entry.get("quality_score", 0.0)
                        grade = entry.get("grade", "N/A")
                        name_str = entry.get("product_name", "Unknown")
                        asin_str = entry.get("asin", "")

                        if name_str.startswith("Amazon Product ("):
                            display_name = f"Product: {asin_str}"
                        else:
                            display_name = name_str
                            
                        name_short = display_name[:35] + "..." if len(display_name) > 35 else display_name
                        
                        if st.button(name_short, key=f"hist_{entry['id']}", use_container_width=True):
                            loaded_report = None
                            if entry.get("full_report"):
                                try:
                                    loaded_report = json.loads(entry["full_report"])
                                except Exception:
                                    pass
                            
                            if loaded_report:
                                st.session_state["history_load"] = loaded_report
                            else:
                                st.session_state["history_load"] = {
                                    "asin": entry["asin"],
                                    "product_name": entry["product_name"],
                                    "quality_score": score,
                                    "grade": grade,
                                    "n_reviews": entry["n_reviews"],
                                    "summary": entry["summary"],
                                    "breakdown": json.loads(entry["breakdown"]),
                                    "flags": json.loads(entry["flags"]),
                                    "defect_source": entry.get("defect_source", "excluded"),
                                    "timing": {"total": 0.0},
                                    "device": "SQLite DB Cache",
                                    "module_outputs": {
                                        "absa": {},
                                        "fake": [],
                                        "sentiment": [],
                                        "defect": {}
                                    }
                                }
                            st.rerun()
                else:
                    st.caption("No history found in database.")
        except Exception:
            st.caption("Failed to load history from backend.")

        st.divider()
        st.subheader("🔗 API Connection")
        api_url = st.text_input("Backend API Server", value="http://localhost:8000")

        st.divider()
        st.subheader("About")
        st.caption(
            "Built with DistilBERT (sentiment), ResNet-50 (defect), "
            "Isolation Forest + GBM (fake reviews), DeBERTa (ABSA)."
        )

    # ── Category & Aspects Selector Helper ────────────────────
    CATEGORIES = [
        "Auto (detect from reviews)",
        "Electronics",
        "Beauty and Personal Care",
        "Home and Kitchen",
        "Sports and Outdoors",
        "Office Products",
    ]

    selected_category = "Auto (detect from reviews)"
    custom_aspects = []

    def render_category_selector(key_prefix: str):
        if not run_absa:
            return "Auto (detect from reviews)", []
        
        cat = st.selectbox(
            "Product Category",
            CATEGORIES,
            index=0,
            key=f"{key_prefix}_cat_select"
        )
        return cat, []

    product_url = None
    reviews = []
    ratings = None
    product_name = None
    uploaded_image = None

    # ── Input section ──
    if not has_history_result:
        if mode == "🔗 Paste Product URL":
            product_url = st.text_input(
                "Paste product link (Amazon)",
                placeholder="e.g. https://www.amazon.in/dp/B0CS59QFLC"
            )

            selected_category, custom_aspects = render_category_selector("url")

            if product_url:
                st.success(
                    f"🔗 **URL Loaded Successfully**\n\n"
                    f"Click **Analyze Product** to trigger the live scraper to fetch customer reviews."
                )
                reviews = ["URL_SCRAPE_PENDING"]
                ratings = None
                product_name = None
                uploaded_image = None
            else:
                reviews = []
                ratings = None
                product_name = None
                uploaded_image = None

        else:
            c_name, c_cat = st.columns(2)
            with c_name:
                product_name = st.text_input(
                    "Product name", placeholder="e.g. Sony WH-1000XM5 Headphones"
                )
            with c_cat:
                if run_absa:
                    selected_category = st.selectbox(
                        "Product category",
                        CATEGORIES,
                        index=0,
                        key="manual_cat_select",
                        help="Selects which aspects to analyse. 'Auto' extracts aspects from the review text itself — works for any product.",
                    )
                else:
                    selected_category = "Auto (detect from reviews)"
                custom_aspects = []

            st.markdown("---")

            col1, col2 = st.columns([2, 1])

            with col1:
                reviews_text = st.text_area(
                    "Customer reviews",
                    height=250,
                    placeholder=(
                        "Paste reviews here. Use --- on the next line to separate them.\n"
                        "Example:\n"
                        "Battery life is amazing, lasts all day!\n"
                        "The display is vibrant and very responsive.\n"
                        "---\n"
                        "Sound quality is excellent but build feels a bit cheap.\n"
                        "Shipping was fast, happy with the purchase.\n"
                        "---\n"
                        "Stopped working after two weeks. Very disappointed."
                    ),
                )
                if reviews_text:
                    raw_blocks = reviews_text.split("\n---\n")
                    reviews = [b.strip() for b in raw_blocks if b.strip()]
                else:
                    reviews = []

                ratings_text = st.text_input(
                    "Star ratings (optional, comma-separated — one per review)",
                    placeholder="5, 4, 1  (one per review block above)"
                )
                try:
                    ratings = [float(r.strip()) for r in ratings_text.split(",")
                               if r.strip()] if ratings_text else None
                except ValueError:
                    ratings = None
                    st.warning("Invalid ratings — will proceed without them.")

                if reviews:
                    st.caption(f"📝 **{len(reviews)} review(s)** detected" +
                               (f" · {len(ratings)} rating(s)" if ratings else ""))

            with col2:
                uploaded_images = st.file_uploader(
                    "Product image(s) (optional) - Upload only electronics and computer accessories related images",
                    type=["jpg", "jpeg", "png", "webp"],
                    accept_multiple_files=True,
                    help="Upload any number of product images. Assign each to a review using the dropdown that appears below."
                )
                uploaded_image = uploaded_images[0] if uploaded_images else None

                image_review_map = {}  
                if uploaded_images:
                    st.caption(f"📷 {len(uploaded_images)} image(s) — assign each to a review:")
                    num_reviews = max(len(reviews), 1) if reviews else 1
                    review_options = [f"Review {r+1}" for r in range(num_reviews)] + ["(No review)"]

                    cols_per_row = min(len(uploaded_images), 3)
                    img_cols = st.columns(cols_per_row)
                    for i, img_file in enumerate(uploaded_images):
                        with img_cols[i % cols_per_row]:
                            pil_thumb = Image.open(img_file)
                            st.image(pil_thumb, width="stretch",
                                     caption=f"Image #{i+1}")
                            default_idx = min(i, num_reviews - 1)   
                            choice = st.selectbox(
                                f"Assign #\u200b{i+1} to",
                                options=review_options,
                                index=default_idx,
                                key=f"img_assign_{i}",
                                label_visibility="collapsed",
                            )
                            if choice != "(No review)":
                                review_num = int(choice.split(" ")[1]) 
                                image_review_map[i] = review_num

                    if len(uploaded_images) > 1:
                        st.info("💡 Multiple images assigned to the same review → defect results are aggregated (worst-case shown).")

        # ── Analyze button ────────────────────────────────────────
        analyze_clicked = False
        show_button = True 

        if show_button:
            st.divider()
            col_btn, col_info = st.columns([1, 3])
            with col_btn:
                analyze_clicked = st.button(
                    "🔍 Analyze Product",
                    type="primary",
                    width="stretch",
                )

    # ── Run analysis ──────────────────────────────────────────
    result = st.session_state.get("history_load")
    has_history_result = result is not None

    # ── Run analysis ──────────────────────────────────────────
    if not has_history_result and analyze_clicked:
        if "history_load" in st.session_state:
            del st.session_state["history_load"]
            has_history_result = False
            result = None

        if mode == "🔗 Paste Product URL" and not product_url:
            st.warning("⚠️ Please paste a valid Amazon product URL first.")
            st.stop()
        elif mode != "🔗 Paste Product URL" and not reviews:
            st.warning("⚠️ Please write or paste customer reviews in the text box first.")
            st.stop()

        pil_image = None
        pil_all_images = []
        if mode != "🔗 Paste Product URL" and uploaded_images:
            for uf in uploaded_images:
                try:
                    pil_all_images.append(Image.open(uf).convert("RGB"))
                except Exception:
                    pass
            pil_image = pil_all_images[0] if pil_all_images else None

        spinner_msg = (
            "Scraping and analyzing customer reviews via API..."
            if mode == "🔗 Paste Product URL"
            else f"Analyzing {len(reviews)} reviews via API..."
        )
        with st.spinner(spinner_msg):
            try:
                absa_category = selected_category if selected_category != "Custom aspects" else "Auto (detect from reviews)"
                absa_aspects  = custom_aspects if custom_aspects else None

                if uploaded_image is None:
                    if mode == "🔗 Paste Product URL":
                        payload = {
                            "url":            product_url,
                            "run_absa":       run_absa,
                            "category":       absa_category,
                            "custom_aspects": absa_aspects,
                        }
                        response = requests.post(f"{api_url}/analyze/url", json=payload)
                    else:
                        payload = {
                            "reviews":        reviews,
                            "ratings":        ratings,
                            "asin":           "USER_INPUT",
                            "product_name":   product_name or "Product",
                            "run_absa":       run_absa,
                            "category":       absa_category,
                            "custom_aspects": absa_aspects,
                        }
                        response = requests.post(f"{api_url}/analyze/text", json=payload)
                else:
                    files = {
                        "file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)
                    }
                    data = {
                        "reviews":      json.dumps(reviews),
                        "ratings":      json.dumps(ratings) if ratings else json.dumps([]),
                        "asin":         "USER_INPUT",
                        "product_name": product_name or "Product",
                        "run_absa":       run_absa,
                        "category":       absa_category,
                        "custom_aspects": json.dumps(absa_aspects),
                    }
                    response = requests.post(f"{api_url}/analyze", data=data, files=files)
                
                if response.status_code != 200:
                    try:
                        err_msg = response.json().get("detail", response.text)
                    except Exception:
                        err_msg = response.text
                    st.error(f"⚠️ Analysis Failed: {err_msg}")
                    result = None
                else:
                    result = response.json()
            except Exception as e:
                st.error(f"🔌 Connection Error: Make sure the backend server is running on {api_url}.")
                result = None
                
        if result:
            st.session_state["history_load"] = result
            st.rerun()

    # ── Display Results Layout ─────────────────────────────────
    if has_history_result:
        col_space, col_reset = st.columns([5, 1])
        with col_reset:
            if st.button("🧹 Clear & Analyze Another", type="secondary", use_container_width=True):
                if "history_load" in st.session_state:
                    del st.session_state["history_load"]
                st.rerun()



        st.divider()

        # ── Results layout ─────────────────────────────────────
        col_gauge, col_summary = st.columns([1, 2])

        with col_gauge:
            st.markdown("### 📊 Quality Score Breakdown")
            
            # Extract subscores
            bd = result.get("breakdown", {})
            sent_sub = bd.get("sentiment", 0.0)
            fake_sub = bd.get("authenticity", 0.0)
            asp_sub  = bd.get("aspect", 0.0)
            def_sub  = bd.get("defect")
            
            ew = result.get("effective_weights", {})
            w_sent = ew.get("sentiment", 0.35)
            w_asp  = ew.get("aspect", 0.20)
            w_fake = ew.get("authenticity", 0.20)
            w_def  = ew.get("defect", 0.0) if def_sub is not None else 0.0
            
            def_val = float(def_sub) if def_sub is not None else 0.0

            c_sent = sent_sub * w_sent
            c_asp  = asp_sub * w_asp
            c_fake = fake_sub * w_fake
            c_def  = def_val * w_def

            st.markdown(
                f"""
                <div style="background-color:#1e2130; padding:20px; border-radius:10px; font-family:monospace; line-height:1.6;">
                    <p style="margin:5px 0; font-size:15px; color:#cfd2db;">Customer Sentiment <span style="float:right; color:#1d9e75;">+{c_sent:.1f}</span></p>
                    <p style="margin:5px 0; font-size:15px; color:#cfd2db;">Aspect Quality     <span style="float:right; color:#a278ff;">+{c_asp:.1f}</span></p>
                    <p style="margin:5px 0; font-size:15px; color:#cfd2db;">Review Authenticity<span style="float:right; color:#ef9f27;">+{c_fake:.1f}</span></p>
                    <p style="margin:5px 0; font-size:15px; color:#cfd2db;">Image Defects       <span style="float:right; color:#d85a30;">+{c_def:.1f}</span></p>
                    <hr style="border:0; border-top:1px dashed #cfd2db; margin:15px 0;">
                    <p style="margin:5px 0; font-size:18px; font-weight:bold; color:#ffffff;">Final Score <span style="float:right; color:#ffffff; font-size:20px;">{result['quality_score']:.1f}</span></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.caption(
                f"**Score calculation explanation:**\n"
                f"• **Sentiment ({w_sent:.0%})**: Contributed **{c_sent:.1f} pts** based on positive review ratio ({sent_sub:.1f}/100).\n"
                f"• **Aspects ({w_asp:.0%})**: Contributed **{c_asp:.1f} pts** based on absolute satisfaction across key features ({asp_sub:.1f}/100).\n"
                f"• **Authenticity ({w_fake:.0%})**: Contributed **{c_fake:.1f} pts** by factoring out suspicious review patterns ({fake_sub:.1f}/100).\n"
                f"• **Image Condition ({w_def:.0%})**: Contributed **{c_def:.1f} pts** based on anomaly/defect detection ({def_val:.1f}/100)."
            )
            st.write("")
            flags = result.get("flags", [])
            
            # Extract and parse weak aspects into Actionable Recommendations
            aspect_recs = []
            other_flags = []
            
            for flag in flags:
                if flag.startswith("weak aspects:"):
                    raw_aspects = flag.replace("weak aspects:", "").strip().split(",")
                    for asp in raw_aspects:
                        asp = asp.strip().lower()
                        if not asp:
                            continue

                        if "charge" in asp or "charging" in asp:
                            aspect_recs.append("Improve charging speed.")
                        elif "camera" in asp or "photo" in asp:
                            aspect_recs.append("Camera quality receives mixed reviews.")
                        elif "battery" in asp or "power" in asp:
                            aspect_recs.append("Potential buyers concerned about battery.")
                        elif "display" in asp or "screen" in asp:
                            aspect_recs.append("Calibrate display brightness and glare levels.")
                        elif "sound" in asp or "audio" in asp or "volume" in asp:
                            aspect_recs.append("Optimize audio profiles and speaker performance.")
                        elif "comfort" in asp or "fit" in asp:
                            aspect_recs.append("Improve ergonomics and comfort fit.")
                        elif "price" in asp or "cost" in asp:
                            aspect_recs.append("Adjust product pricing to improve market competitiveness.")
                        elif "build" in asp or "material" in asp or "durability" in asp:
                            aspect_recs.append("Address product build quality and material durability.")
                        else:
                            aspect_recs.append(f"{asp.capitalize()} performance receives mixed reviews.")
                else:
                    other_flags.append(flag)
            
            if aspect_recs:
                st.subheader("💡 Recommendations")
                for rec in aspect_recs:
                    st.info(rec)
                    
            if other_flags:
                st.subheader("⚠️ Quality Flags")
                for flag in other_flags:
                    st.warning(flag)
            elif not aspect_recs:
                st.success("No quality warnings detected")

        with col_summary:
            st.subheader("Summary")
            st.write(result["summary"])

            st.subheader("Score Breakdown")
            fig_bars = render_breakdown_bars(result["breakdown"])
            st.plotly_chart(fig_bars, width="stretch", key="breakdown")

        st.divider()

        defect_result = result.get("module_outputs", {}).get("defect", {})
        has_defect = bool(defect_result)

        url_product_image = None
        if mode == "🔗 Paste Product URL" and result.get("product_image_b64"):
            try:
                url_product_image = Image.open(
                    io.BytesIO(base64.b64decode(result["product_image_b64"]))
                ).convert("RGB")
            except Exception:
                pass

        tab_labels = ["💬 Sentiment"]
        if run_fake:   tab_labels.append("🔍 Fake Reviews")
        if run_absa:   tab_labels.append("🎯 Aspects")
        if has_defect: tab_labels.append("📷 Defect")

        tabs = st.tabs(tab_labels)
        tab_idx = 0

        # ── Sentiment tab ──────────────────────────────────────
        with tabs[tab_idx]:
            sentiment_results = result["module_outputs"].get("sentiment", [])
            if sentiment_results:
                col_donut, col_stats = st.columns([1, 1])
                with col_donut:
                    st.subheader("Sentiment Distribution")
                    fig_donut = render_sentiment_donut(sentiment_results)
                    st.plotly_chart(fig_donut, width="stretch", key="donut")

                with col_stats:
                    st.subheader("Stats")
                    meta = result.get("module_outputs", {})
                    from collections import Counter
                    counts = Counter(r.get("label") for r in sentiment_results)
                    n = len(sentiment_results)
                    st.metric("Total Reviews", n)
                    st.metric("Positive", f"{counts.get('positive',0)} ({counts.get('positive',0)/n*100:.0f}%)")
                    st.metric("Neutral",  f"{counts.get('neutral',0)}  ({counts.get('neutral',0)/n*100:.0f}%)")
                    st.metric("Negative", f"{counts.get('negative',0)} ({counts.get('negative',0)/n*100:.0f}%)")

                st.subheader("Review-level Predictions")
                
                aspect_kws = ["battery", "display", "screen", "speed", "performance", "sound", "volume", "comfort", "fit", "camera", "photo", "charging", "charge", "price", "cost", "build", "material", "durability"]
                
                rows_data = []
                for i, r in enumerate(sentiment_results):
                    text = r.get("text", reviews[i] if i < len(reviews) else "")
                    text_lower = text.lower()
                    label = r.get("label", "neutral").upper()
                    conf = r.get("confidence", 0.0)
                    label_with_conf = f"{label} ({conf:.0%})"
                    
                    found_aspects = [kw for kw in aspect_kws if kw in text_lower]
                    if found_aspects:
                        aspects_str = ", ".join(found_aspects[:3])
                        reason = f"Positive because: {aspects_str}" if label == "POSITIVE" else (f"Negative because: {aspects_str}" if label == "NEGATIVE" else f"Mentions: {aspects_str}")
                    else:
                        reason = "General product sentiment"
                        
                    rows_data.append({
                        "Review":     text[:80] + "..." if len(text) > 80 else text,
                        "Label":      label_with_conf,
                        "Reason":     reason,
                        "Uncertain":  "⚠️" if r.get("uncertain") else "✅",
                    })
                
                sent_df = pd.DataFrame(rows_data)
                st.dataframe(sent_df, width="stretch", hide_index=True)

                with st.expander("📖 Read All Full Extracted Reviews"):
                    for i, r in enumerate(sentiment_results):
                        full_text = r.get("text", "")
                        label = r.get("label", "neutral").upper()
                        conf = r.get("confidence", 0.0)
                        emoji = "🟢" if label == "POSITIVE" else "🔴" if label == "NEGATIVE" else "🟡"
                        st.markdown(f"**Review #{i+1}** ({emoji} {label} · {conf:.1%} confidence)")
                        st.markdown(f"> {full_text}")
                        st.divider()
            tab_idx += 1

        # ── Fake Reviews tab ───────────────────────────────────
        if run_fake:
            with tabs[tab_idx]:
                fake_results = result["module_outputs"].get("fake", [])
                if fake_results:
                    n_fake  = sum(1 for r in fake_results if r.get("is_fake"))
                    n_total = len(fake_results)

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Reviews Analyzed", n_total)
                    col_b.metric("Flagged as Suspicious", n_fake,
                                 delta=f"{n_fake/n_total*100:.0f}%",
                                 delta_color="inverse")
                    col_c.metric(
                        "Authenticity Score",
                        f"{result['breakdown'].get('authenticity', 0):.1f}/100"
                    )

                    st.subheader("Risk Assessment per Review")
                    if n_fake == 0:
                        st.success("✅ No suspicious or fake reviews detected in this dataset.")
                        
                    fake_df = render_fake_review_table(reviews, fake_results)
                    if not fake_df.empty:
                        st.dataframe(fake_df, width="stretch", hide_index=True)
                    else:
                        st.info("✓ No review authenticity records available.")
            tab_idx += 1

        # ── ABSA tab ───────────────────────────────────────────
        if run_absa:
            with tabs[tab_idx]:
                absa_results = result["module_outputs"].get("absa", {})
                if absa_results:
                    col_radar, col_bars = st.columns([1, 1])

                    with col_radar:
                        st.subheader("Aspect Radar")
                        fig_radar = render_aspect_radar(absa_results)
                        if fig_radar:
                            st.plotly_chart(fig_radar, width="stretch", key="radar")

                    with col_bars:
                        st.subheader("Aspect Scores")
                        aspect_rows = []
                        for aspect, data in sorted(
                            absa_results.items(),
                            key=lambda x: x[1].get("mean_score", 0),
                            reverse=True,
                        ):
                            score = data.get("mean_score", 0)
                            sentiment = data.get("sentiment", "neutral")
                            emoji = "🟢" if sentiment == "positive" \
                                    else "🔴" if sentiment == "negative" else "🟡"
                            aspect_rows.append({
                                "Aspect":    aspect,
                                "Sentiment": f"{emoji} {sentiment}",
                                "Score":     f"{score:+.2f}",
                                "Reviews":   data.get("review_count", "-"),
                            })
                        st.dataframe(
                            pd.DataFrame(aspect_rows),
                            width="stretch", hide_index=True,
                        )
                else:
                    st.info("No aspects detected in the provided reviews.")
            tab_idx += 1

        # ── Defect tab (grouped by review, all images analyzed) ──
        if has_defect:
            with tabs[tab_idx]:

                # ── Helper: call /analyze/defect for one PIL image ─
                def run_defect_api(pil_img):
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG")
                    buf.seek(0)
                    dr = requests.post(
                        f"{api_url}/analyze/defect",
                        files={"file": ("img.jpg", buf, "image/jpeg")},
                        timeout=60,
                    )
                    return dr.json() if dr.status_code == 200 else None

                # ── Helper: render one image card ──────────────────
                def render_defect_card(img_pil, d_result, label=""):
                    d_label  = d_result.get("label", "unknown")
                    d_conf   = d_result.get("confidence", 0)
                    d_color  = "🔴" if d_label == "defective" else "🟢"
                    overlay_b = d_result.get("overlay_b64")
                    info_c, orig_c, heat_c = st.columns([1, 1.5, 1.5])
                    with info_c:
                        st.markdown(f"**{label}{d_color} {d_label.upper()}**")
                        st.metric("Confidence", f"{d_conf:.1%}")
                        if d_result.get("defect_type"):
                            st.metric("Defect Type", d_result["defect_type"])
                        if d_result.get("uncertain"):
                            st.warning("⚠️ Low confidence")
                    with orig_c:
                        if img_pil:
                            st.caption("Original Image")
                            st.image(img_pil, width="stretch")
                    with heat_c:
                        if overlay_b:
                            heat_img = Image.open(io.BytesIO(base64.b64decode(overlay_b)))
                            st.caption("GradCAM Heatmap")
                            st.image(heat_img, width="stretch",
                                     caption="Red = defect regions (ResNet-50)")
                        else:
                            st.info("Heatmap not generated.")


                # ── URL mode: single image from server ─────────────
                if mode == "🔗 Paste Product URL":
                    display_original = url_product_image
                    render_defect_card(display_original, defect_result,
                                       label="Scraped Product Image — ")

                # ── Manual mode: group images by assigned review ────
                else:
                    from collections import defaultdict
                    groups = defaultdict(list)

                    for img_idx, pil_img in enumerate(pil_all_images):
                        rev_num = image_review_map.get(img_idx, img_idx + 1)
                        groups[rev_num].append((img_idx, pil_img))

                    for rev_num in sorted(groups.keys()):
                        imgs_in_group = groups[rev_num]
                        rev_label = f"Review {rev_num}" if rev_num <= len(reviews) else "Unassigned"
                        st.markdown(f"### 📝 {rev_label}")
                        if reviews and rev_num <= len(reviews):
                            st.caption(f"> {reviews[rev_num - 1][:120]}...")

                        all_results = []
                        for img_idx, pil_img in imgs_in_group:
                            img_label = f"Image #{img_idx + 1} — "
                            with st.spinner(f"Analyzing Image #{img_idx + 1}..."):
                                if img_idx == 0 and defect_result:
                                    d = defect_result
                                else:
                                    d = run_defect_api(pil_img)
                            if d:
                                all_results.append(d)
                                render_defect_card(pil_img, d, label=img_label)
                            else:
                                st.error(f"Defect analysis failed for Image #{img_idx + 1}")

                        if len(all_results) > 1:
                            n_defective = sum(1 for r in all_results if r.get("label") == "defective")
                            avg_conf    = sum(r.get("confidence", 0) for r in all_results) / len(all_results)
                            worst       = "defective" if n_defective > 0 else "normal"
                            agg_color   = "🔴" if worst == "defective" else "🟢"
                            st.info(
                                f"**{rev_label} Summary:** {agg_color} {worst.upper()} "
                                f"({n_defective}/{len(all_results)} images flagged defective, "
                                f"avg confidence {avg_conf:.1%})"
                            )
                        st.divider()


if __name__ == "__main__":
    import sys
    if not st.runtime.exists():
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    else:
        main()
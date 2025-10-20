# ============================================================
# 📦 AMAZON COMPANY HUB — TWO-MONTH INSIGHTS (CMSE 830 MIDTERM)
# ============================================================
# - Non-technical, executive-friendly hub with six tabs:
#     1) How To Use (intro, datasets, why it matters)
#     2) Executive Snapshot (KPIs, trend, mix, highlights, plan)
#     3) Finance & Ops (discount posture, volumes, ops checklist)
#     4) Marketing & Content Strategy (engagement, keywords, platform plan, content calendar)
#     5) Product & Pricing (ratings, price ladder, underperformers)
#     6) Plans & Accountability (cross-team action list + download)
# - Everything is filter-aware by month and category
# - Colors:
#       • green / yellow / red  → results or alerts only
#       • blue                  → informational guidance
# - Data expected in data/data.zip (preferred) or as loose CSVs in /data
#   Required:  amazon_sales_with_two_months_small.csv  (or full)
#   Optional:  amazon_reviews_small.csv, amazon_products_small.csv (or full)
# ============================================================

import os, zipfile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

# Keep TextBlob if available (sentiment), but don't break if it's not installed
try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

# ---------------------------------
# Page header
# ---------------------------------
st.set_page_config(page_title="Amazon Company Hub", layout="wide")
st.title("📦 Amazon Company Hub — Two-Month Insights")

# ---------------------------------
# Color boxes + helpers
# ---------------------------------
PALETTE = {
    "good": {"bg": "#DCFCE7", "bar": "#22C55E", "emoji": "✅"},
    "ok":   {"bg": "#FEF9C3", "bar": "#EAB308", "emoji": "🟨"},
    "bad":  {"bg": "#FEE2E2", "bar": "#EF4444", "emoji": "🟥"},
    "info": {"bg": "#E0E7FF", "bar": "#6366F1", "emoji": "ℹ️"},
    "warn": {"bg": "#FFE4E6", "bar": "#FB7185", "emoji": "⚠️"},
}

def _box(title, headline, text, tone="info"):
    style = PALETTE.get(tone, PALETTE["info"])
    st.markdown(
        f"""
<div style="border-left:8px solid {style['bar']}; background:{style['bg']};
            padding:14px 18px; border-radius:10px; margin:14px 0;">
  <div style="font-weight:700; color:#111827; margin-bottom:6px;">{style['emoji']} {title}</div>
  <div style="font-weight:800; color:#111827; margin-bottom:6px;">{headline}</div>
  <div style="color:#111827; line-height:1.6;">{text}</div>
</div>
""",
        unsafe_allow_html=True,
    )

def box_result(headline, text, level="ok", title="Result"):
    _box(title=title, headline=headline, text=text, tone=level)

def box_info(headline, text, title="Notes"):
    _box(title=title, headline=headline, text=text, tone="info")

def box_warn(headline, text, title="Attention"):
    _box(title=title, headline=headline, text=text, tone="warn")

def fig_notes(why, what, sowhat, nextstep):
    st.markdown(
        f"""
<div style="font-size: 0.92rem; color:#374151; background:#F3F4F6; border-radius:10px; padding:12px 14px; margin-top:-6px;">
  <div><strong>Why:</strong> {why}</div>
  <div><strong>What:</strong> {what}</div>
  <div><strong>So What:</strong> {sowhat}</div>
  <div><strong>What Next:</strong> {nextstep}</div>
</div>
""",
        unsafe_allow_html=True
    )

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def safe_bubble_sizes(series, min_size=8, max_size=40):
    if series is None:
        return None
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    if len(s) == 0:
        return None
    m = float(s.max())
    if m <= 0:
        return None
    return (min_size + (s / m) * (max_size - min_size)).astype(float)

def guard(df, message):
    if df is None or df.empty:
        st.info(message)
        return False
    return True

def pct(n, d):
    if d in [0, None, np.nan] or pd.isna(d):
        return np.nan
    return 100.0 * n / d

# ---------------------------------
# Data loader (Cloud-proof)
# Tries: data/data.zip (top-level or data/ paths), or loose files in /data
# Accepts either *_small.csv or full-size names
# ---------------------------------
DATA_DIR = "data"
DATA_ZIP_PATH = os.path.join(DATA_DIR, "data.zip")

CANDIDATES = {
    "sales": [
        "amazon_sales_with_two_months_small.csv",
        "data/amazon_sales_with_two_months_small.csv",
        "amazon_sales_with_two_months.csv",
        "data/amazon_sales_with_two_months.csv",
        os.path.join(DATA_DIR, "amazon_sales_with_two_months_small.csv"),
        os.path.join(DATA_DIR, "amazon_sales_with_two_months.csv"),
    ],
    "reviews": [
        "amazon_reviews_small.csv",
        "data/amazon_reviews_small.csv",
        "amazon_reviews.csv",
        "data/amazon_reviews.csv",
        os.path.join(DATA_DIR, "amazon_reviews_small.csv"),
        os.path.join(DATA_DIR, "amazon_reviews.csv"),
    ],
    "catalog": [
        "amazon_products_small.csv",
        "data/amazon_products_small.csv",
        "amazon_products.csv",
        "data/amazon_products.csv",
        os.path.join(DATA_DIR, "amazon_products_small.csv"),
        os.path.join(DATA_DIR, "amazon_products.csv"),
    ],
}

def _read_csv_any(src):
    try:
        return pd.read_csv(src, low_memory=False)
    except Exception:
        if isinstance(src, str) and os.path.exists(src):
            for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
                try:
                    return pd.read_csv(src, encoding=enc, low_memory=False)
                except Exception:
                    continue
        return None

def _read_zip_member(zf: zipfile.ZipFile, member: str):
    try:
        with zf.open(member) as f:
            return pd.read_csv(f, low_memory=False)
    except Exception:
        return None

def _read_from_zip(zf: zipfile.ZipFile, candidates):
    members = {m for m in zf.namelist() if not m.startswith("__MACOSX/")}
    for name in candidates:
        if name in members:
            df = _read_zip_member(zf, name)
            if df is not None:
                return df, name.split("/")[-1]
    return None, None

def load_data():
    sales = reviews = catalog = None
    sales_src = reviews_src = catalog_src = None

    if os.path.exists(DATA_ZIP_PATH):
        try:
            with zipfile.ZipFile(DATA_ZIP_PATH) as z:
                sales, sales_src = _read_from_zip(z, CANDIDATES["sales"])
                reviews, reviews_src = _read_from_zip(z, CANDIDATES["reviews"])
                catalog, catalog_src = _read_from_zip(z, CANDIDATES["catalog"])
        except Exception:
            pass

    # fallback to loose files if any missing
    if sales is None:
        for p in CANDIDATES["sales"]:
            if isinstance(p, str) and os.path.exists(p):
                tmp = _read_csv_any(p)
                if tmp is not None:
                    sales, sales_src = tmp, os.path.basename(p)
                    break
    if reviews is None:
        for p in CANDIDATES["reviews"]:
            if isinstance(p, str) and os.path.exists(p):
                tmp = _read_csv_any(p)
                if tmp is not None:
                    reviews, reviews_src = tmp, os.path.basename(p)
                    break
    if catalog is None:
        for p in CANDIDATES["catalog"]:
            if isinstance(p, str) and os.path.exists(p):
                tmp = _read_csv_any(p)
                if tmp is not None:
                    catalog, catalog_src = tmp, os.path.basename(p)
                    break

    return sales, reviews, catalog, sales_src, reviews_src, catalog_src

sales, reviews, catalog, sales_src, reviews_src, catalog_src = load_data()

if sales is None:
    st.error("Missing required data. Provide **data/data.zip** containing "
             "`amazon_sales_with_two_months_small.csv` (or the full file), "
             "or place the CSV in the **data/** folder.")
    st.stop()

# ---------------------------------
# Light cleaning
# ---------------------------------
def prep_sales(df):
    df = df.copy()

    if "month" not in df.columns:
        if "data_collected_at" in df.columns:
            df["month"] = pd.to_datetime(df["data_collected_at"], errors="coerce").dt.to_period("M").astype(str)
        else:
            df["month"] = "2025-08"

    for c in ["purchased_last_month", "product_rating", "total_reviews",
              "discount_percentage", "discounted_price", "original_price"]:
        if c in df.columns:
            df[c] = safe_num(df[c])

    if "product_category" in df.columns:
        df["Category_std"] = df["product_category"].astype(str).str.strip()
    elif "Category_std" not in df.columns:
        df["Category_std"] = "Unknown"

    for src, dst in [("is_best_seller", "flag_best_seller"),
                     ("has_coupon", "flag_coupon"),
                     ("is_sponsored", "flag_sponsored")]:
        if src in df.columns:
            s = df[src].astype(str).str.lower().str.strip()
            df[dst] = s.isin(["true", "1", "yes", "y", "t"])

    return df.dropna(subset=["month"])

def prep_reviews(df):
    if df is None:
        return None
    df = df.copy()

    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col:
        df["month"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").astype(str)

    rate_col = next((c for c in df.columns if "rating" in c.lower()), None)
    if rate_col:
        df["rating"] = safe_num(df[rate_col])

    text_col = next((c for c in df.columns if ("text" in c.lower() or "review" in c.lower())), None)
    if text_col:
        df["review_text"] = df[text_col].astype(str).fillna("")
        if TextBlob is not None:
            try:
                df["sentiment"] = df["review_text"].apply(lambda x: TextBlob(x).sentiment.polarity)
            except Exception:
                df["sentiment"] = np.nan
        else:
            df["sentiment"] = np.nan

    cat_col = next((c for c in df.columns if "category" in c.lower()), None)
    if cat_col:
        df["Category_std"] = df[cat_col].astype(str).str.strip()

    return df

def prep_catalog(df):
    if df is None:
        return None
    df = df.copy()
    for c in ["discounted_price", "actual_price", "original_price"]:
        if c in df.columns:
            df[c] = safe_num(df[c])
    for c in ["product_name", "product_title"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "category" in df.columns and "Category_std" not in df.columns:
        df["Category_std"] = df["category"].astype(str).str.strip()
    return df

sales   = prep_sales(sales)
reviews = prep_reviews(reviews)
catalog = prep_catalog(catalog)

# ---------------------------------
# Sidebar controls
# ---------------------------------
st.sidebar.header("Controls")

months_all = sorted(sales["month"].dropna().unique().tolist())
month_focus = st.sidebar.selectbox("Month", months_all, index=len(months_all) - 1)

prev_idx = max(0, months_all.index(month_focus) - 1)
prev_month = months_all[prev_idx]

cat_counts = (
    sales[sales["month"] == month_focus]
    .groupby("Category_std")["purchased_last_month"]
    .sum()
    .sort_values(ascending=False)
)
labels = [f"{c} ({int(cat_counts[c]):,})" for c in cat_counts.index]
label_to_val = dict(zip(labels, cat_counts.index.tolist()))
chosen = st.sidebar.multiselect("Categories", labels, default=labels[:5] if len(labels) > 5 else labels)
cat_sel = [label_to_val[x] for x in chosen] if chosen else []

sales_f = sales[sales["Category_std"].isin(cat_sel)] if cat_sel else sales.copy()
if sales_f.empty:
    st.warning("That category selection returned no rows. Showing all categories instead.")
    sales_f = sales.copy()

curm  = sales_f[sales_f["month"] == month_focus]
prevm = sales_f[sales_f["month"] == prev_month]

st.sidebar.markdown("### Legend")
st.sidebar.write("✅ Good result")
st.sidebar.write("🟨 Okay result")
st.sidebar.write("🟥 Needs attention")
st.sidebar.write("ℹ️ Information")
st.sidebar.write("⚠️ Reminder")

st.sidebar.markdown("### Reminders")
st.sidebar.write("• Change the month to see different snapshots.")
st.sidebar.write("• Narrow categories to focus the story.")
st.sidebar.write("• Download actions at the end for follow up.")

# ---------------------------------
# KPIs
# ---------------------------------
def kpi_block(df):
    if df is None or df.empty:
        return dict(purchases=0, rating=np.nan, disc=np.nan, reviews=0)
    return dict(
        purchases = safe_num(df.get("purchased_last_month")).sum(skipna=True),
        rating    = safe_num(df.get("product_rating")).mean(),
        disc      = safe_num(df.get("discount_percentage")).mean(),
        reviews   = safe_num(df.get("total_reviews")).sum(skipna=True),
    )

cur = kpi_block(curm)
pre = kpi_block(prevm)
delta_purch  = None if pre["purchases"] in [0, None, np.nan] else cur["purchases"] - pre["purchases"]
delta_rating = None if pd.isna(pre["rating"]) else cur["rating"] - pre["rating"]
delta_disc   = None if pd.isna(pre["disc"]) else cur["disc"] - pre["disc"]

# ---------------------------------
# Tabs
# ---------------------------------
tabs = st.tabs([
    "How To Use",
    "Executive Snapshot",
    "Finance & Ops",
    "Marketing & Content Strategy",
    "Product & Pricing",
    "Plans & Accountability",
])

# ---------------------------
# Tab 1 — How To Use
# ---------------------------
with tabs[0]:
    st.subheader("Welcome")
    st.write(
        "This hub helps executives and teams understand performance without reading code. "
        "Use the left sidebar to pick a month and narrow to a few categories. "
        "Each tab shows clear visuals, a short caption that explains the chart, and color-coded results boxes. "
        "Green means good, yellow means needs a look, red means fix now. Blue boxes give simple guidance."
    )

    st.subheader("Project Goal")
    st.write(
        "The goal is a shared snapshot that combines sales activity, ratings, and review signals. "
        "Executives get a one-page story for the month. Teams get deeper views with the why behind the numbers and a concrete plan. "
        "This format is reusable across companies and roles because it avoids jargon and uses plain language."
    )

    st.subheader("What This App Does")
    st.write(
        "• Merges product-level activity by month with quality signals like ratings and review volume.  \n"
        "• Highlights where volume concentrates and how discounts are used.  \n"
        "• Surfaces review language to inform content ideas and customer education.  \n"
        "• Converts findings into a simple action list with owners and due dates."
    )

    st.subheader("Why I Chose This Project")
    st.write(
        "I want to work on the business side where communication matters. "
        "Many dashboards show numbers but do not tell people what to do next. "
        "This app forces a full data science workflow from cleaning to explanation to action, "
        "and it teaches non-technical teammates how to use data without reading code."
    )

    # Show which datasets are loaded and what each adds (show actual file names)
    st.subheader("Datasets")
    ds_rows = []
    sales_label   = (sales_src or "amazon_sales_with_two_months.csv")
    reviews_label = (reviews_src or "amazon_reviews.csv")
    catalog_label = (catalog_src or "amazon_products.csv")

    ds_rows.append([sales_label, len(sales), "Required",
                    "Sales by product and category with month, rating, reviews, and discount fields."])
    if reviews is not None:
        ds_rows.append([reviews_label, len(reviews), "Optional",
                        "Text reviews, ratings, and sentiment used for keywords and content planning."])
    else:
        ds_rows.append([reviews_label, 0, "Optional", "Not loaded. Add to enable keyword insights and content briefs."])
    if catalog is not None:
        ds_rows.append([catalog_label, len(catalog), "Optional",
                        "Extra product metadata such as titles, categories, and prices."])
    else:
        ds_rows.append([catalog_label, 0, "Optional", "Not loaded."])
    st.dataframe(pd.DataFrame(ds_rows, columns=["File", "Rows", "Role", "What It Adds"]), use_container_width=True)

    # Missing data glance
    st.subheader("Data Quality At A Glance")
    def null_table(df, name):
        if df is None or df.empty:
            return pd.DataFrame({"column": [], "missing_count": [], "missing_%": []})
        t = df.isna().sum().reset_index()
        t.columns = ["column", "missing_count"]
        t["missing_%"] = (100 * t["missing_count"] / len(df)).round(1)
        t = t[t["missing_count"] > 0].sort_values("missing_%", ascending=False)
        t.insert(0, "dataset", name)
        return t
    dq = pd.concat([
        null_table(sales, "sales"),
        null_table(reviews, "reviews"),
        null_table(catalog, "catalog")
    ], ignore_index=True)
    if dq.empty:
        st.write("No missing values detected in the loaded columns.")
    else:
        st.dataframe(dq, use_container_width=True)

    box_info(
        "How We Handle Missing Data",
        "Core KPIs stay raw so we do not hide problems. For charts only, you can turn on a median-by-category imputation in the sidebar. "
        "If a slice has no rows after filters, the page shows a friendly message."
    )

    st.subheader("How To Read This Dashboard")
    st.write(
        "• Start at Executive Snapshot for KPIs, a month-to-month trend, category mix, and highlights.  \n"
        "• Open each team tab for deeper context and a short caption under each chart that says why it matters.  \n"
        "• Go to Plans & Accountability to download the action list and assign owners."
    )

    box_warn(
        "Caution",
        "If you filter to a very small set of categories, some charts may be empty. Widen the selection to see a fuller picture."
    )

# ---------------------------
# Tab 2 — Executive Snapshot
# ---------------------------
with tabs[1]:
    st.header(f"Executive Snapshot — {month_focus}")
    st.caption("📅 Focus for this month, a quick read on the trend, and what to do next.")

    if not guard(curm, "No rows for this month after filters. Pick more categories or another month."):
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Purchases", f"{cur['purchases']:,.0f}", None if delta_purch is None else f"{delta_purch:,.0f}")
    c2.metric("Average Rating", f"{cur['rating']:.2f}" if pd.notna(cur['rating']) else "n/a",
              None if delta_rating is None else f"{delta_rating:+.2f}")
    c3.metric("Average Discount %", f"{cur['disc']:.1f}" if pd.notna(cur['disc']) else "n/a",
              None if delta_disc is None else f"{delta_disc:+.1f}")
    c4.metric("Total Reviews", f"{cur['reviews']:,.0f}")

    trend = (sales_f.groupby("month", as_index=False)["purchased_last_month"].sum().sort_values("month"))
    if guard(trend, "No monthly trend to display."):
        fig = px.line(trend, x="month", y="purchased_last_month", markers=True, title="Purchases Across Months")
        fig.update_layout(xaxis_title="Month", yaxis_title="Purchases (count)")
        st.plotly_chart(fig, use_container_width=True)
        fig_notes(
            why="Show the overall direction month to month.",
            what="Line shows total purchases across the selected categories.",
            sowhat="Confirms if we are growing, flat, or slipping.",
            nextstep="If down, check category deltas and discount posture below."
        )

    if delta_purch is None:
        box_info(
            "Baseline Month",
            f"In {month_focus}, the selected categories recorded {cur['purchases']:,.0f} purchases. "
            "This is the baseline for comparison. Use the Category Contribution to see where performance concentrates."
        )
    elif delta_purch > 0:
        box_result(
            "Month-over-Month Growth",
            f"Purchases grew by {delta_purch:,.0f} versus {prev_month}. "
            "The lift likely came from category mix and clearer product pages. "
            "Keep momentum with creator videos and concise value bullets at the top of each page.",
            "good"
        )
    else:
        box_result(
            "Purchases Decreased",
            f"Purchases fell by {abs(delta_purch):,.0f} versus {prev_month}. "
            "Check category mix and reduce reliance on discounts. Improve page clarity with short demos and verified reviews.",
            "bad"
        )

    mix = (
        curm.groupby("Category_std", as_index=False)["purchased_last_month"]
        .sum()
        .sort_values("purchased_last_month", ascending=False)
        .head(12)
    )
    if guard(mix, "No category contribution to display."):
        fig2 = px.bar(mix, x="Category_std", y="purchased_last_month", text_auto=".2s",
                      title=f"Top 12 Categories in {month_focus}")
        fig2.update_layout(xaxis_title="Category", yaxis_title="Purchases")
        st.plotly_chart(fig2, use_container_width=True)
        fig_notes(
            why="Find where purchases concentrate this month.",
            what="Bars rank categories by total purchases.",
            sowhat="Shows the leaders to protect and the mid-pack to develop.",
            nextstep="Keep leaders stocked and featured. Test bundles in the middle group."
        )
        top_cat = mix.iloc[0]
        box_result(
            "Leader Category",
            f"{top_cat['Category_std']} leads this month. Protect the lead with strong visuals and clear benefits.",
            "good"
        )

    g = (sales_f.groupby(["month", "Category_std"])["purchased_last_month"].sum().reset_index())
    if guard(g, "No data for month-over-month category change."):
        last_two = g[g["month"].isin([prev_month, month_focus])]
        if guard(last_two, "Not enough months to compare."):
            pivot = last_two.pivot(index="Category_std", columns="month", values="purchased_last_month").fillna(0)
            if prev_month in pivot.columns and month_focus in pivot.columns:
                pivot["delta"] = pivot[month_focus] - pivot[prev_month]
                ranked = pivot.sort_values("delta", ascending=False).reset_index()
                fig3 = px.bar(ranked, x="Category_std", y="delta", text_auto=".2s",
                              title=f"Month-over-Month Change ({prev_month} → {month_focus})")
                fig3.update_layout(xaxis_title="Category", yaxis_title="Δ Purchases")
                st.plotly_chart(fig3, use_container_width=True)
                fig_notes(
                    why="Spot what changed the most month to month.",
                    what="Bars show increases and decreases by category.",
                    sowhat="Explains the headline KPI move.",
                    nextstep="Double-click the biggest moves in each team tab."
                )
                winner = ranked.iloc[0]; loser = ranked.iloc[-1]
                box_info(
                    "Monthly Highlights",
                    f"📈 Biggest gain: {winner['Category_std']} (+{winner['delta']:,.0f}). "
                    f"📉 Biggest drop: {loser['Category_std']} ({abs(loser['delta']):,.0f} down). "
                    "Review images, copy, and return reasons for the declining category."
                )

    # Short action plan blurb
    recs = []
    if delta_purch and delta_purch < 0:
        recs.append("🔴 Finance: Reduce discount depth and test bundles with clear benefits.")
    if pd.notna(cur["rating"]) and cur["rating"] < 4.0:
        recs.append("🟡 Marketing: Add 20–30s demo videos and highlight verified reviews.")
    if pd.notna(cur["disc"]) and cur["disc"] > 25:
        recs.append("🟡 Product: Improve above-the-fold value bullets so pages convert at lighter promos.")
    if not recs:
        recs.append("✅ Keep scaling creator content and keep PDP copy short, visual, and specific.")
    box_result("Next Month Plan", " • " + "\n • ".join(recs), "ok")

# ---------------------------
# Tab 3 — Finance & Ops
# ---------------------------
with tabs[2]:
    st.header(f"Finance & Operations — {month_focus}")
    st.caption("📊 This tab focuses on volume drivers, discount posture, and operational guardrails.")

    if not guard(curm, "No rows for this month after filters."):
        st.stop()

    by_cat = (
        curm.groupby("Category_std", as_index=False)
        .agg(purchases=("purchased_last_month", "sum"),
             avg_disc=("discount_percentage", "mean"))
    )
    if guard(by_cat, "No category statistics available."):
        fig1 = px.scatter(by_cat, x="avg_disc", y="purchases", color="Category_std",
                          title="Purchases vs Average Discount by Category")
        fig1.update_layout(xaxis_title="Average Discount %", yaxis_title="Purchases (count)")
        st.plotly_chart(fig1, use_container_width=True)
        fig_notes(
            why="Check if volume depends on deep discounts.",
            what="Each dot is a category with its average discount and purchases.",
            sowhat="If high volume sits at high discount, margin is at risk.",
            nextstep="Move some gains to bundles and sharper product value copy."
        )
        lead = by_cat.sort_values("purchases", ascending=False).head(1)
        if not lead.empty:
            r = lead.iloc[0]
            level = "good" if r["avg_disc"] < 25 else "ok"
            box_result(
                "Volume Driver",
                f"{r['Category_std']} drives the most purchases with an average discount of {r['avg_disc']:.1f}%.",
                level
            )

    if "discount_percentage" in curm.columns and curm["discount_percentage"].notna().any():
        fig2 = px.box(curm.dropna(subset=["discount_percentage"]),
                      x="Category_std", y="discount_percentage", color="Category_std",
                      title="Discount % Distribution by Category")
        fig2.update_layout(xaxis_title="", yaxis_title="Discount %")
        st.plotly_chart(fig2, use_container_width=True)
        fig_notes(
            why="See how discounts are spread within each category.",
            what="Boxes show the range and median discount by category.",
            sowhat="Heavy tails or high medians hint at promo dependence.",
            nextstep="Tighten discount bands and fix clarity on pages before promos."
        )
        tone = "bad" if (pd.notna(cur["disc"]) and cur["disc"] >= 30) else "ok"
        box_result(
            "Discount Posture",
            f"Average discount is {cur['disc']:.1f}%." if pd.notna(cur["disc"]) else "Discount level not available.",
            tone
        )
        if tone == "bad":
            box_warn(
                "Caution On Discount Depth",
                "Heavy average discounts can hide product clarity problems. Fix value messaging before raising promo depth."
            )

    ops = [
        "✅ Check stock and shipping for top 3 categories with the largest increase.",
        "✅ Review return reasons where discounts rose but volume did not.",
        "✅ Confirm delivery promise on PDP matches actual average delivery time.",
    ]
    box_info("Operations Checklist", " • " + "\n • ".join(ops))

# -------------------------------------------
# Tab 4 — Marketing & Content Strategy
# -------------------------------------------
with tabs[3]:
    st.header(f"Marketing & Content Strategy — {month_focus}")
    st.caption("🎬 Turn ratings and reviews into a content plan teams can use.")

    if not guard(curm, "No rows for this month after filters."):
        st.stop()

    box_info(
        "Focus For This Month",
        "Identify categories that convince without heavy discounts, clarify topics that block conversion, and publish a simple 7-day plan."
    )

    plot_df = curm.dropna(subset=["product_rating"])
    if guard(plot_df, "No rating values to visualize."):
        sizes = safe_bubble_sizes(plot_df.get("purchased_last_month"))
        if sizes is None:
            figm1 = px.scatter(plot_df, x="product_rating", y="total_reviews",
                               color="Category_std", opacity=0.9,
                               title="Engagement Map — Rating vs Reviews")
        else:
            figm1 = px.scatter(plot_df, x="product_rating", y="total_reviews",
                               color="Category_std", size=sizes, opacity=0.9,
                               title="Engagement Map — Rating vs Reviews (Bubble = Purchases)")
        figm1.update_layout(xaxis_title="Average Rating (★)", yaxis_title="Review Volume")
        st.plotly_chart(figm1, use_container_width=True)
        fig_notes(
            why="Find products people like and talk about.",
            what="Dots show rating on the x-axis and review volume on the y-axis.",
            sowhat="High rating with many reviews is a signal to scale with social proof.",
            nextstep="Add short creator videos and pin top reviews on the product page."
        )
        top = (plot_df.assign(score=lambda d: d["product_rating"].rank(pct=True)
                                           + safe_num(d["total_reviews"]).rank(pct=True))
                      .sort_values("score", ascending=False).head(1))
        if not top.empty:
            box_result(
                "Advocacy Engine",
                "Products with high ratings and many reviews grow on their own. "
                "Run creator shorts and position social proof above the fold to compound this advantage.",
                "good"
            )

    # Conversion proxy
    conv = (curm.groupby("Category_std", as_index=False)
                 .agg(purchases=("purchased_last_month", "sum"),
                      reviews=("total_reviews", "sum")))
    if guard(conv, "No review totals available for conversion proxy."):
        conv["purch_per_review"] = conv.apply(
            lambda r: r["purchases"] / r["reviews"] if r["reviews"] not in [0, np.nan] else np.nan,
            axis=1
        )
        conv = conv.dropna(subset=["purch_per_review"]).sort_values("purch_per_review", ascending=False).head(10)
        if guard(conv, "No conversion proxy to show."):
            figc = px.bar(conv, x="Category_std", y="purch_per_review", text_auto=".2f",
                          title="Purchases Per Review (Conversion Proxy)")
            figc.update_layout(xaxis_title="Category", yaxis_title="Purchases / Review")
            st.plotly_chart(figc, use_container_width=True)
            fig_notes(
                why="Approximate how efficiently reviews convert to purchases.",
                what="Bars show purchases per review by category.",
                sowhat="High values are strong bets for paid and organic placement.",
                nextstep="Feature the top categories across paid and owned channels."
            )
            lead = conv.iloc[0]
            box_result(
                "High Intent Signal",
                f"{lead['Category_std']} converts reviews to purchases efficiently.",
                "good"
            )

    # Keywords + sentiment (if reviews present)
    if reviews is not None:
        rcur = reviews.copy()
        if "month" in rcur.columns:
            rcur = rcur[rcur["month"] == month_focus]
        if "Category_std" in rcur.columns and cat_sel:
            rcur = rcur[rcur["Category_std"].isin(cat_sel)]
        if guard(rcur, "No reviews available for this slice."):
            if "review_text" in rcur.columns:
                # Sentiment if TextBlob is available
                if TextBlob is not None:
                    try:
                        rcur["sentiment"] = rcur["review_text"].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
                    except Exception:
                        rcur["sentiment"] = np.nan

                low = rcur["review_text"].astype(str).str.lower()
                words = pd.Series(" ".join(low.tolist()))
                words = words.str.replace(r"[^a-z\s]", "", regex=True).str.split()
                words = pd.Series([w for lst in words.dropna().tolist() for w in lst])
                stop = set("the and to a of is it for in on my this that very too but not as be are you i so at have its they we just would".split())
                topw = words[~words.isin(stop)].value_counts().head(12).reset_index()
                topw.columns = ["word", "count"]
                if guard(topw, "No frequent words to display."):
                    figw = px.bar(topw, x="word", y="count", text_auto=".2s", title="Common Customer Words (This Month)")
                    st.plotly_chart(figw, use_container_width=True)
                    fig_notes(
                        why="Turn common complaints and questions into content.",
                        what="Bars show the most frequent words in reviews this month.",
                        sowhat="These are plain-language topics customers care about.",
                        nextstep="Record short clips that answer these questions directly."
                    )
                    box_warn("Caution", "Do not copy words blindly. Skim a few reviews to confirm real meaning before posting.")

    # Recommended Platform Mix (restored)
    plat = (curm.groupby("Category_std", as_index=False)
                 .agg(rating=("product_rating", "mean"),
                      reviews=("total_reviews", "sum"),
                      price=("discounted_price", "median")))
    if guard(plat, "Not enough fields to build a platform plan."):
        def platform_row(r):
            tiktok = reels = youtube = paid = 0.0
            if pd.notna(r["rating"]) and pd.notna(r["reviews"]):
                if r["rating"] >= 4.2 and r["reviews"] >= np.nanpercentile(plat["reviews"], 60):
                    tiktok += 0.35; reels += 0.35
                if (r["rating"] < 4.2 and pd.notna(r["price"]) and r["price"] >= np.nanmedian(plat["price"])):
                    youtube += 0.4
                if r["reviews"] < np.nanpercentile(plat["reviews"], 40) and r["rating"] < 4.1:
                    paid += 0.4
            s = tiktok + reels + youtube + paid
            if s == 0: s = 1.0
            return pd.Series({
                "TikTok": round(100 * tiktok / s, 1),
                "IG Reels": round(100 * reels / s, 1),
                "YouTube": round(100 * youtube / s, 1),
                "Paid Search": round(100 * paid / s, 1),
            })
        mix = plat.join(plat.apply(platform_row, axis=1))
        melt = mix.melt(id_vars=["Category_std"],
                        value_vars=["TikTok", "IG Reels", "YouTube", "Paid Search"],
                        var_name="Platform", value_name="Share %")
        figpm = px.bar(melt, x="Category_std", y="Share %", color="Platform", barmode="stack",
                       title="Recommended Platform Mix By Category")
        st.plotly_chart(figpm, use_container_width=True)
        fig_notes(
            why="Give a simple starting point for content allocation.",
            what="Stacked bars show suggested share by platform for each category.",
            sowhat="Avoids guessing and helps the team start fast.",
            nextstep="Use this split for the next sprint and adjust based on results."
        )

    st.markdown("### Seven-Day Content Plan")
    top_cats = (curm.groupby("Category_std")["purchased_last_month"]
                .sum().sort_values(ascending=False).head(3).index.tolist())
    if not top_cats:
        top_cats = ["Top Category"]
    cal = [
        f"📅 Day 1 — Creator short (15–30s): “Why {top_cats[0]} solves a real problem.” Show it in use.",
        f"📅 Day 2 — Product page demo: unbox {top_cats[0]} and cover the top three benefits in under 30 seconds.",
        f"📅 Day 3 — Social proof reel: put the best review on screen over product footage.",
        f"📅 Day 4 — Quick fix: answer the most common question from reviews in one short clip.",
        f"📅 Day 5 — Compare: {top_cats[0]} vs {top_cats[1] if len(top_cats)>1 else 'another option'} — who is each for.",
        f"📅 Day 6 — Daily use montage: three quick shots that show real life use cases.",
        f"📅 Day 7 — Community Q&A: answer two real customer questions in 30 seconds each.",
    ]
    box_info("Publishing Guide", "Post on TikTok and IG Reels. Pin the best demo to the product page hero video.")
    st.markdown("\n".join([f"- {row}" for row in cal]))

# ---------------------------
# Tab 5 — Product & Pricing
# ---------------------------
with tabs[4]:
    st.header(f"Product & Pricing — {month_focus}")
    st.caption("🧪 Quality signals and how prices step down from list.")

    if not guard(curm, "No rows for this month after filters."):
        st.stop()

    if "product_rating" in curm.columns and curm["product_rating"].notna().any():
        figp1 = px.box(curm.dropna(subset=["product_rating"]),
                       x="Category_std", y="product_rating", color="Category_std",
                       title="Product Ratings by Category")
        figp1.update_layout(xaxis_title="", yaxis_title="Rating (★)")
        st.plotly_chart(figp1, use_container_width=True)
        fig_notes(
            why="Assess perceived quality by category.",
            what="Boxes show distribution and median rating.",
            sowhat="Low medians signal friction or wrong expectations.",
            nextstep="Add short demos and clarify sizing and care details."
        )
        worst = curm.groupby("Category_std")["product_rating"].mean().sort_values().head(1)
        if not worst.empty:
            wcat, wr = worst.index[0], worst.values[0]
            level = "bad" if wr < 4.0 else "ok"
            box_result("Quality Signal", f"{wcat} averages {wr:.2f} stars. Add demos and clear benefit bullets.", level)

    if {"original_price", "discounted_price"}.issubset(curm.columns):
        ladder = (curm[["Category_std", "original_price", "discounted_price"]]
                  .dropna().groupby("Category_std").median().reset_index())
        if guard(ladder, "No price ladder to show."):
            figp2 = px.scatter(ladder, x="original_price", y="discounted_price", color="Category_std",
                               title="Price Ladder (Median Original vs Discounted)")
            figp2.update_layout(xaxis_title="Original Price (median)", yaxis_title="Discounted Price (median)")
            st.plotly_chart(figp2, use_container_width=True)
            fig_notes(
                why="See how far discounting steps prices down from list.",
                what="Each dot is a category with median original and discounted price.",
                sowhat="Large gaps suggest reliance on promos.",
                nextstep="Test lighter promos paired with clearer product value bullets."
            )
            heavy = ladder[ladder["discounted_price"] <= 0.7 * ladder["original_price"]]
            tone = "ok" if heavy.empty else "bad"
            msg = "Median discounted prices sit near list. This is healthy." if heavy.empty else \
                  "Some categories lean on deep discounts. Test lighter promos with sharper value messages."
            box_result("Pricing Posture", msg, tone)

    under = (curm.assign(rating=curm["product_rating"],
                         purch=curm["purchased_last_month"])
                  .dropna(subset=["rating", "purch"]))
    if not under.empty:
        under = under.sort_values(["rating", "purch"], ascending=[True, False])
        under = under.drop_duplicates(subset=["product_title"], keep="first").head(10)
        view = under[["product_title", "Category_std", "rating", "purch", "discount_percentage"]]
        st.markdown("### Underperformers Needing Attention")
        st.dataframe(view.rename(columns={
            "product_title": "Product",
            "Category_std": "Category",
            "rating": "Rating (★)",
            "purch": "Purchases",
            "discount_percentage": "Discount %",
        }), use_container_width=True)
        box_result(
            "What To Fix First",
            "Start with low-rating, high-volume items. Clarify expectations and show the product in use.",
            "bad"
        )
        box_warn(
            "Watch Returns & CS Load",
            "Low-rating, high-volume items often create support tickets and returns. Fix these first to protect margin."
        )

# ---------------------------
# Tab 6 — Plans & Accountability
# ---------------------------
with tabs[5]:
    st.header("Plans & Accountability")
    st.caption("🗂️ Concrete items with owners and due dates that match this month’s story.")

    actions = []

    g = (sales_f.groupby(["month", "Category_std"])["purchased_last_month"].sum()
         .reset_index().sort_values("month"))
    if not g.empty and prev_month in g["month"].values and month_focus in g["month"].values:
        pm = g[g["month"] == prev_month].set_index("Category_std")["purchased_last_month"]
        cm = g[g["month"] == month_focus].set_index("Category_std")["purchased_last_month"]
        both = pd.concat([pm, cm], axis=1); both.columns = ["prev", "cur"]
        both["delta"] = both["cur"] - both["prev"]; both = both.dropna()
        if not both.empty:
            top = both["delta"].abs().sort_values(ascending=False).head(1)
            cat = top.index[0]; d = both.loc[cat, "delta"]; direction = "up" if d > 0 else "down"
            actions.append(dict(team="Executive",
                                action=f"{cat} moved {direction} by {abs(d):,.0f} MoM. Set a numeric target and an owner.",
                                priority="High", due=(date.today() + timedelta(days=7)).isoformat(), status="Open"))

    if cur["reviews"] and pre["reviews"] and cur["reviews"] < pre["reviews"]:
        actions.append(dict(team="Marketing & Content",
                            action="Review volume slipped. Refresh creatives, seed UGC, and add social proof blocks on PDP.",
                            priority="High", due=(date.today() + timedelta(days=10)).isoformat(), status="Open"))

    if pd.notna(cur["disc"]) and cur["disc"] >= 30:
        actions.append(dict(team="Finance & Ops",
                            action=f"Average discount {cur['disc']:.1f}%. Test lighter discounts with bundles and sharper value bullets.",
                            priority="Medium", due=(date.today() + timedelta(days=14)).isoformat(), status="Open"))

    lowcat = curm.groupby("Category_std")["product_rating"].mean().sort_values().head(1)
    if not lowcat.empty and lowcat.values[0] < 4.0:
        actions.append(dict(team="Product & Pricing",
                            action=f"{lowcat.index[0]} averages {lowcat.values[0]:.2f}★. Add 20–30s demos and clarify sizing and care.",
                            priority="Medium", due=(date.today() + timedelta(days=14)).isoformat(), status="Open"))

    actions.append(dict(team="Cross-Team",
                        action="Record a 10-minute Loom walkthrough linking to this hub and the key wins and issues.",
                        priority="Low", due=(date.today() + timedelta(days=5)).isoformat(), status="Open"))

    if actions:
        act_df = pd.DataFrame(actions, columns=["team", "action", "priority", "due", "status"])
        st.dataframe(act_df, use_container_width=True)
        st.download_button(
            "Download Action Plan (CSV)",
            act_df.to_csv(index=False).encode("utf-8"),
            file_name="action_plan.csv",
            mime="text/csv"
        )
        box_info("How To Use This List", "Share the CSV in your team channel and assign owners immediately.")
    else:
        box_warn("No Actions Generated", "Try a different month or widen categories to reveal more signals.")

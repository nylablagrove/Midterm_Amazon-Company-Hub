# ============================================================
#  AMAZON COMPANY HUB - TWO MONTH INSIGHTS (CMSE 830 MIDTERM)
# ============================================================
# what this file delivers (plain english):
#   - a non-technical, executive-friendly hub with six tabs:
#       1) how to use (full intro, datasets, why it matters)
#       2) executive snapshot (kpis, trend, mix, highlights, plan)
#       3) finance and ops (discount posture, volumes, ops checklist)
#       4) marketing and content strategy (engagement, keywords, platform plan, content calendar)
#       5) product and pricing (ratings, price ladder, underperformers)
#       6) plans and accountability (action list + download)
#   - everything respects the month and category filters
#   - color rules:
#         green / yellow / red  -> results or alerts only
#         blue                  -> information and guidance
#   - comments above blocks, clear sentences, no em dashes
#   - data is expected in data/data.zip (small csvs) or as loose csvs in /data
# ============================================================

# imports for the ui and analysis
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

# make textblob optional so cloud does not build heavy pattern dependency
try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

# page setup
st.set_page_config(page_title="Amazon Company Hub", layout="wide")
st.title("📦 Amazon Company Hub - Two Month Insights")

# a small color system for reusable boxes
PALETTE = {
    "good": {"bg": "#DCFCE7", "bar": "#22C55E", "emoji": "✅"},
    "ok":   {"bg": "#FEF9C3", "bar": "#EAB308", "emoji": "🟨"},
    "bad":  {"bg": "#FEE2E2", "bar": "#EF4444", "emoji": "🟥"},
    "info": {"bg": "#E0E7FF", "bar": "#6366F1", "emoji": "ℹ️"},
    "warn": {"bg": "#FFE4E6", "bar": "#FB7185", "emoji": "⚠️"},
}

# helper to render callout boxes
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

# figure caption helper that sits under each chart
def fig_notes(why, what, sowhat, nextstep):
    st.markdown(
        f"""
<div style="font-size:0.92rem; color:#374151; background:#F3F4F6; border-radius:10px; padding:12px 14px; margin-top:-6px;">
  <div><strong>Why:</strong> {why}</div>
  <div><strong>What:</strong> {what}</div>
  <div><strong>So What:</strong> {sowhat}</div>
  <div><strong>What Next:</strong> {nextstep}</div>
</div>
""",
        unsafe_allow_html=True
    )

# small numeric and guard helpers
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

# ------------------------------------------------------------
# data loading that works on cloud and local
# accepts either:
#   data/data.zip with small files
#   or loose csvs inside /data
# handles files stored at top level of zip or in data/ inside zip
# ignores __MACOSX entries
# ------------------------------------------------------------
import os, zipfile

DATA_DIR = "data"
DATA_ZIP = os.path.join(DATA_DIR, "data.zip")

# candidates we will try in order for each dataset
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

def _read_csv_any(path_or_filelike):
    try:
        return pd.read_csv(path_or_filelike, low_memory=False)
    except Exception:
        # try a couple of encodings if it is a real path string
        if isinstance(path_or_filelike, str) and os.path.exists(path_or_filelike):
            for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
                try:
                    return pd.read_csv(path_or_filelike, encoding=enc, low_memory=False)
                except Exception:
                    continue
        return None

def _read_from_zip(zf: zipfile.ZipFile, candidates):
    members = set([m for m in zf.namelist() if not m.startswith("__MACOSX/")])
    for name in candidates:
        if name in members:
            try:
                with zf.open(name) as f:
                    return _read_csv_any(f), name
            except Exception:
                continue
    return None, None

def load_data():
    sales = reviews = catalog = None
    sales_name = reviews_name = catalog_name = None

    if os.path.exists(DATA_ZIP):
        try:
            with zipfile.ZipFile(DATA_ZIP) as z:
                sales,  sales_name  = _read_from_zip(z, CANDIDATES["sales"])
                reviews, reviews_name = _read_from_zip(z, CANDIDATES["reviews"])
                catalog, catalog_name = _read_from_zip(z, CANDIDATES["catalog"])
        except Exception as e:
            st.warning(f"could not open data.zip, falling back to loose files. details: {e}")

    # if any still missing, try loose files on disk
    if sales is None:
        for p in CANDIDATES["sales"]:
            if isinstance(p, str) and os.path.exists(p):
                sales = _read_csv_any(p)
                if sales is not None:
                    sales_name = p
                    break
    if reviews is None:
        for p in CANDIDATES["reviews"]:
            if isinstance(p, str) and os.path.exists(p):
                reviews = _read_csv_any(p)
                if reviews is not None:
                    reviews_name = p
                    break
    if catalog is None:
        for p in CANDIDATES["catalog"]:
            if isinstance(p, str) and os.path.exists(p):
                catalog = _read_csv_any(p)
                if catalog is not None:
                    catalog_name = p
                    break

    return (sales, reviews, catalog, sales_name, reviews_name, catalog_name)

# actually load the data now
sales, reviews, catalog, sales_src, reviews_src, catalog_src = load_data()

# sales is required to run anything
if sales is None:
    st.error("data not found. expected data/data.zip with small files or csvs inside /data. "
             "required file: amazon_sales_with_two_months_small.csv")
    st.stop()

# ------------------------------------------------------------
# light standardization helpers so plots and summaries are simple
# ------------------------------------------------------------
def prep_sales(df):
    df = df.copy()

    # month column
    if "month" not in df.columns:
        if "data_collected_at" in df.columns:
            df["month"] = pd.to_datetime(df["data_collected_at"], errors="coerce").dt.to_period("M").astype(str)
        else:
            df["month"] = "2025-08"

    # numerics
    for c in ["purchased_last_month", "product_rating", "total_reviews",
              "discount_percentage", "discounted_price", "original_price"]:
        if c in df.columns:
            df[c] = safe_num(df[c])

    # category normalize
    if "product_category" in df.columns:
        df["Category_std"] = df["product_category"].astype(str).str.strip()
    elif "Category_std" not in df.columns:
        df["Category_std"] = "Unknown"

    # promo flags if present
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

    # month guess from any date-like column
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col:
        df["month"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").astype(str)

    # rating
    rate_col = next((c for c in df.columns if "rating" in c.lower()), None)
    if rate_col:
        df["rating"] = safe_num(df[rate_col])

    # text and optional sentiment
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

    # category normalize if present
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

# apply cleaning
sales   = prep_sales(sales)
reviews = prep_reviews(reviews)
catalog = prep_catalog(catalog)

# ------------------------------------------------------------
# sidebar controls
# ------------------------------------------------------------
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
    st.warning("that category selection returned no rows. showing all categories instead.")
    sales_f = sales.copy()

curm  = sales_f[sales_f["month"] == month_focus]
prevm = sales_f[sales_f["month"] == prev_month]

st.sidebar.markdown("### Legend")
st.sidebar.write("✅ good result")
st.sidebar.write("🟨 okay result")
st.sidebar.write("🟥 needs attention")
st.sidebar.write("ℹ️ information")
st.sidebar.write("⚠️ reminder")

st.sidebar.markdown("### Reminders")
st.sidebar.write("• change the month to see different snapshots")
st.sidebar.write("• narrow categories to focus the story")
st.sidebar.write("• download actions at the end for follow up")

# kpi helpers
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

# ------------------------------------------------------------
# tabs
# ------------------------------------------------------------
tabs = st.tabs([
    "How To Use",
    "Executive Snapshot",
    "Finance and Ops",
    "Marketing and Content Strategy",
    "Product and Pricing",
    "Plans and Accountability",
])

# ============================================================
# tab 1 - how to use
# ============================================================
with tabs[0]:
    st.subheader("Welcome")
    st.write(
        "this hub helps non technical teammates understand performance without reading code. "
        "use the left sidebar to pick a month and a few categories. "
        "each tab has a clear chart with a short caption and color coded callouts."
    )

    st.subheader("Project Goal")
    st.write(
        "combine sales activity, ratings, and review signals into a one page story for the month. "
        "leaders get a quick read. teams get the why and concrete next steps."
    )

    st.subheader("What This App Does")
    st.write(
        "• merges product level activity by month with quality signals like ratings and review volume  \n"
        "• highlights where volume concentrates and how discounts are used  \n"
        "• surfaces review language for content ideas and customer education  \n"
        "• converts findings into a simple action list with owners and due dates"
    )

    st.subheader("Why I Chose This Project")
    st.write(
        "i want to practice the full path from data to action. many dashboards show numbers but not what to do next. "
        "this format makes the story plain and keeps jargon out."
    )

    st.subheader("Datasets Loaded")
    ds = []
    ds.append([sales_src or "amazon_sales_with_two_months_small.csv", len(sales), "required", "sales by product and category with month, rating, reviews, discount fields"])
    ds.append([reviews_src or "amazon_reviews_small.csv (optional)", 0 if reviews is None else len(reviews), "optional", "review text and basic sentiment for content planning"])
    ds.append([catalog_src or "amazon_products_small.csv (optional)", 0 if catalog is None else len(catalog), "optional", "extra product metadata like titles, prices, categories"])
    st.dataframe(pd.DataFrame(ds, columns=["file", "rows", "role", "what it adds"]), use_container_width=True)

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
        st.write("no missing values detected in the loaded columns.")
    else:
        st.dataframe(dq, use_container_width=True)

    box_info(
        "How We Handle Missing Data",
        "core kpis stay raw so we do not hide problems. for charts only, you can turn on median by category imputation in code if needed. "
        "if a slice has no rows after filters, the page shows a friendly note."
    )

    st.subheader("How To Read This Dashboard")
    st.write(
        "• start at executive snapshot for kpis, trend, mix, and highlights  \n"
        "• open team tabs for deeper context with a caption under each figure  \n"
        "• go to plans and accountability to download the action list"
    )

    box_warn("Caution", "if you filter to a very small set of categories, some charts may be empty. widen the selection for a fuller picture.")

# ============================================================
# tab 2 - executive snapshot
# ============================================================
with tabs[1]:
    st.header(f"Executive Snapshot - {month_focus}")
    st.caption("focus for this month, quick trend read, and what to do next")

    if not guard(curm, "no rows for this month after filters. pick more categories or another month."):
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Purchases", f"{cur['purchases']:,.0f}", None if delta_purch is None else f"{delta_purch:,.0f}")
    c2.metric("Average Rating", f"{cur['rating']:.2f}" if pd.notna(cur['rating']) else "n/a",
              None if delta_rating is None else f"{delta_rating:+.2f}")
    c3.metric("Average Discount %", f"{cur['disc']:.1f}" if pd.notna(cur['disc']) else "n/a",
              None if delta_disc is None else f"{delta_disc:+.1f}")
    c4.metric("Total Reviews", f"{cur['reviews']:,.0f}")

    trend = (sales_f.groupby("month", as_index=False)["purchased_last_month"].sum().sort_values("month"))
    if guard(trend, "no monthly trend to display."):
        fig = px.line(trend, x="month", y="purchased_last_month", markers=True, title="Purchases Across Months")
        fig.update_layout(xaxis_title="Month", yaxis_title="Purchases")
        st.plotly_chart(fig, use_container_width=True)
        fig_notes("show the overall direction by month",
                  "line shows total purchases for the selected categories",
                  "confirms if we are growing, flat, or slipping",
                  "if down, check category changes and discount posture below")

    if delta_purch is None:
        box_info("Baseline Month",
                 f"in {month_focus} the selected categories recorded {cur['purchases']:,.0f} purchases. "
                 "use the category contribution to see where performance concentrates.")
    elif delta_purch > 0:
        box_result("Month over Month Growth",
                   f"purchases grew by {delta_purch:,.0f} versus {prev_month}. keep momentum with short demos and clear value bullets.",
                   "good")
    else:
        box_result("Purchases Decreased",
                   f"purchases fell by {abs(delta_purch):,.0f} versus {prev_month}. "
                   "check mix and reduce reliance on discounts. improve page clarity and verified reviews.",
                   "bad")

    mix = (curm.groupby("Category_std", as_index=False)["purchased_last_month"]
                .sum().sort_values("purchased_last_month", ascending=False).head(12))
    if guard(mix, "no category contribution to display."):
        fig2 = px.bar(mix, x="Category_std", y="purchased_last_month", text_auto=".2s",
                      title=f"Top 12 Categories in {month_focus}")
        fig2.update_layout(xaxis_title="Category", yaxis_title="Purchases")
        st.plotly_chart(fig2, use_container_width=True)
        fig_notes("find where purchases concentrate",
                  "bars rank categories by total purchases",
                  "shows leaders to protect and mid pack to develop",
                  "keep leaders stocked and featured, test bundles in middle group")
        top_cat = mix.iloc[0]
        box_result("Leader Category",
                   f"{top_cat['Category_std']} leads this month. protect with strong visuals and clear benefits.",
                   "good")

    g = (sales_f.groupby(["month", "Category_std"])["purchased_last_month"].sum().reset_index())
    if guard(g, "no data for month over month category change."):
        last_two = g[g["month"].isin([prev_month, month_focus])]
        if guard(last_two, "not enough months to compare."):
            pivot = last_two.pivot(index="Category_std", columns="month", values="purchased_last_month").fillna(0)
            if prev_month in pivot.columns and month_focus in pivot.columns:
                pivot["delta"] = pivot[month_focus] - pivot[prev_month]
                ranked = pivot.sort_values("delta", ascending=False).reset_index()
                fig3 = px.bar(ranked, x="Category_std", y="delta", text_auto=".2s",
                              title=f"Month over Month Change ({prev_month} -> {month_focus})")
                fig3.update_layout(xaxis_title="Category", yaxis_title="Delta Purchases")
                st.plotly_chart(fig3, use_container_width=True)
                fig_notes("spot what moved the most",
                          "bars show increases and decreases by category",
                          "explains the headline kpi move",
                          "double click the biggest moves in each team tab")
                winner = ranked.iloc[0]; loser = ranked.iloc[-1]
                box_info("Monthly Highlights",
                         f"📈 biggest gain: {winner['Category_std']} (+{winner['delta']:,.0f}). "
                         f"📉 biggest drop: {loser['Category_std']} ({abs(loser['delta']):,.0f} down).")

# ============================================================
# tab 3 - finance and ops
# ============================================================
with tabs[2]]:
    st.header(f"Finance and Ops - {month_focus}")
    st.caption("volume drivers, discount posture, and operational guardrails")

    if not guard(curm, "no rows for this month after filters."):
        st.stop()

    by_cat = (curm.groupby("Category_std", as_index=False)
                  .agg(purchases=("purchased_last_month", "sum"),
                       avg_disc=("discount_percentage", "mean")))
    if guard(by_cat, "no category statistics available."):
        fig1 = px.scatter(by_cat, x="avg_disc", y="purchases", color="Category_std",
                          title="Purchases vs Average Discount by Category")
        fig1.update_layout(xaxis_title="Average Discount %", yaxis_title="Purchases")
        st.plotly_chart(fig1, use_container_width=True)
        fig_notes("check if volume depends on deep discounts",
                  "each dot is a category with average discount and purchases",
                  "if high volume sits at high discount, margin is at risk",
                  "move some gains to bundles and sharper product value copy")
        lead = by_cat.sort_values("purchases", ascending=False).head(1)
        if not lead.empty:
            r = lead.iloc[0]
            level = "good" if r["avg_disc"] < 25 else "ok"
            box_result("Volume Driver",
                       f"{r['Category_std']} drives the most purchases with an average discount of {r['avg_disc']:.1f}%.",
                       level)

    if "discount_percentage" in curm.columns and curm["discount_percentage"].notna().any():
        fig2 = px.box(curm.dropna(subset=["discount_percentage"]),
                      x="Category_std", y="discount_percentage", color="Category_std",
                      title="Discount % Distribution by Category")
        fig2.update_layout(xaxis_title="", yaxis_title="Discount %")
        st.plotly_chart(fig2, use_container_width=True)
        fig_notes("see how discounts are spread within each category",
                  "boxes show the range and median discount",
                  "heavy tails or high medians hint at promo dependence",
                  "tighten discount bands and fix clarity before promos")
        tone = "bad" if (pd.notna(cur["disc"]) and cur["disc"] >= 30) else "ok"
        box_result("Discount Posture",
                   f"average discount is {cur['disc']:.1f}%." if pd.notna(cur["disc"]) else "discount level is not available.",
                   tone)
        if tone == "bad":
            box_warn("Caution on Discount Depth",
                     "deep average discounts can hide product clarity problems. fix value messaging first.")

    ops = [
        "✅ check stock and shipping for the top movers",
        "✅ review return reasons where discounts rose but volume did not",
        "✅ confirm delivery promise on the product page is accurate",
    ]
    box_info("Operations Checklist", " • " + "\n • ".join(ops))

# ============================================================
# tab 4 - marketing and content strategy
# ============================================================
with tabs[3]]:
    st.header(f"Marketing and Content Strategy - {month_focus}")
    st.caption("turn ratings and reviews into a content plan a non technical teammate can use")

    if not guard(curm, "no rows for this month after filters."):
        st.stop()

    box_info("Focus For This Month",
             "find categories that convince without heavy discounts, learn common questions from reviews, and publish a simple 7 day plan")

    plot_df = curm.dropna(subset=["product_rating"])
    if guard(plot_df, "no rating values to visualize."):
        sizes = safe_bubble_sizes(plot_df.get("purchased_last_month"))
        if sizes is None:
            figm1 = px.scatter(plot_df, x="product_rating", y="total_reviews",
                               color="Category_std", opacity=0.9,
                               title="Engagement Map - Rating vs Reviews")
        else:
            figm1 = px.scatter(plot_df, x="product_rating", y="total_reviews",
                               color="Category_std", size=sizes, opacity=0.9,
                               title="Engagement Map - Rating vs Reviews (bubble = purchases)")
        figm1.update_layout(xaxis_title="Average Rating (★)", yaxis_title="Review Volume")
        st.plotly_chart(figm1, use_container_width=True)
        fig_notes("find products people like and talk about",
                  "dots show rating on x and review volume on y",
                  "high rating with many reviews is a signal to scale with social proof",
                  "add short creator videos and pin top reviews on the product page")
        top = (plot_df.assign(score=lambda d: d["product_rating"].rank(pct=True)
                                           + safe_num(d["total_reviews"]).rank(pct=True))
                      .sort_values("score", ascending=False).head(1))
        if not top.empty:
            box_result("Advocacy Engine",
                       "products with high ratings and many reviews compound with social proof. keep feeding those with short reels.",
                       "good")

    conv = (curm.groupby("Category_std", as_index=False)
                 .agg(purchases=("purchased_last_month", "sum"),
                      reviews=("total_reviews", "sum")))
    if guard(conv, "no review totals available for conversion proxy."):
        conv["purch_per_review"] = conv.apply(lambda r: r["purchases"] / r["reviews"] if r["reviews"] not in [0, np.nan] else np.nan, axis=1)
        conv = conv.dropna(subset=["purch_per_review"]).sort_values("purch_per_review", ascending=False).head(10)
        if guard(conv, "no conversion proxy to show."):
            figc = px.bar(conv, x="Category_std", y="purch_per_review", text_auto=".2f",
                          title="Purchases Per Review (conversion proxy)")
            figc.update_layout(xaxis_title="Category", yaxis_title="Purchases / Review")
            st.plotly_chart(figc, use_container_width=True)
            fig_notes("approximate how efficiently reviews convert to purchases",
                      "bars show purchases per review by category",
                      "high values are strong bets for paid and organic placement",
                      "feature those categories across paid and owned channels")
            lead = conv.iloc[0]
            box_result("High Intent Signal",
                       f"{lead['Category_std']} converts reviews to purchases efficiently.",
                       "good")

    if reviews is not None and "review_text" in reviews.columns:
        rcur = reviews.copy()
        if "month" in rcur.columns:
            rcur = rcur[rcur["month"] == month_focus]
        if "Category_std" in rcur.columns and cat_sel:
            rcur = rcur[rcur["Category_std"].isin(cat_sel)]
        if guard(rcur, "no reviews for this slice."):
            low = rcur["review_text"].astype(str).str.lower()
            words = pd.Series(" ".join(low.tolist()))
            words = words.str.replace(r"[^a-z\s]", "", regex=True).str.split()
            words = pd.Series([w for lst in words.dropna().tolist() for w in lst])
            stop = set("the and to a of is it for in on my this that very too but not as be are you i so at have its they we just would".split())
            topw = words[~words.isin(stop)].value_counts().head(12).reset_index()
            topw.columns = ["word", "count"]
            if guard(topw, "no frequent words to display."):
                figw = px.bar(topw, x="word", y="count", text_auto=".2s", title="Common Customer Words (this month)")
                st.plotly_chart(figw, use_container_width=True)
                fig_notes("turn common questions into content",
                          "bars show frequent words in reviews",
                          "these are plain language topics customers care about",
                          "record short clips that answer these directly")
                box_warn("Caution", "do not copy words blindly. skim a few reviews to confirm meaning before posting.")

    # simple seven day calendar
    st.markdown("### Seven Day Content Plan")
    top_cats = (curm.groupby("Category_std")["purchased_last_month"]
                .sum().sort_values(ascending=False).head(3).index.tolist())
    if not top_cats:
        top_cats = ["Top Category"]
    cal = [
        f"day 1 - creator short 15 to 30s: why {top_cats[0]} solves a real problem. show it in use",
        f"day 2 - product page demo: unbox {top_cats[0]} and cover top three benefits in under 30 seconds",
        f"day 3 - social proof reel: put the best review on screen over product footage",
        f"day 4 - quick fix: answer the most common question from reviews in one short clip",
        f"day 5 - compare: {top_cats[0]} vs {top_cats[1] if len(top_cats)>1 else 'another option'} and who each is for",
        f"day 6 - daily use montage: three quick shots that show real life use cases",
        f"day 7 - community q and a: answer two real customer questions in 30 seconds each",
    ]
    box_info("Publishing Guide", "post on tiktok and ig reels. pin the best demo to the product page hero video.")
    st.markdown("\n".join([f"- {row}" for row in cal]))

# ============================================================
# tab 5 - product and pricing
# ============================================================
with tabs[4]]:
    st.header(f"Product and Pricing - {month_focus}")
    st.caption("quality signals and how prices step down from list")

    if not guard(curm, "no rows for this month after filters."):
        st.stop()

    if "product_rating" in curm.columns and curm["product_rating"].notna().any():
        figp1 = px.box(curm.dropna(subset=["product_rating"]),
                       x="Category_std", y="product_rating", color="Category_std",
                       title="Product Ratings by Category")
        figp1.update_layout(xaxis_title="", yaxis_title="Rating (★)")
        st.plotly_chart(figp1, use_container_width=True)
        fig_notes("assess perceived quality by category",
                  "boxes show distribution and median rating",
                  "low medians signal friction or wrong expectations",
                  "add short demos and clarify sizing and care details")
        worst = curm.groupby("Category_std")["product_rating"].mean().sort_values().head(1)
        if not worst.empty:
            wcat, wr = worst.index[0], worst.values[0]
            level = "bad" if wr < 4.0 else "ok"
            box_result("Quality Signal", f"{wcat} averages {wr:.2f} stars. add demos and clear benefit bullets.", level)

    if {"original_price", "discounted_price"}.issubset(curm.columns):
        ladder = (curm[["Category_std", "original_price", "discounted_price"]]
                  .dropna().groupby("Category_std").median().reset_index())
        if guard(ladder, "no price ladder to show."):
            figp2 = px.scatter(ladder, x="original_price", y="discounted_price", color="Category_std",
                               title="Price Ladder (median original vs discounted)")
            figp2.update_layout(xaxis_title="Original Price (median)", yaxis_title="Discounted Price (median)")
            st.plotly_chart(figp2, use_container_width=True)
            fig_notes("see how far discounting steps prices down from list",
                      "each dot is a category with median original and discounted price",
                      "large gaps suggest reliance on promos",
                      "test lighter promos paired with clearer value bullets")
            heavy = ladder[ladder["discounted_price"] <= 0.7 * ladder["original_price"]]
            tone = "ok" if heavy.empty else "bad"
            msg = "median discounted prices sit near list. this looks healthy." if heavy.empty else \
                  "some categories lean on deep discounts. test lighter promos with sharper value messages."
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
        box_result("What To Fix First",
                   "start with low rating, high volume items. clarify expectations and show the product in use.",
                   "bad")
        box_warn("Watch Returns and Support Load",
                 "low rating, high volume items often create tickets and returns. fix these first to protect margin.")

# ============================================================
# tab 6 - plans and accountability
# ============================================================
with tabs[5]]:
    st.header("Plans and Accountability")
    st.caption("concrete items with owners and due dates that match this month story")

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
                                action=f"{cat} moved {direction} by {abs(d):,.0f} month over month. set a target and an owner.",
                                priority="High", due=(date.today() + timedelta(days=7)).isoformat(), status="Open"))

    if cur["reviews"] and pre["reviews"] and cur["reviews"] < pre["reviews"]:
        actions.append(dict(team="Marketing and Content",
                            action="review volume slipped. refresh creatives, seed ugc, and add social proof blocks on the product page.",
                            priority="High", due=(date.today() + timedelta(days=10)).isoformat(), status="Open"))

    if pd.notna(cur["disc"]) and cur["disc"] >= 30:
        actions.append(dict(team="Finance and Ops",
                            action=f"average discount {cur['disc']:.1f}%. test lighter discounts with bundles and sharper value bullets.",
                            priority="Medium", due=(date.today() + timedelta(days=14)).isoformat(), status="Open"))

    lowcat = curm.groupby("Category_std")["product_rating"].mean().sort_values().head(1)
    if not lowcat.empty and lowcat.values[0] < 4.0:
        actions.append(dict(team="Product and Pricing",
                            action=f"{lowcat.index[0]} averages {lowcat.values[0]:.2f} stars. add 20 to 30 second demos and clarify sizing and care.",
                            priority="Medium", due=(date.today() + timedelta(days=14)).isoformat(), status="Open"))

    actions.append(dict(team="Cross Team",
                        action="record a 10 minute loom walkthrough linking to this hub and the key wins and issues.",
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
        box_info("How To Use This List", "share the csv in your team channel and assign owners right away.")
    else:
        box_warn("No Actions Generated", "try a different month or widen categories to reveal more signals.")

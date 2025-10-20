# 📦 Amazon Company Hub — Two-Month Insights (CMSE 830 Midterm)

**Goal:**  
Provide a clear, interactive Streamlit dashboard that helps non-technical teams understand **what happened, why it happened, what it means, and what to do next** — with structured charts, captions, and exportable insights.

---

## 🧭 Quick Start

To run locally:

```
pip install -r requirements.txt
streamlit run final_app.py
```

---

## 📂 Files Expected (Place beside `final_app.py`)

| File Name | Required | Description |
|------------|-----------|-------------|
| amazon_sales_with_two_months.csv | ✅ Yes | Core dataset — purchases, discounts, prices, ratings, reviews, category, and month |
| amazon_reviews.csv | ⚙️ Optional | Review text, sentiment, and keywords for content planning |
| amazon_products.csv | ⚙️ Optional | Extra product metadata such as names, prices, and categories |

---

## 📊 App Structure

| Tab | Description |
|------|-------------|
| **1. How To Use** | Full project intro, dataset overview, why this project matters, and instructions for interpreting the dashboard |
| **2. Executive Snapshot** | KPIs, monthly trend, category mix, highlights, and next month’s improvement plan |
| **3. Finance & Ops** | Discount posture, volume vs discount, and operational checklist |
| **4. Marketing & Content Strategy** | Engagement map, conversion proxy, keyword analysis, platform mix, and 7-day content calendar |
| **5. Product & Pricing** | Ratings distribution, pricing posture, and underperforming items |
| **6. Plans & Accountability** | Auto-generated team actions with owners, priorities, and due dates |

---

## 🧹 Data Preparation

- Standardized all date fields into monthly format  
- Converted numeric columns (purchases, discounts, prices, ratings)  
- Normalized product categories for consistency  
- Created flags for promotional items (best seller, coupon, sponsored)  
- Combined data sources for two months to enable trend comparisons  

---

## 🧠 Key Design Choices

- Focused on **clarity** and **explainability** over complex modeling  
- Each chart includes **gray subtext** summarizing:  
  - **Why:** the motivation for this visualization  
  - **What:** what the data shows  
  - **So What:** interpretation of the result  
  - **What Next:** actionable follow-up steps  
- Added traffic-light color logic (green/yellow/red) for fast scanning by executives  
- Integrated sentiment from text reviews to strengthen marketing insights  
- Designed all visuals to be readable without technical training  

---

## 🚀 Deployment Notes

- App tested with Streamlit 1.38+  
- Place all datasets beside the main Python file  
- Avoid nested folders — Streamlit expects direct relative paths  
- Hosted version can be deployed on **Streamlit Community Cloud** or **GitHub Pages (via Streamlit link)**  

---

## 🏁 Summary

This project demonstrates end-to-end data analysis for Amazon product sales across two months.  
It integrates **finance, marketing, and product analytics** to provide a full company snapshot.  
The app promotes collaboration between teams through shared, easy-to-read insights rather than technical code or reports.  

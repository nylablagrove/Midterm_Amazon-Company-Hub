# 🧩 WHY REPORT — Amazon Company Hub (CMSE 830 Midterm)

## 💡 Why I Chose This Project

I wanted to create something that shows how data science can support **decision-making across teams**, not just visualize numbers.  
The idea for the *Amazon Company Hub* came from real-world corporate dashboards used by marketing, finance, and operations teams.  
Instead of building a traditional model, I wanted to design a **shared hub** where every department could explore data and understand what to do next without coding.  
This aligns with the goal of turning raw data into clear, business-ready insights.

---

## 🧭 Project Purpose

The dashboard answers four main questions:
1. **What happened?** — It shows how purchases, discounts, and ratings changed across months.  
2. **Why did it happen?** — It breaks down category trends, reviews, and discount behavior.  
3. **So what does it mean?** — Each figure includes context and interpretation boxes so anyone can read it easily.  
4. **What’s next?** — Every tab includes action plans and next-step ideas automatically generated from the data.

The goal is to **help executives and non-technical teams** quickly see what’s going well, what’s struggling, and how they can fix it.

---

## 🧮 Data Collection & Preparation

### Data Sources
I used three datasets focused on Amazon sales and reviews:
| File | Description |
|------|--------------|
| `amazon_sales_with_two_months.csv` | Core dataset showing purchases, discounts, and prices for two months |
| `amazon_reviews.csv` | Review text and sentiment used for keyword and tone analysis |
| `amazon_products.csv` | Product titles, categories, and pricing information |

### Cleaning & Integration
- Standardized all **date fields** into month-based format  
- Converted numeric columns like purchases, discounts, and ratings  
- Removed duplicates and handled missing data with `pandas.to_numeric(errors="coerce")`  
- Created new columns for **promo flags** (`best_seller`, `coupon`, `sponsored`)  
- Integrated the three datasets by category for comparison across months  

This step was critical because the datasets had inconsistent encodings and text formatting that caused parsing issues.

---

## 📊 Exploratory Data Analysis & Visualization

I focused on **interpretable visualizations** that answer practical questions rather than abstract ones.  
The app includes:
- Line charts for **month-over-month purchase trends**  
- Bar charts for **category mix and performance changes**  
- Scatter plots for **volume vs. discount** and **rating vs. engagement**  
- Box plots for **price and rating distributions**  
- Tables for **underperforming items** and **team accountability**

Each figure is followed by a concise explanation (in gray text) describing **why it’s relevant**, **what it shows**, and **what action to take next**.

---

## 🧠 Thinking Process & Avoiding Mistakes

- I focused on **designing first**, not coding immediately.  
  I sketched how an executive would use this tool and which insights matter most.  
- A common mistake in dashboards is **showing data with no interpretation**.  
  To avoid that, I built every figure with a paired explanation or “result box.”  
- I also avoided overcomplication — no predictive modeling, since that wasn’t the goal.  
  The goal was communication and insight, not algorithmic performance.

---

## 🧩 Data Processing

- Implemented a simple imputation method: replaced missing numeric values with `NaN`-safe conversions and median where appropriate.  
- Flagged potential errors in the dataset with **caution boxes** (⚠️) to guide interpretation.  
- Added a secondary data validation layer so empty filters or missing months don’t crash the app.  

---

## 🧱 Streamlit App Development

The final dashboard includes:
- **Six main tabs** representing teams and company roles  
- Interactive **month and category filters**  
- Built-in **download button** for the action plan CSV  
- Text-based “Why, What, So What, What Next” boxes under each visualization  
- Traffic-light color coding for **quick visual interpretation**

The app is completely **self-explanatory** — someone with no coding experience can open it and understand the company’s performance story.

---

## ⚙️ Deployment & GitHub

- Code stored in `final_app.py`  
- Deployed using **Streamlit Cloud**  
- Repository includes:
  - `final_app.py`
  - `README.md`
  - `WHY_REPORT.md`
  - CSV data files  
- Added detailed comments and documentation throughout the script so anyone can modify or learn from it.

---

## 🎯 Reflection

This project helped me understand how to turn raw data into a meaningful business narrative.  
Instead of relying on complex models, I learned that **clarity, context, and accessibility** matter just as much.  
If I continued this project, I’d integrate **real-time data streaming** and possibly a **predictive sales component** to forecast next month’s results.

---

## 🏁 Summary

- Combines sales, marketing, and product analytics into one interactive dashboard  
- Designed for executives, not coders  
- Includes explanations, color-coded results, and team-specific action items  
- Fully deployable and readable by anyone, not just data scientists  

This project demonstrates leadership in managing data science from **collection to insight delivery** — bridging the gap between analytics and decision-making.

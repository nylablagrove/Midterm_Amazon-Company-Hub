# WHY REPORT — Amazon Company Hub (CMSE 830 Midterm)

## Why I Chose This Project

I wanted to create a project that shows how data science supports company decision-making across teams.  
The idea for the Amazon Company Hub came from real dashboards used by marketing, finance, and operations teams.  
Rather than focusing on predictive modeling, I wanted to design a shared hub where anyone could explore data and understand what to do next.  
This approach connects data science with communication and teamwork.

---

## Project Purpose

The dashboard answers four main questions:
1. **What happened?** It shows how purchases, discounts, and ratings changed across months.  
2. **Why did it happen?** It breaks down category trends, reviews, and discount behavior.  
3. **What does it mean?** Each figure includes a short explanation to help interpret the results.  
4. **What’s next?** Each tab includes action plans and improvement ideas created from the data.

The goal is to help executives and teams understand performance quickly without needing to read code.

---

## Data Collection and Preparation

### Data Sources
| File | Description |
|------|--------------|
| `amazon_sales_with_two_months.csv` | Core dataset showing purchases, discounts, and prices for two months |
| `amazon_reviews.csv` | Review text and sentiment used for keyword and tone analysis |
| `amazon_products.csv` | Product titles, categories, and pricing information |

### Cleaning and Integration
- Standardized all date fields into a monthly format  
- Converted numeric columns such as purchases, discounts, and ratings  
- Removed duplicates and handled missing data with simple numeric conversion  
- Created flags for promotional items like best seller, coupon, and sponsored  
- Integrated datasets by category to compare across months  

This cleaning was essential because of inconsistent encodings and formatting.

---

## Exploratory Data Analysis and Visualization

I created visualizations that answer direct questions about company performance.  
The app includes:
- Line charts for monthly purchase trends  
- Bar charts for category performance  
- Scatter plots for volume versus discount and rating versus engagement  
- Box plots for price and rating distributions  
- Tables for underperforming products and accountability plans  

Each figure has a brief gray caption that explains why it matters and what the next action should be.

---

## Thinking Process and Avoiding Mistakes

- I began with a design plan instead of coding first.  
- I focused on what an executive would want to see rather than what I wanted to analyze.  
- A common dashboard mistake is showing data without explanation. I avoided this by adding a clear “result box” under each chart.  
- I did not include complex algorithms since my goal was clarity, not prediction.

---

## Data Processing

- Used median and safe numeric conversion to handle missing data  
- Added validation checks for empty filters and missing months  
- Created caution boxes to highlight potential data reliability issues  

---

## Streamlit App Development

The final app includes:
- Six main tabs for different teams  
- Interactive filters for month and category  
- Downloadable CSV for the action plan  
- “Why, What, So What, What Next” explanations under each visualization  
- Color-coded result boxes for fast interpretation  

The app is designed for non-technical users and can be understood without data science experience.

---

## Deployment and GitHub

- Code is stored in `final_app.py`  
- Deployed on Streamlit Cloud  
- Repository includes:  
  - `final_app.py`  
  - `README.md`  
  - `WHY_REPORT.md`  
  - CSV datasets  
- Added detailed comments throughout the code so others can follow the logic easily.

---

## Reflection

This project taught me how to translate raw data into a meaningful story.  
It emphasized that clear communication and usability are just as important as technical accuracy.  
If I continued this project, I would add real-time data updates and possibly a simple forecasting model for next-month projections.

---

## Summary

- Combines sales, marketing, and product analytics into one interactive dashboard  
- Written and designed for executives and decision makers  
- Includes contextual explanations, color-coded results, and team actions  
- Demonstrates how to lead a complete data science project from collection to communication


#Hotel Booking Demand & Customer Segmentation

## Project Decription
This project analyzes booking data from a Resort Hotel with the goal of understanding customer booking and cancellation behavior. K-Means Clustering was used to group customers and extract business insights.

## Tools used
- **Data Manipulation:** Python, pandas, numpy
- **Machine Learning:** K-Means, StandardScaler, Elbow Method
- **Data Visualization:** matplotlib, seaborn

##  Business Insights
Customers were divided into 3 categories. Further analysis of cancellation time revealed critical patterns:
1. **Early Birds:** They cancel frequently (38.9%), but do so on average **~120 days before arrival**. The hotel has plenty of resale time *(Suggestion: Flexible free cancellation policy)*.
2. **Premium Guests (High Spenders):** They cancel on average 43 days in advance. They are the group that brings in the highest revenue per night (ADR).
3. **Last-Minute /Low cost:** They have the lowest cancellation rate, but when they do cancel they do so a few days before arrival on average, *(Suggestion: Strict/Non-Refundable cancellation policy)*.

## How to run the code
1. Clone the repository.
2. Install the libraries: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run the Python file to see the graphs and results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# step 1: Loading & Cleaning Data
# ==========================================
print("--- 1. Loading & cleaning Data ---")
df = pd.read_csv("H1.csv")

# 1. Replace the empty cells
df['Country'] = df['Country'].fillna('Unknown')

# 2. Delete negative prices
df = df[df['ADR'] >= 0]

# 3. Create new variable
df['TotalNights'] = df['StaysInWeekendNights'] + df['StaysInWeekNights']

# 4. Delete reservations with 0 overnight stays
df = df[df['TotalNights'] > 0]

# Delete empty columns
df = df.dropna(subset=['LeadTime', 'ADR'])

print(f"The final clear dataset has {df.shape[0]} reservations.\n")


# ==========================================
# Step 2: Preparing for CLUSTERING (SCALING)
# ==========================================
print("--- 2. Data preparing ---")
# Select the characteristics: How early did the reservation (LeadTime) & Price (ADR)
features = ['LeadTime', 'ADR']
X = df[features].copy()

#Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==========================================
# Step 3: Finding Optimum k (ELBOW METHOD)
# ==========================================
print("--- 3. Elbow Method ---")
wcss = []
max_k = 10

# Check the script for 1 to 10 clusters
for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Create Chart
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--', color='#1f77b4', linewidth=2, markersize=8)
plt.title('Elbow Method to find the optimum k', fontsize=14, pad=15)
plt.xlabel('Number of clusters (k)', fontsize=12)
plt.ylabel('WCSS (Error)', fontsize=12)
plt.xticks(range(1, max_k + 1))
plt.grid(True, linestyle=':', alpha=0.7)
#plt.annotate(' (Optimum k=3)', xy=(3, 24609), xytext=(4, 40000),
             #arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             #fontsize=12, color='red')
plt.tight_layout()
plt.show()


# ==========================================
# Step 4: K-MEANS with K=3
# ==========================================
print("--- 4. K-Means Clustering (k=3) ---")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Guests clusters names
cluster_names = {
    0: 'Early Birds',
    1: 'Last-Minute / Low cost',
    2: 'Premium Guests (High Spenders)'
}
df['Customer_Profile'] = df['Cluster'].map(cluster_names)

# Statistics sumary / cluster
summary = df.groupby('Customer_Profile')[['LeadTime', 'ADR']].mean().round(2)
summary['Number of guests'] = df.groupby('Customer_Profile').size()
print(summary.to_string())


# ==========================================
# Step 5: Visualization of clustering results
# ==========================================
print("\n--- 5. Clusterings Plot ---")
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df,
    x='LeadTime',
    y='ADR',
    hue='Customer_Profile',
    palette=['#1f77b4', '#ff7f0e', '#2ca02c'],
    alpha=0.6,
    s=30
)

plt.title('Customer Segmentation', fontsize=14, pad=15)
plt.xlabel('Waiting days from booking to arrival (Lead Time)', fontsize=12)
plt.ylabel('Average Daily Price (ADR σε €)', fontsize=12)
plt.legend(title='Guest Profile')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==========================================
# Step 6: CANCELLATIONS ANALYSIS BY PROFILE (BUSINESS INSIGHT)
# ==========================================
print("\n--- 6. Calculation of Cancellation Rates ---")

# We calculate the column average IsCanceled (has 0 and 1)
# and we multiply wit 100 (%)
cancel_rates = df.groupby('Customer_Profile')['IsCanceled'].mean() * 100

# We sort the results for better visual presentation
cancel_rates = cancel_rates.sort_values(ascending=False)
print(cancel_rates.round(1).astype(str) + '%')


# Plot creation
plt.figure(figsize=(10, 6))

# Create the Barplot
ax = sns.barplot(
    x=cancel_rates.index,
    y=cancel_rates.values,
    palette=['#d62728', '#ff7f0e', '#2ca02c']
)

# Add titles and format
plt.title('Cancellation Rate by Customer Profile', fontsize=14, pad=15, fontweight='bold')
plt.ylabel('Cancellation Rate (%)', fontsize=12)
plt.xlabel('Customer Profile ( K-Means Clustering)', fontsize=12)
plt.ylim(0, 50)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Adding the exact percentage above each bar (Data Labels)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 10), textcoords='offset points',
                fontsize=12, fontweight='bold', color='black')

plt.tight_layout()
plt.show()

# We only keep cancelled reservations.
canceled_df = df[df['IsCanceled'] == 1].copy()

# We create the arrival date (Arrival Date) by combining Year, Month, Day
canceled_df['ArrivalDate'] = pd.to_datetime(
    canceled_df['ArrivalDateYear'].astype(str) + '-' +
    canceled_df['ArrivalDateMonth'] + '-' +
    canceled_df['ArrivalDateDayOfMonth'].astype(str),
    format='%Y-%B-%d'
)

#Convert cancellation date to date format
canceled_df['ReservationStatusDate'] = pd.to_datetime(canceled_df['ReservationStatusDate'])

#Calculation of difference (How many days before arrival was the cancellation made?)
canceled_df['DaysBeforeArrival_Canceled'] = (canceled_df['ArrivalDate'] - canceled_df['ReservationStatusDate']).dt.days

# ==========================================
# Step 8: Cansellation time per cluster (BOXPLOT)
# ==========================================
print("\n--- 8. Cancellation Time per Customer Profile ---")

# Boxplot creation
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=canceled_df,
    x='Customer_Profile',
    y='DaysBeforeArrival_Canceled',
    palette=['#1f77b4', '#ff7f0e', '#2ca02c'],
    showfliers=False # We hide extreme outliers
)

# Chart Formatting
plt.title('How early does each customer category cancel?', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Customer Profile', fontsize=12)
plt.ylabel('Days before arrival when cancellation occurred', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a horizontal "Danger" line (14 days)
plt.axhline(y=14, color='red', linestyle='--', linewidth=2, label='Danger Zone (14 days)')
plt.legend()

plt.tight_layout()
plt.show()

# Print the results
print("Average cancellation time (in days) per group:")
print(canceled_df.groupby('Customer_Profile')['DaysBeforeArrival_Canceled'].median().round(0))
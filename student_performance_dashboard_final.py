import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import zipfile
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# -------------------------------------------------------------------
# STATISTICAL SAFETY HELPERS (Transparency for the end-user. Understand how reliable data is. Include ⚠️ as a warning symbol)
# -------------------------------------------------------------------

MIN_RELIABLE_N = 20

def is_reliable(df, min_n=MIN_RELIABLE_N):
    """Return True if sample size is large enough for reliable inference."""
    return len(df) >= min_n

def safe_mean(series):
    """Return mean only if sample size is reliable; otherwise return None."""
    return series.mean() if len(series) >= MIN_RELIABLE_N else None

def safe_corr(x, y):
    """Return correlation only if sample size is reliable; otherwise None."""
    if len(x) < MIN_RELIABLE_N:
        return None
    return x.corr(y)

def safe_rate(successes, total):
    """Return smoothed rate for small samples, raw rate for reliable samples."""
    if total == 0:
        return None
    if total < MIN_RELIABLE_N:
        # Laplace smoothing
        return (successes + 1) / (total + 2)
    return successes / total

def reliability_caption(df):
    """Return a caption explaining reliability level."""
    n = len(df)
    if n == 0:
        return "⚠️ No data available for this selection."
    if n < MIN_RELIABLE_N:
        return f"⚠️ Based on a small sample (n={n}). Interpret with caution."
    return f"Sample size: n={n} (statistically reliable)."
# -------------------------------------------------------------------
# STRICT STATISTICAL DESCRIPTORS (Trial - use to aid data description)
# -------------------------------------------------------------------

MIN_RELIABLE_N = 20

def describe_bar_chart(data, x_col, y_col, title_prefix=""):
    """Describe mean differences between groups with strict reliability rules."""
    if len(data) < MIN_RELIABLE_N:
        return f"**{title_prefix}**\nNot enough data for reliable comparison (n={len(data)})."

    grouped = data.groupby(x_col)[y_col].agg(['mean', 'count']).reset_index()

    # To keep only groups with reliable sample sizes
    reliable_groups = grouped[grouped['count'] >= MIN_RELIABLE_N]

    if reliable_groups.empty:
        return f"**{title_prefix}**\nNo groups have enough data for reliable comparison."

    max_row = reliable_groups.loc[reliable_groups['mean'].idxmax()]
    min_row = reliable_groups.loc[reliable_groups['mean'].idxmin()]
    diff = max_row['mean'] - min_row['mean']

    desc = f"**{title_prefix}**\n"
    desc += f"Among groups with reliable sample sizes (n ≥ {MIN_RELIABLE_N}), "
    desc += f"**{max_row[x_col]}** has the highest average score ({max_row['mean']:.1f}), "
    desc += f"while **{min_row[x_col]}** has the lowest ({min_row['mean']:.1f}). "
    desc += f"The difference between these reliable groups is **{diff:.1f} points**."

    return desc


def describe_scatter(data, x_col, y_col, title_prefix=""):
    """Describe correlation only when sample size is reliable."""
    if len(data) < MIN_RELIABLE_N:
        return f"**{title_prefix}**\nNot enough data to compute a reliable correlation (n={len(data)})."

    corr = data[x_col].corr(data[y_col])

    desc = f"**{title_prefix}**\n"
    desc += f"The correlation between **{x_col}** and **{y_col}** is **{corr:.2f}**. "

    # Test to see if this interprets correlation strength
    if abs(corr) < 0.1:
        desc += "This suggests little to no linear relationship."
    elif abs(corr) < 0.3:
        desc += "This suggests a weak relationship."
    elif abs(corr) < 0.5:
        desc += "This suggests a moderate relationship."
    else:
        desc += "This suggests a strong relationship."

    # Another attempt to trial direction
    if corr > 0:
        desc += f" As **{x_col}** increases, **{y_col}** tends to increase."
    elif corr < 0:
        desc += f" As **{x_col}** increases, **{y_col}** tends to decrease."

    return desc


def describe_histogram(data, col, title_prefix=""):
    """Describe distribution only when sample size is reliable."""
    if len(data) < MIN_RELIABLE_N:
        return f"**{title_prefix}**\nNot enough data to describe the distribution reliably (n={len(data)})."

    mean_val = data[col].mean()
    median_val = data[col].median()
    std_val = data[col].std()

    desc = f"**{title_prefix}**\n"
    desc += f"The average **{col}** is **{mean_val:.1f}**, with a median of **{median_val:.1f}**. "
    desc += f"Most values fall within about **{std_val:.1f}** points of the average. "

    if abs(mean_val - median_val) > 5:
        desc += "The distribution appears somewhat skewed."
    else:
        desc += "The distribution appears fairly symmetric."

    return desc


def describe_heatmap(corr_matrix, title_prefix=""):
    """Describe strongest correlations only when matrix is reliable."""
    desc = f"**{title_prefix}**\n"

    corr_vals = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).stack()

    if corr_vals.empty:
        return desc + "No correlations available."

    max_corr = corr_vals.idxmax()
    min_corr = corr_vals.idxmin()

    desc += f"The strongest positive correlation is between **{max_corr[0]}** and **{max_corr[1]}** "
    desc += f"({corr_vals[max_corr]:.2f}). "

    desc += f"The strongest negative correlation is between **{min_corr[0]}** and **{min_corr[1]}** "
    desc += f"({corr_vals[min_corr]:.2f})."

    return desc


def describe_line_chart(data, x_col, y_col, title_prefix=""):
    """Describe trend only when sample size is reliable."""
    if len(data) < MIN_RELIABLE_N:
        return f"**{title_prefix}**\nNot enough data to describe a reliable trend (n={len(data)})."

    y_vals = data[y_col].values

    if y_vals[-1] > y_vals[0]:
        trend = "increasing"
    elif y_vals[-1] < y_vals[0]:
        trend = "decreasing"
    else:
        trend = "stable"

    desc = f"**{title_prefix}**\n"
    desc += f"As **{x_col}** increases, **{y_col}** shows an **{trend}** trend. "
    desc += f"Values range from **{y_vals.min():.1f}** to **{y_vals.max():.1f}**."

    return desc

# ------------------------------------------------------------------------------
# DESCRIPTION HELPER FUNCTIONS (My visualisation description function - satisfies new user need)
# ------------------------------------------------------------------------------
def describe_bar_chart(data, x_col, y_col, title_prefix=""):
    """Generate a description for a bar chart showing mean of y_col per x_col."""
    if data.empty:
        return "No data available for this chart."
    grouped = data.groupby(x_col)[y_col].agg(['mean', 'count']).reset_index()
    if len(grouped) == 0:
        return "No groups to display."
    max_row = grouped.loc[grouped['mean'].idxmax()]
    min_row = grouped.loc[grouped['mean'].idxmin()]
    diff = max_row['mean'] - min_row['mean']
    desc = f"**{title_prefix}**  \n"
    desc += f"On average, students in **{max_row[x_col]}** score highest ({max_row['mean']:.1f}), "
    desc += f"while those in **{min_row[x_col]}** score lowest ({min_row['mean']:.1f}). "
    desc += f"The gap between the highest and lowest group is **{diff:.1f} points**."
    if diff > 10:
        desc += " This is a substantial difference."
    desc += f"  \n*Sample sizes*: "
    desc += ", ".join([f"{row[x_col]} (n={row['count']})" for _, row in grouped.iterrows()])
    return desc

def describe_scatter(data, x_col, y_col, title_prefix=""):
    """Generate a description for a scatter plot with correlation."""
    if len(data) < 2:
        return "Not enough data points to analyze correlation."
    corr = data[x_col].corr(data[y_col])
    desc = f"**{title_prefix}**  \n"
    desc += f"The correlation between **{x_col}** and **{y_col}** is **{corr:.2f}**. "
    if abs(corr) < 0.1:
        desc += "There is little to no linear relationship."
    elif abs(corr) < 0.3:
        desc += "This indicates a weak relationship."
    elif abs(corr) < 0.5:
        desc += "This indicates a moderate relationship."
    else:
        desc += "This indicates a strong relationship."
    if corr > 0:
        desc += f" As {x_col} increases, {y_col} tends to increase."
    elif corr < 0:
        desc += f" As {x_col} increases, {y_col} tends to decrease."
    return desc

def describe_histogram(data, col, title_prefix=""):
    """Describe a histogram of a single column."""
    if data.empty:
        return "No data available."
    mean_val = data[col].mean()
    median_val = data[col].median()
    std_val = data[col].std()
    desc = f"**{title_prefix}**  \n"
    desc += f"The average **{col}** is **{mean_val:.1f}**, and the median is **{median_val:.1f}**. "
    desc += f"Most scores fall within about **{std_val:.1f}** points of the average. "
    if abs(mean_val - median_val) > 5:
        desc += "The distribution is somewhat skewed."
    else:
        desc += "The distribution is fairly symmetric."
    return desc

def describe_heatmap(corr_matrix, title_prefix=""):
    """Describe key correlations from a correlation matrix."""
    
    corr_vals = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).stack()
    if corr_vals.empty:
        return "No correlations to describe."
    max_corr = corr_vals.idxmax()
    min_corr = corr_vals.idxmin()
    desc = f"**{title_prefix}**  \n"
    desc += f"The strongest positive correlation is between **{max_corr[0]}** and **{max_corr[1]}** ({corr_vals[max_corr]:.2f}). "
    desc += f"The strongest negative correlation is between **{min_corr[0]}** and **{min_corr[1]}** ({corr_vals[min_corr]:.2f})."
    return desc

def describe_line_chart(data, x_col, y_col, title_prefix=""):
    """Describe a line chart (e.g., tutoring sessions vs success rate)."""
    if data.empty or len(data) < 2:
        return "Not enough data to describe a trend."
    # Rough trend using first and last points
    y_vals = data[y_col].values
    if y_vals[-1] > y_vals[0]:
        trend = "increasing"
    elif y_vals[-1] < y_vals[0]:
        trend = "decreasing"
    else:
        trend = "stable"
    desc = f"**{title_prefix}**  \n"
    desc += f"As **{x_col}** increases, the **{y_col}** shows an **{trend}** trend. "
    desc += f"The values range from {y_vals.min():.1f} to {y_vals.max():.1f}."
    return desc

# ------------------------------------------------------------------------------
# LOAD DATA (Helps ensure my dashboard is connected to a live data source - use need for up to date visuals)
# ------------------------------------------------------------------------------
script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, "student_performance_data.csv")
excel_path = os.path.join(script_dir, "student_performance_data.xlsx")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
elif os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
else:
    st.error("Data file not found. Please ensure the CSV or XLSX is in the same folder.")
    st.stop()

df = df.fillna("Unknown")

# ------------------------------------------------------------------------------
# COLUMNS VALIDATION
# ------------------------------------------------------------------------------
expected_cols = [
    "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
    "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
    "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Physical_Activity", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender", "Exam_Score"
]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns in dataset: {missing}")
    st.stop()

# ------------------------------------------------------------------------------
# MAPPINGS FOR CORRELATIONS - ORDINAL
# ------------------------------------------------------------------------------
ordinal_maps = {
    "Parental_Involvement": {"Low": 1, "Medium": 2, "High": 3},
    "Access_to_Resources": {"Low": 1, "Medium": 2, "High": 3},
    "Motivation_Level": {"Low": 1, "Medium": 2, "High": 3},
    "Teacher_Quality": {"Low": 1, "Medium": 2, "High": 3},
    "Peer_Influence": {"Negative": 1, "Neutral": 2, "Positive": 3},
    "Parental_Education_Level": {"High School": 1, "College": 2, "Postgraduate": 3},
    "Distance_from_Home": {"Near": 1, "Moderate": 2, "Far": 3},
}

for col, mapping in ordinal_maps.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# numeric columns assurance - hard wire
numeric_cols = [
    "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
    "Motivation_Level", "Tutoring_Sessions", "Physical_Activity",
    "Parental_Involvement", "Access_to_Resources", "Teacher_Quality",
    "Peer_Influence", "Parental_Education_Level", "Distance_from_Home",
    "Exam_Score"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------------------------
# BASIC CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Accessible Student Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# Accessibility needs - colour-blindness friendly palette
sns.set_palette("colorblind")
plt.style.use("seaborn-v0_8")
plt.rcParams.update({"font.size": 12})

HIGH_GRADE_THRESHOLD = 80
df["High_Grade"] = (df["Exam_Score"] >= HIGH_GRADE_THRESHOLD).astype(int)

# ------------------------------------------------------------------------------
# DASHBOARD TITLE AND INTRODUCTION (I need to rememer digital services need a meaningful title)
# ------------------------------------------------------------------------------
st.title("📚 Accessible Student Performance Dashboard")
st.markdown("### For students, parents, teachers, and analysts")
st.markdown(
    "Use filters on the left to focus on specific groups of students, "
    "then learn about them from the user needs based tabs."
)
st.markdown("---")

# ------------------------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------------------------
st.sidebar.header("🔍 Filter the Data")

def multiselect_all(label, series):
    options = sorted(series.dropna().unique())
    return st.sidebar.multiselect(label, options=options, default=options)

school_type_filter = multiselect_all("School Type", df["School_Type"])
gender_filter = multiselect_all("Gender", df["Gender"])
income_filter = multiselect_all("Family Income", df["Family_Income"])
extra_filter = multiselect_all("Extracurricular Activities", df["Extracurricular_Activities"])
ld_filter = multiselect_all("Learning Disabilities", df["Learning_Disabilities"])
internet_filter = multiselect_all("Internet Access", df["Internet_Access"])

attendance_min, attendance_max = int(df["Attendance"].min()), int(df["Attendance"].max())
attendance_range = st.sidebar.slider(
    "Attendance (%)",
    min_value=attendance_min,
    max_value=attendance_max,
    value=(attendance_min, attendance_max)
)

hours_min, hours_max = float(df["Hours_Studied"].min()), float(df["Hours_Studied"].max())
hours_range = st.sidebar.slider(
    "Hours Studied",
    min_value=float(hours_min),
    max_value=float(hours_max),
    value=(float(hours_min), float(hours_max))
)

distance_reverse_map = {1: "Near", 2: "Moderate", 3: "Far"}
df["Distance_Label"] = df["Distance_from_Home"].map(distance_reverse_map).fillna("Unknown")
distance_filter = multiselect_all("Distance from Home", df["Distance_Label"])

HIGH_GRADE_THRESHOLD = st.sidebar.slider(
    "High-grade threshold (Exam Score ≥)",
    min_value=50,
    max_value=100,
    value=80,
    step=1
)
df["High_Grade"] = (df["Exam_Score"] >= HIGH_GRADE_THRESHOLD).astype(int)

tutoring_min, tutoring_max = int(df["Tutoring_Sessions"].min()), int(df["Tutoring_Sessions"].max())
tutoring_range = st.sidebar.slider(
    "Number of Tutoring Sessions",
    min_value=tutoring_min,
    max_value=tutoring_max,
    value=(tutoring_min, tutoring_max)
)

# ------------------------------------------------------------------------------
# APPLY ALL MY FILTERS
# ------------------------------------------------------------------------------
filtered_df = df[
    (df["School_Type"].isin(school_type_filter)) &
    (df["Gender"].isin(gender_filter)) &
    (df["Family_Income"].isin(income_filter)) &
    (df["Tutoring_Sessions"].between(tutoring_range[0], tutoring_range[1])) &
    (df["Extracurricular_Activities"].isin(extra_filter)) &
    (df["Learning_Disabilities"].isin(ld_filter)) &
    (df["Internet_Access"].isin(internet_filter)) &
    (df["Attendance"].between(attendance_range[0], attendance_range[1])) &
    (df["Hours_Studied"].between(hours_range[0], hours_range[1])) &
    (df["Distance_Label"].isin(distance_filter) if distance_filter else True)
].copy()

# ------------------------------------------------------------------------------
# THE KEY METRICS AND AUTOMATIC INSIGHTS
# ------------------------------------------------------------------------------

count = len(filtered_df)
high = filtered_df["High_Grade"].sum()

rate = safe_rate(high, count)
avg_score = safe_mean(filtered_df["Exam_Score"])
avg_att = safe_mean(filtered_df["Attendance"])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total number of students", count)

with col2:
    st.metric("Avg Exam Score",
              f"{avg_score:.1f}" if avg_score is not None else "Unreliable")

with col3:
    st.metric("Avg Attendance",
              f"{avg_att:.1f}%" if avg_att is not None else "Unreliable")

with col4:
    if rate is None:
        st.metric(" High‑grade rate", "N/A")
    else:
        st.metric("High‑grade rate", f"{rate*100:.1f}%")

st.caption(reliability_caption(filtered_df))

st.markdown("### 🔎 Key Insights for Current Selection")

def auto_insights(df_view):
    if len(df_view) == 0:
        st.info("No data available for the current filters.")
        return
    insights = []
    # Attendance insight
    if df_view["Attendance"].nunique() > 1:
        corr_att = df_view["Attendance"].corr(df_view["Exam_Score"])
        if corr_att > 0.5:
            insights.append("Students with higher attendance tend to achieve much higher exam scores.")
        elif corr_att > 0.3:
            insights.append("Better attendance is linked to higher exam scores.")
        elif corr_att > 0.1:
            insights.append("There is a small link between attendance and exam scores.")
        else:
            insights.append("Attendance does not show a clear link with exam scores in this group.")
    # Hours studied insight
    if df_view["Hours_Studied"].nunique() > 1:
        corr_hours = df_view["Hours_Studied"].corr(df_view["Exam_Score"])
        if corr_hours > 0.5:
            insights.append("Students who study more hours usually achieve much higher scores.")
        elif corr_hours > 0.3:
            insights.append("Studying more hours is linked to higher exam scores.")
        elif corr_hours > 0.1:
            insights.append("Study hours have a small positive effect on exam scores.")
        else:
            insights.append("Study hours do not show a clear link with exam scores in this group.")
    # Family income difference
    if df_view["Family_Income"].nunique() > 1:
        income_avg = df_view.groupby("Family_Income")["Exam_Score"].mean()
        diff_inc = income_avg.max() - income_avg.min()
        if diff_inc > 3:
            insights.append(f"Family income groups differ by about {diff_inc:.1f} points in average exam scores.")
    # Learning disabilities gap
    if df_view["Learning_Disabilities"].nunique() > 1:
        ld_avg = df_view.groupby("Learning_Disabilities")["Exam_Score"].mean()
        if "Yes" in ld_avg.index and "No" in ld_avg.index:
            gap = ld_avg["No"] - ld_avg["Yes"]
            if abs(gap) > 3:
                direction = "lower" if gap > 0 else "higher"
                insights.append(
                    f"Students with learning disabilities tend to score {abs(gap):.1f} points {direction} than those without."
                )
    if not insights:
        st.write("- No strong patterns detected with the current filters.")
    else:
        for i in insights:
            st.write(f"- {i}")

auto_insights(filtered_df)
st.markdown("---")

# ----------------------------------------------------------------------
# FACTORS INFLUENCING HIGH SCORES
# ----------------------------------------------------------------------
def explain_high_score_factors(df, threshold=70):

    if df.empty:
        return "No data available for analysis."

    df = df.copy()

    # Create the binary target ONCE
    df["High_Performer"] = (df["Exam_Score"] >= threshold).astype(int)

    # Check class balance AFTER creating the correct target
    class_counts = df["High_Performer"].value_counts()

    if len(class_counts) < 2:
        return (
            "⚠️ Not enough variation in high‑score outcomes to train a reliable model. "
            f"All students in this filtered group are classified as {class_counts.index[0]}."
        )

    # Check minimum sample size
    if len(df) < MIN_RELIABLE_N:
        return (
            f"⚠️ Not enough data to train a reliable model (n={len(df)}). "
            "Increase the sample size by widening your filters."
        )

    # Select numeric predictors
    predictors = [
        "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
        "Motivation_Level", "Tutoring_Sessions", "Physical_Activity",
        "Parental_Involvement", "Access_to_Resources", "Teacher_Quality",
        "Peer_Influence", "Parental_Education_Level", "Distance_from_Home"
    ]

    X = df[predictors].fillna(0)
    y = df["High_Performer"]

    # Scale predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit logistic regression (safe now)
    model = LogisticRegression(max_iter=200)
    model.fit(X_scaled, y)

    # Extract coefficients
    importance = pd.DataFrame({
        "Factor": predictors,
        "Influence": model.coef_[0]
    }).sort_values("Influence", ascending=False)

    # Build explanation
    explanation = "###  Factors that most influence achieving a high exam score\n"
    explanation += f"(High score defined as ≥ {threshold})\n\n"

    for _, row in importance.iterrows():
        direction = "positive" if row["Influence"] > 0 else "negative"
        strength = abs(row["Influence"])

        if strength > 1.0:
            level = "strong"
        elif strength > 0.5:
            level = "moderate"
        else:
            level = "weak"

        explanation += (
            f"- **{row['Factor']}** has a **{level} {direction} influence** "
            "on achieving a high score.\n"
        )

    return explanation

# ------------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# ------------------------------------------------------------------------------
def plot_bar_with_stats(data, x_col, y_col, title, xlabel, ylabel, ax, value_suffix=''):
    """
    Plot bar chart with mean of y_col per x_col.
    Shows counts above bars (without 'n=') and mean values inside bars.
    X‑axis ticks are forced to whole numbers when the x‑values are integers.
    """
    grouped = data.groupby(x_col)[y_col].agg(['mean', 'count']).reset_index()
    x = grouped[x_col]
    means = grouped['mean']
    counts = grouped['count']

    colors = sns.color_palette("colorblind", len(x))
    bars = ax.bar(x, means, color=colors, edgecolor='black', alpha=0.8)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim(0, max(100, means.max() + 5))

    # --- Remove 'n=' from top labels ---
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontsize=9)

    # --- Show mean values inside bars ---
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        facecolor = bar.get_facecolor()
        if len(facecolor) == 3:
            r, g, b = facecolor
            luminance = 0.299*r + 0.587*g + 0.114*b
        else:
            r, g, b, _ = facecolor
            luminance = 0.299*r + 0.587*g + 0.114*b
        text_color = 'white' if luminance < 0.5 else 'black'
        y_pos = height / 2
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{mean_val:.1f}{value_suffix}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=text_color)

    # --- Force integer ticks only if x values are numeric and whole numbers ---
    if pd.api.types.is_numeric_dtype(x):
        # Check if all values are effectively integers (within a small tolerance)
        if all(abs(v - round(v)) < 1e-10 for v in x):
            ax.set_xticks(sorted(x))
            ax.set_xticklabels([int(round(v)) for v in sorted(x)])

def analyze_tutoring_performance(df):
    """Create a scatter plot of tutoring sessions vs exam score with regression line."""
    if len(df) == 0:
        st.info("No data available for tutoring analysis.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(x=df["Tutoring_Sessions"], y=df["Exam_Score"],
                ax=ax, scatter_kws={'alpha':0.7, 'color':'#1D4ED8', 'edgecolor':'black'},
                line_kws={'color':'red'})
    ax.set_xlabel("Tutoring Sessions")
    ax.set_ylabel("Exam Score")
    ax.set_title("Tutoring Sessions vs Exam Score (individual students)")
    ax.set_ylim(0, 100)
    corr = df["Tutoring_Sessions"].corr(df["Exam_Score"]) if len(df) > 1 else np.nan
    if not np.isnan(corr):
        ax.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    st.pyplot(fig)
    desc = describe_scatter(df, "Tutoring_Sessions", "Exam_Score", "📊 Tutoring Scatter Insight")
    st.info(desc)

# ------------------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------------------
tab_overview, tab_student, tab_parent, tab_socio, tab_attendance, tab_ld, tab_factors, tab_data  = st.tabs([
    "Overview",
    "Student Success Profile",
    "Parent Guidance Hub",
    "socio-economic Impact",
    "Attendance & Performance",
    "Learning Disabilities Insights",
    "What Influences High Scores?",
    "Raw Data"    
])

with tab_factors:
    st.subheader("What factors influence achieving a high exam score?")

    # If no data at all
    if len(filtered_df) == 0:
        st.info("No data available for the current selection.")
        st.stop()

    # Prepare target
    threshold = 70
    filtered_df = filtered_df.copy()
    filtered_df["High_Performer"] = (filtered_df["Exam_Score"] >= threshold).astype(int)

    # Class balance check
    class_counts = filtered_df["High_Performer"].value_counts()

    # Minimum sample size check
    if len(filtered_df) < MIN_RELIABLE_N:
        st.warning(
            f"⚠️ Not enough data to train a reliable model (n={len(filtered_df)}). "
            "Widen your filters to include more students."
        )

        # Greyed‑out placeholder chart
        st.markdown("### Factor Influence Chart (Unavailable)")
        st.info("Not enough data to compute factor influence.")
        fig_placeholder = px.bar(
            x=[0], y=["Insufficient Data"],
            orientation="h",
            title="Factor Influence (Unavailable)",
            color=[0],
            color_continuous_scale="Greys"
        )
        st.plotly_chart(fig_placeholder, use_container_width=True)
        st.stop()

    # Single‑class check
    if len(class_counts) < 2:
        st.warning(
            "⚠️ Not enough variation in high‑score outcomes to train a reliable model. "
            f"All students in this filtered group are classified as {class_counts.index[0]}."
        )

        # Greyed‑out placeholder chart
        st.markdown("###  Factor Influence Chart (Unavailable)")
        st.info("Cannot compute factor influence because all students fall into one outcome class.")
        fig_placeholder = px.bar(
            x=[0], y=["Single Outcome Class"],
            orientation="h",
            title="Factor Influence (Unavailable)",
            color=[0],
            color_continuous_scale="Greys"
        )
        st.plotly_chart(fig_placeholder, use_container_width=True)
        st.stop()

    #  model training safety point/benchmark
    predictors = [
        "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
        "Motivation_Level", "Tutoring_Sessions", "Physical_Activity",
        "Parental_Involvement", "Access_to_Resources", "Teacher_Quality",
        "Peer_Influence", "Parental_Education_Level", "Distance_from_Home"
    ]

    X = filtered_df[predictors].fillna(0)
    y = filtered_df["High_Performer"]

    # Scale predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model safely
    model = LogisticRegression(max_iter=200)
    model.fit(X_scaled, y)

    # Extract coefficients
    importance = pd.DataFrame({
        "Factor": predictors,
        "Influence": model.coef_[0]
    }).sort_values("Influence", ascending=True)

    # Narrative explanation
    explanation = explain_high_score_factors(filtered_df, threshold=threshold)
    st.markdown(explanation)

    # Visualisation
    st.markdown("###  Factor Influence Chart")

    fig_imp = px.bar(
        importance,
        x="Influence",
        y="Factor",
        orientation="h",
        title="Factor Influence on Achieving a High Exam Score",
        color="Influence",
        color_continuous_scale="RdBu",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Ranked table
    st.markdown("###  Ranked Factors Table")

    ranked_table = importance.copy()
    ranked_table["Strength"] = ranked_table["Influence"].abs().round(2)
    ranked_table["Direction"] = ranked_table["Influence"].apply(
        lambda x: "Positive" if x > 0 else "Negative"
    )

    st.dataframe(
        ranked_table[["Factor", "Direction", "Strength"]]
            .sort_values("Strength", ascending=False),
        use_container_width=True
    )
# ------------------------------------------------------------------------------
# OVERVIEW (stacked layout)
# ------------------------------------------------------------------------------
with tab_overview:
    st.subheader("Score Distribution")
    if len(filtered_df) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(filtered_df["Exam_Score"], bins=12, color=sns.color_palette("colorblind")[0],
                edgecolor="black", alpha=0.9)
        ax.axvline(filtered_df["Exam_Score"].mean(), color='red', linestyle='--', label=f'Mean: {filtered_df["Exam_Score"].mean():.1f}')
        ax.axvline(filtered_df["Exam_Score"].median(), color='green', linestyle='--', label=f'Median: {filtered_df["Exam_Score"].median():.1f}')
        ax.set_xlabel("Exam Score")
        ax.set_ylabel("Number of Students")
        ax.set_title("How exam scores are spread")
        ax.legend()
        st.pyplot(fig)
        desc = describe_histogram(filtered_df, "Exam_Score", " Score Distribution Insight")
        st.info(desc)
    else:
        st.info("No data for current filters.")

    st.markdown("---")
    st.subheader("Correlation Heatmap (Key Variables)")
    if len(filtered_df) > 1:
        corr_vars = [
            "Exam_Score", "Hours_Studied", "Attendance", "Previous_Scores",
            "Motivation_Level", "Sleep_Hours", "Tutoring_Sessions", "Physical_Activity",
            "Parental_Involvement", "Access_to_Resources", "Teacher_Quality",
            "Peer_Influence", "Parental_Education_Level", "Distance_from_Home"
        ]
        corr_df = filtered_df[corr_vars].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_df, annot=True, cmap="Blues", ax=ax, fmt=".2f",
                    cbar_kws={'label': 'Correlation'})
        ax.set_title("Correlation between behaviours and exam scores")
        st.pyplot(fig)
        desc = describe_heatmap(corr_df, " Correlation Insight")
        st.info(desc)
    else:
        st.info("Not enough data to compute correlations.")

    st.markdown("---")
    st.subheader("Average Exam Score by School Type")
    if filtered_df["School_Type"].nunique() > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_bar_with_stats(filtered_df, "School_Type", "Exam_Score",
                            "Average Exam Score by School Type",
                            "School Type", "Exam Score", ax, value_suffix='%')
        st.pyplot(fig)
        desc = describe_bar_chart(filtered_df, "School_Type", "Exam_Score", " School Type Insight")
        st.info(desc)
    else:
        st.info("No school type data for current filters.")

    st.markdown("---")
    st.subheader("Average Exam Score by Gender")
    if filtered_df["Gender"].nunique() > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_bar_with_stats(filtered_df, "Gender", "Exam_Score",
                            "Average Exam Score by Gender",
                            "Gender", "Exam Score", ax, value_suffix='%')
        st.pyplot(fig)
        desc = describe_bar_chart(filtered_df, "Gender", "Exam_Score", "👫 Gender Insight")
        st.info(desc)
    else:
        st.info("No gender data for current filters.")

# ------------------------------------------------------------------------------
# STUDENT SUCCESS PROFILE
# ------------------------------------------------------------------------------
with tab_student:
    st.subheader("What behaviours are linked to high performance?")
    if len(filtered_df) == 0:
        st.info("No data for current filters.")
    else:
        st.markdown("#### Hours Studied vs Exam Score")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.regplot(x=filtered_df["Hours_Studied"], y=filtered_df["Exam_Score"],
                    ax=ax, scatter_kws={'alpha':0.7, 'color':"#1D4AC5", 'edgecolor':'black'},
                    line_kws={'color':'red'})
        ax.set_xlabel("Hours Studied")
        ax.set_ylabel("Exam Score")
        ax.set_title("More study hours, higher scores?")
        ax.set_ylim(0, 100)
        corr = filtered_df["Hours_Studied"].corr(filtered_df["Exam_Score"])
        ax.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        st.pyplot(fig)
        desc = describe_scatter(filtered_df, "Hours_Studied", "Exam_Score", " Study Hours Insight")
        st.info(desc)

        st.markdown("---")
        st.markdown("#### Sleep Hours vs Exam Score")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.regplot(x=filtered_df["Sleep_Hours"], y=filtered_df["Exam_Score"],
                    ax=ax2, scatter_kws={'alpha':0.7, 'color':'#10B981', 'edgecolor':'black'},
                    line_kws={'color':'red'})
        ax2.set_xlabel("Sleep Hours")
        ax2.set_ylabel("Exam Score")
        ax2.set_title("Sleep and performance")
        ax2.set_ylim(0, 100)
        corr = filtered_df["Sleep_Hours"].corr(filtered_df["Exam_Score"])
        ax2.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax2.transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        st.pyplot(fig2)
        desc = describe_scatter(filtered_df, "Sleep_Hours", "Exam_Score", " Sleep Insight")
        st.info(desc)

        st.markdown("---")
        st.markdown("#### Motivation Level vs Exam Score")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.regplot(x=filtered_df["Motivation_Level"], y=filtered_df["Exam_Score"],
                    ax=ax3, scatter_kws={'alpha':0.7, 'color':'#F97316', 'edgecolor':'black'},
                    line_kws={'color':'red'})
        
        ax3.set_xlabel("Motivation Level (1=Low, 2=Medium, 3=High)")
        ax3.set_ylabel("Exam Score")
        ax3.set_title("Motivation and performance")
        ax3.set_ylim(0, 100)
        corr = filtered_df["Motivation_Level"].corr(filtered_df["Exam_Score"])
        ax3.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax3.transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        st.pyplot(fig3)
        desc = describe_scatter(filtered_df, "Motivation_Level", "Exam_Score", " Motivation Insight")
        st.info(desc)

        st.markdown("---")
        st.markdown("#### Tutoring Sessions vs Average Exam Score")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        plot_bar_with_stats(filtered_df, "Tutoring_Sessions", "Exam_Score",
                            "Tutoring Sessions vs Exam Score",
                            "Tutoring Sessions", "Exam Score", ax4, value_suffix='%')
        st.pyplot(fig4)
        desc = describe_bar_chart(filtered_df, "Tutoring_Sessions", "Exam_Score", " Tutoring Insight")
        st.info(desc)

        st.markdown("---")
        st.markdown("#### Previous Scores vs Current Exam Score")
        fig5, ax5 = plt.subplots(figsize=(7, 4))
        sns.regplot(x=filtered_df["Previous_Scores"], y=filtered_df["Exam_Score"],
                    ax=ax5, scatter_kws={'alpha':0.7, 'color':'#6366F1', 'edgecolor':'black'},
                    line_kws={'color':'red'})
        ax5.set_xlabel("Previous Scores")
        ax5.set_ylabel("Exam Score")
        ax5.set_title("Do past results predict current performance?")
        ax5.set_ylim(0, 100)
        corr = filtered_df["Previous_Scores"].corr(filtered_df["Exam_Score"])
        ax5.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax5.transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        st.pyplot(fig5)
        desc = describe_scatter(filtered_df, "Previous_Scores", "Exam_Score", " Previous Scores Insight")
        st.info(desc)

        st.markdown("---")
        analyze_tutoring_performance(filtered_df)

# ------------------------------------------------------------------------------
# PARENT GUIDANCE HUB
# ------------------------------------------------------------------------------
with tab_parent:
    st.subheader("What can parents do to support better grades?")
    if len(filtered_df) == 0:
        st.info("No data for current filters.")
    else:
        st.markdown("#### Parental Involvement vs Average Exam Score")
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_bar_with_stats(filtered_df, "Parental_Involvement", "Exam_Score",
                            "Parental Involvement vs Exam Score",
                            "Parental Involvement (1=Low,2=Medium,3=High)", "Exam Score", ax, value_suffix='%')
        st.pyplot(fig)
        desc = describe_bar_chart(filtered_df, "Parental_Involvement", "Exam_Score", " Parental Involvement Insight")
        st.info(desc)
        

        st.markdown("---")
        st.markdown("#### Access to Resources vs Average Exam Score")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        plot_bar_with_stats(filtered_df, "Access_to_Resources", "Exam_Score",
                            "Access to Resources vs Exam Score",
                            "Access to Resources (1=Low,2=Medium,3=High)", "Exam Score", ax2, value_suffix='%')
        st.pyplot(fig2)
        desc = describe_bar_chart(filtered_df, "Access_to_Resources", "Exam_Score", " Resource Access Insight")
        st.info(desc)

        st.markdown("---")
        st.markdown("#### Parental Education Level vs Average Exam Score")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        plot_bar_with_stats(filtered_df, "Parental_Education_Level", "Exam_Score",
                            "Parental Education Level vs Exam Score",
                            "Parental Education (1=High School,2=College,3=Postgrad)", "Exam Score", ax3, value_suffix='%')
        st.pyplot(fig3)
        desc = describe_bar_chart(filtered_df, "Parental_Education_Level", "Exam_Score", " Parental Education Insight")
        st.info(desc)

        st.markdown("---")
        st.markdown("#### Internet Access vs High-grade Rate")
        internet_rate = filtered_df.groupby("Internet_Access")["High_Grade"].agg(['mean', 'count']).reset_index()
        x = internet_rate["Internet_Access"]
        means = internet_rate['mean'] * 100
        counts = internet_rate['count']
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        colors = sns.color_palette("colorblind", len(x))
        bars = ax4.bar(x, means, color=colors, edgecolor='black', alpha=0.8)
        ax4.set_ylabel("High-grade rate (%)")
        ax4.set_xlabel("Internet Access")
        ax4.set_title("Internet Access vs High-grade Rate")
        ax4.set_ylim(0, max(100, max(means) + 5))

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                     f'{count}', ha='center', va='bottom', fontsize=9)

        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            facecolor = bar.get_facecolor()
            r, g, b, _ = facecolor
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = 'white' if luminance < 0.5 else 'black'
            y_pos = height / 2
            ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                     f'{mean_val:.0f}%', ha='center', va='center',
                     fontsize=10, fontweight='bold', color=text_color)
        st.pyplot(fig4)

        desc = f"** Internet Access Insight**  \n"
        if len(internet_rate) == 2:
            yes_rate = internet_rate.loc[internet_rate['Internet_Access']=='Yes', 'mean'].values[0]*100
            no_rate = internet_rate.loc[internet_rate['Internet_Access']=='No', 'mean'].values[0]*100
            desc += f"Students with internet access have a high‑grade rate of **{yes_rate:.1f}%**, compared to **{no_rate:.1f}%** for those without."
        else:
            desc += f"The high‑grade rate for the selected group is **{means[0]:.1f}%**."
        st.info(desc)

# ------------------------------------------------------------------------------
# SOCIO-ECONOMIC IMPACT
# ------------------------------------------------------------------------------
with tab_socio:
    st.subheader("Do socio-economic factors affect performance?")
    if len(filtered_df) == 0:
        st.info("No data for current filters.")
    else:
        st.markdown("#### Family Income vs Average Exam Score")
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_bar_with_stats(filtered_df, "Family_Income", "Exam_Score",
                            "Family Income vs Exam Score",
                            "Family Income", "Exam Score", ax, value_suffix='%')
        st.pyplot(fig)
        desc = describe_bar_chart(filtered_df, "Family_Income", "Exam_Score", " Family Income Insight")
        st.info(desc)

        st.markdown("---")
        st.markdown("#### Distance from Home vs Exam Score")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.regplot(x=filtered_df["Distance_from_Home"], y=filtered_df["Exam_Score"],
                    ax=ax2, scatter_kws={'alpha':0.7, 'color':'#DC2626', 'edgecolor':'black'},
                    line_kws={'color':'red'})
        ax2.set_xlabel("Distance from Home (1=Near, 2=Moderate, 3=Far)")
        ax2.set_ylabel("Exam Score")
        ax2.set_title("Does distance affect performance?")
        ax2.set_ylim(0, 100)
        corr = filtered_df["Distance_from_Home"].corr(filtered_df["Exam_Score"])
        ax2.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax2.transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        st.pyplot(fig2)
        desc = describe_scatter(filtered_df, "Distance_from_Home", "Exam_Score", " Distance Insight")
        st.info(desc)

        st.markdown("---")
        st.markdown("#### School Type vs Average Exam Score")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        plot_bar_with_stats(filtered_df, "School_Type", "Exam_Score",
                            "School Type vs Exam Score",
                            "School Type", "Exam Score", ax3, value_suffix='%')
        st.pyplot(fig3)
        desc = describe_bar_chart(filtered_df, "School_Type", "Exam_Score", " School Type Insight")
        st.info(desc)

        st.markdown("---")
        st.markdown("#### Teacher Quality vs Average Exam Score")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        plot_bar_with_stats(filtered_df, "Teacher_Quality", "Exam_Score",
                            "Teacher Quality vs Exam Score",
                            "Teacher Quality (1=Low, 2=Medium, 3=High)", "Exam Score", ax4, value_suffix='%')
        st.pyplot(fig4)
        desc = describe_bar_chart(filtered_df, "Teacher_Quality", "Exam_Score", " Teacher Quality Insight")
        st.info(desc)

        st.markdown("---")
        st.markdown("#### Peer Influence vs Average Exam Score")
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        plot_bar_with_stats(filtered_df, "Peer_Influence", "Exam_Score",
                            "Peer Influence vs Exam Score",
                            "Peer Influence (1=Negative, 2=Neutral, 3=Positive)", "Exam Score", ax5, value_suffix='%')
        st.pyplot(fig5)
        desc = describe_bar_chart(filtered_df, "Peer_Influence", "Exam_Score", " Peer Influence Insight")
        st.info(desc)

# ------------------------------------------------------------------------------
# ATTENDANCE & PERFORMANCE
# ------------------------------------------------------------------------------
with tab_attendance:
    st.subheader("Does attendance affect grades?")
    if len(filtered_df) == 0:
        st.info("No data for current filters.")
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.regplot(x=filtered_df["Attendance"], y=filtered_df["Exam_Score"],
                    ax=ax, scatter_kws={'alpha':0.7, 'color':'#0EA5E9', 'edgecolor':'black'},
                    line_kws={'color':'red'})
        ax.set_xlabel("Attendance (%)")
        ax.set_ylabel("Exam Score")
        ax.set_title("Attendance vs Exam Score")
        ax.set_ylim(0, 100)
        corr = filtered_df["Attendance"].corr(filtered_df["Exam_Score"])
        ax.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        st.pyplot(fig)
        desc = describe_scatter(filtered_df, "Attendance", "Exam_Score", " Attendance Insight")
        st.info(desc)

        st.markdown("---")
        bins = [0, 60, 75, 90, 100]
        labels = ["<60%", "60--75%", "75--90%", "90--100%"]
        filtered_df["Attendance_Band"] = pd.cut(
            filtered_df["Attendance"], bins=bins, labels=labels, include_lowest=True
        )
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        plot_bar_with_stats(filtered_df.dropna(subset=["Attendance_Band"]), "Attendance_Band", "Exam_Score",
                            "Average Exam Score by Attendance Band",
                            "Attendance Band", "Exam Score", ax2, value_suffix='%')
        st.pyplot(fig2)
        desc = describe_bar_chart(filtered_df.dropna(subset=["Attendance_Band"]), "Attendance_Band", "Exam_Score", "📊 Attendance Band Insight")
        st.info(desc)

# ------------------------------------------------------------------------------
# 9F. LEARNING DISABILITIES INSIGHTS (stacked layout)
# ------------------------------------------------------------------------------
with tab_ld:
    st.subheader("What helps students with learning disabilities succeed?")
    ld_df = filtered_df[filtered_df["Learning_Disabilities"] == "Yes"]
    non_ld_df = filtered_df[filtered_df["Learning_Disabilities"] == "No"]

    if len(ld_df) == 0:
        st.info("No students with learning disabilities in the current selection.")
    else:
        st.markdown("#### LD vs Non-LD: Average Exam Score")
        if len(non_ld_df) > 0:
            comp = pd.DataFrame({
                "LD_Status": ["LD", "Non-LD"],
                "Avg_Score": [ld_df["Exam_Score"].mean(), non_ld_df["Exam_Score"].mean()],
                "Count": [len(ld_df), len(non_ld_df)]
            })
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = sns.color_palette("colorblind", 2)
            bars = ax.bar(comp["LD_Status"], comp["Avg_Score"],
                          color=colors, edgecolor='black', alpha=0.8)   # remove error bars
            ax.set_ylabel("Exam Score")
            ax.set_title("LD vs Non-LD: Average Exam Score")
            max_y = max(comp["Avg_Score"]) if len(comp) else 0
            ax.set_ylim(0, max(100, max_y + 5))

            for bar, count in zip(bars, comp["Count"]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{count}', ha='center', va='bottom', fontsize=9)   # remove 'n='

            for bar, avg_val in zip(bars, comp["Avg_Score"]):
                height = bar.get_height()
                facecolor = bar.get_facecolor()
                r, g, b, _ = facecolor
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = 'white' if luminance < 0.5 else 'black'
                y_pos = height / 2
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{avg_val:.0f}%', ha='center', va='center',   # add '%'
                        fontsize=10, fontweight='bold', color=text_color)
            st.pyplot(fig)

            gap = comp.loc[comp["LD_Status"]=="Non-LD", "Avg_Score"].values[0] - comp.loc[comp["LD_Status"]=="LD", "Avg_Score"].values[0]
            desc = f"**⚖️ LD vs Non‑LD Insight**  \n"
            desc += f"Students with learning disabilities score on average **{comp.loc[comp['LD_Status']=='LD', 'Avg_Score'].values[0]:.1f}**, "
            desc += f"while non‑LD students score **{comp.loc[comp['LD_Status']=='Non-LD', 'Avg_Score'].values[0]:.1f}**. "
            desc += f"The difference is **{abs(gap):.1f} points**."
            st.info(desc)
        else:
            st.info("No non-LD students in current selection for comparison.")

        st.markdown("---")
        st.markdown("#### LD Students: Extracurricular Activities vs Average Score")
        if ld_df["Extracurricular_Activities"].nunique() > 1:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            plot_bar_with_stats(ld_df, "Extracurricular_Activities", "Exam_Score",
                                "Extracurricular vs Exam Score (LD)",
                                "Extracurricular", "Exam Score", ax2, value_suffix='%')
            st.pyplot(fig2)
            desc = describe_bar_chart(ld_df, "Extracurricular_Activities", "Exam_Score", " Extracurricular (LD) Insight")
            st.info(desc)
        else:
            st.info("No variation in extracurricular activities for LD students.")

        st.markdown("---")
        st.markdown("#### LD Students: Tutoring Sessions vs Average Score")
        if ld_df["Tutoring_Sessions"].nunique() > 1:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            plot_bar_with_stats(ld_df, "Tutoring_Sessions", "Exam_Score",
                                "Tutoring Sessions vs Exam Score (LD)",
                                "Tutoring Sessions", "Exam Score", ax3, value_suffix='%')
            st.pyplot(fig3)
            desc = describe_bar_chart(ld_df, "Tutoring_Sessions", "Exam_Score", " Tutoring (LD) Insight")
            st.info(desc)
        else:
            st.info("No variation in tutoring sessions for LD students.")

        st.markdown("---")
        st.markdown("#### LD Students: Attendance vs Exam Score")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.regplot(x=ld_df["Attendance"], y=ld_df["Exam_Score"],
                    ax=ax4, scatter_kws={'alpha':0.7, 'color':'#EC4899', 'edgecolor':'black'},
                    line_kws={'color':'red'})
        ax4.set_xlabel("Attendance (%)")
        ax4.set_ylabel("Exam Score")
        ax4.set_title("LD Students: Attendance vs Exam Score")
        ax4.set_ylim(0, 100)
        corr = ld_df["Attendance"].corr(ld_df["Exam_Score"]) if len(ld_df) > 1 else np.nan
        if not np.isnan(corr):
            ax4.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax4.transAxes,
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        st.pyplot(fig4)
        if len(ld_df) > 1:
            desc = describe_scatter(ld_df, "Attendance", "Exam_Score", " Attendance (LD) Insight")
        else:
            desc = "Not enough LD students to analyze attendance correlation."
        st.info(desc)

# ------------------------------------------------------------------------------
# RAW DATA (New user need to see raw data for transparency)
# ------------------------------------------------------------------------------
with tab_data:
    st.subheader("Raw Data (Filtered)")
    st.dataframe(filtered_df.reset_index(drop=True))
    st.caption("You can scroll, sort, and export this data from the menu in the top-right of the table.")

# ------------------------------------------------------------------------------
# DOWNLOAD REPORT (Testing this due to question asked by participant during research)
# ------------------------------------------------------------------------------
tab_report = st.tabs(["Download Report"])[0]
with tab_report:
    st.subheader(" Download Full Report (ZIP)")
    st.markdown(
        "This report includes:\n"
        "- Key metrics\n"
        "- Auto‑generated insights\n"
        "- All major charts\n"
        "- A text summary\n"
        "- Filtered dataset (CSV)\n"
    )
    if st.button("Generate ZIP Report"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as z:
            # TEXT REPORT
            report_buffer = io.StringIO()
            report_buffer.write("STUDENT PERFORMANCE REPORT\n")
            report_buffer.write(f"Generated: {datetime.now()}\n\n")
            report_buffer.write("=== KEY METRICS ===\n")
            report_buffer.write(f"Total number of students: {len(filtered_df)}\n")
            report_buffer.write(f"Average Exam Score: {filtered_df['Exam_Score'].mean():.2f}\n")
            report_buffer.write(f"Average Attendance: {filtered_df['Attendance'].mean():.2f}%\n")
            report_buffer.write(f"High‑grade rate: {filtered_df['High_Grade'].mean()*100:.1f}%\n\n")
            report_buffer.write("=== INSIGHTS ===\n")
            insights = []
            if filtered_df["Attendance"].nunique() > 1:
                corr_att = filtered_df["Attendance"].corr(filtered_df["Exam_Score"])
                insights.append(f"Attendance correlation: {corr_att:.2f}")
            if filtered_df["Hours_Studied"].nunique() > 1:
                corr_hours = filtered_df["Hours_Studied"].corr(filtered_df["Exam_Score"])
                insights.append(f"Hours studied correlation: {corr_hours:.2f}")
            for i in insights:
                report_buffer.write(f"- {i}\n")
            z.writestr("report.txt", report_buffer.getvalue())

            # SAVE CHARTS
            def save_fig_to_zip(fig, filename):
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format="png", bbox_inches="tight")
                z.writestr(filename, img_buffer.getvalue())

            # Score distribution
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(filtered_df["Exam_Score"], bins=12, color="skyblue", edgecolor="black")
            ax.set_title("Score Distribution")
            save_fig_to_zip(fig, "score_distribution.png")
            plt.close(fig)

            # Hours studied vs exam score
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.regplot(x=filtered_df["Hours_Studied"], y=filtered_df["Exam_Score"], ax=ax)
            ax.set_title("Hours Studied vs Exam Score")
            save_fig_to_zip(fig, "hours_vs_score.png")
            plt.close(fig)

            # Attendance vs exam score
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.regplot(x=filtered_df["Attendance"], y=filtered_df["Exam_Score"], ax=ax)
            ax.set_title("Attendance vs Exam Score")
            save_fig_to_zip(fig, "attendance_vs_score.png")
            plt.close(fig)

            # FILTERED DATASET
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            z.writestr("filtered_data.csv", csv_buffer.getvalue())

        st.download_button(
            label=" Download ZIP Report",
            data=zip_buffer.getvalue(),
            file_name="student_performance_report.zip",
            mime="application/zip"
        )

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Laré Akinmutande (Feb - March 2026) Dashboard updates automatically when filters change. Theme and layout are designed for clarity and accessibility."
)

# ------------------------------------------------------------------------------
# DOWNLOAD REPORT 2 (Testing this due to question asked by participant during research)
# ------------------------------------------------------------------------------
with st.sidebar.expander(" Download Report"):
    if st.button("Generate Report"):
        report_lines = []
        report_lines.append("STUDENT PERFORMANCE REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("=== KEY METRICS ===")
        report_lines.append(f"Total number of students: {len(filtered_df)}")
        report_lines.append(f"Average Exam Score: {filtered_df['Exam_Score'].mean():.2f}")
        report_lines.append(f"Average Attendance: {filtered_df['Attendance'].mean():.2f}%")
        report_lines.append(f"High-grade rate (≥{HIGH_GRADE_THRESHOLD}): {filtered_df['High_Grade'].mean()*100:.1f}%")
        report_lines.append("")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            report_text = "\n".join(report_lines)
            zip_file.writestr("00_summary_metrics.txt", report_text)

            report_charts = [
                ("School_Type", "Performance by School Type", "01_Overview_School.png"),
                ("Gender", "Performance by Gender", "02_Overview_Gender.png"),
                ("Motivation_Level", "Success vs Motivation", "03_Success_Motivation.png"),
                ("Parental_Education_Level", "Impact of Parental Education", "04_Parent_Education.png"),
                ("Parental_Involvement", "Impact of Parental Involvement", "05_Parent_Involvement.png"),
                ("Family_Income", "socio-economic Performance", "06_socio-economic_Income.png"),
                ("Access_to_Resources", "Impact of Resource Access", "07_socio-economic_Resources.png"),
                ("Learning_Disabilities", "Disability Success Insights", "08_LD_Success_Insights.png")
            ]

            for col, title, filename in report_charts:
                if col in filtered_df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_bar_with_stats(filtered_df, col, "Exam_Score", title, col.replace("_", " "), "Average Score", ax, value_suffix='%')
                    chart_buf = io.BytesIO()
                    plt.savefig(chart_buf, format='png', bbox_inches='tight')
                    plt.close(fig)
                    zip_file.writestr(filename, chart_buf.getvalue())

        st.success("✅ Full report with all visualisations is ready!")
        st.download_button(
            label=" Download ZIP Report",
            data=zip_buffer.getvalue(),
            file_name=f"student_performance_report_{datetime.now().strftime('%Y%m%d')}.zip",
            mime="application/zip"
        )
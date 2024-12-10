import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set page configuration
st.set_page_config(page_title="Lifestyle and Academic Performance", layout="wide")

# Title and Introduction
st.title("Daily Lifestyle and Academic Performance Analysis")
st.markdown("""
This app analyzes the impact of daily lifestyle habits on academic performance 
using clustering and regression techniques.
""")

# File Upload Section
uploaded_file = st.file_uploader("Upload the dataset", type=["csv"])
if uploaded_file:
    # Dataset Description
    st.write("### Dataset Description")
    st.markdown("""
    This dataset contains information about students' daily lifestyle habits, such as study hours, sleep hours, stress levels, 
    and their corresponding academic performance (GPA). The goal is to analyze the relationship between lifestyle factors 
    and academic success.
    """)
    
    # Load and display the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Overview")
    st.dataframe(data.head())

    # Display column names to help identify any discrepancies
    st.write("### Dataset Columns")
    st.write(data.columns)

    # Data Cleaning Section
    st.subheader("Data Cleaning and Preparation")
    missing_values = data.isnull().sum()
    st.write("Missing values in each column:")
    st.write(missing_values)

    if missing_values.any():
        st.write("Filling missing values with column means...")
        data.fillna(data.mean(), inplace=True)
        st.write("Missing values filled.")
    else:
        st.write("No missing values detected.")

    # Handle Categorical Columns
    st.write("### Encoding Categorical Columns")
    # Check for columns with object (string) type and encode them
    categorical_columns = data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    
    for column in categorical_columns:
        data[column] = le.fit_transform(data[column].astype(str))
        st.write(f"Encoded column: {column}")

    # Display summary statistics
        st.write("### Descriptive Statistics")
        st.write(data.describe())

    # Add additional description text
    st.markdown("""
    This table provides descriptive statistics for a dataset of 2,000 students, summarizing variables such as daily study hours, extracurricular, sleep, social, and physical activity hours, along with GPA and stress levels.
    """)


    # **Verify Features List**
    features = ['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 'Sleep_Hours_Per_Day', 
                'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'Stress_Level']

    # Verify if the columns exist in the dataset
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        st.write(f"Warning: The following features are missing in the dataset: {missing_features}")
    else:
        st.write("All selected features are present in the dataset.")

        # Correlation heatmap
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
        st.pyplot(plt)
        # Add additional description text
        st.markdown("""
        This correlation heatmap visualizes the relationships between variables, highlighting a strong positive correlation between study hours and GPA (0.73) and a negative correlation between study hours and stress levels (-0.50).
        """)

        # Feature Scaling
        data_scaled = StandardScaler().fit_transform(data[features])

        # Clustering Section
        st.subheader("K-Means Clustering")
        num_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        data['Cluster'] = kmeans.fit_predict(data_scaled)

        st.write("### Clustered Data")
        st.dataframe(data[['Student_ID', 'GPA', 'Cluster']].head())

        # Cluster visualization
        st.write("### Cluster Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data, x='Study_Hours_Per_Day', y='GPA', hue='Cluster', palette="viridis", ax=ax)
        ax.set_title("Clusters Based on Study Hours and GPA")
        st.pyplot(fig)

        # Regression Analysis Section
        st.subheader("Linear Regression")
        st.write("### Predicting GPA")
        selected_features = st.multiselect("Select Features for Regression", features, default=['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Stress_Level'])

        if selected_features:
            X = data[selected_features]
            y = data['GPA']

            # Train the regression model
            model = LinearRegression()
            model.fit(X, y)

            # Display coefficients
            st.write("**Regression Coefficients:**")
            for feature, coef in zip(selected_features, model.coef_):
                st.write(f"{feature}: {coef:.4f}")
            st.write(f"**Intercept:** {model.intercept_:.4f}")

            # Predict and evaluate
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            st.write(f"**Mean Squared Error:** {mse:.4f}")

            # Visualization
            st.write("### Regression Results Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=data['Study_Hours_Per_Day'], y=data['GPA'], color="blue", label="Actual", ax=ax)
            sns.lineplot(x=data['Study_Hours_Per_Day'], y=y_pred, color="red", label="Predicted", ax=ax)
            ax.set_title("Regression: Study Hours vs GPA")
            st.pyplot(fig)
            st.markdown("""
            This regression plot shows the relationship between study hours per day and GPA, with the red line indicating the predicted GPA values and demonstrating a positive linear trend.
            """)

        # Conclusion Section
        st.subheader("Conclusions and Recommendations")
        st.markdown("""
        **Key Takeaways:**
        - Students with balanced study and sleep habits tend to have higher GPA.
        - High stress negatively impacts academic performance.
        - Clustering revealed distinct groups of students based on their habits.

        **Recommendations:**
        - Encourage time management strategies for balancing study and extracurricular activities.
        - Promote stress-reduction techniques, such as physical activity and relaxation exercises.
        - Suggest optimizing sleep hours to enhance focus and performance.
        """)

else:
    st.warning("Please upload a dataset to begin.")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="DataFlow Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
    .title-style {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None

def main():
    st.markdown('<h1 class="title-style">📊 DataFlow Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Professional Data Analytics Without Coding")

    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Go to", [
        "🏠 Home",
        "📁 Data Upload",
        "🧹 Data Cleaning",
        "📈 Data Visualization", 
        "🔗 Correlation Analysis",
        "📊 Statistical Summary"
    ])

    if page == "🏠 Home":
        show_home()
    elif page == "📁 Data Upload":
        show_upload()
    elif page == "🧹 Data Cleaning":
        show_cleaning()
    elif page == "📈 Data Visualization":
        show_visualization()
    elif page == "🔗 Correlation Analysis":
        show_correlation()
    elif page == "📊 Statistical Summary":
        show_statistics()

def show_home():
    st.markdown("""
    ## Welcome to DataFlow Pro! 🚀

    **Professional Data Analytics Made Simple** - Transform your raw CSV data into powerful insights without writing a single line of code.

    ### ✨ Key Features:
    - **📁 Smart CSV Upload** with instant data preview and validation
    - **🧹 Advanced Data Cleaning** with multiple preprocessing options
    - **📊 Rich Statistical Analysis** including descriptive statistics and data quality reports
    - **📈 Interactive Visualizations** - bar charts, line plots, scatter diagrams, pie charts, histograms, and box plots
    - **🔗 Correlation Intelligence** with heatmaps to discover hidden relationships
    - **📋 Multi-column Comparisons** for comprehensive data exploration
    - **🎨 Professional Themes** with dark/light mode support
    - **⚡ Lightning-fast Processing** with optimized performance

    ### 👥 Perfect For:
    - **📈 Data Analysts** - Advanced analytics and statistical insights
    - **🧑‍💼 Business Managers** - Executive dashboards and KPI tracking
    - **🚀 Product Teams** - User behavior analysis and metrics
    - **🎓 Researchers & Students** - Academic data analysis and learning
    - **💼 Consultants** - Client data exploration and reporting
    - **🏢 Small Business Owners** - Sales and operations analytics

    ### 🛠️ Powered By:
    - **Python 3.11+** - Latest language features
    - **Streamlit** - Modern web framework
    - **Pandas & NumPy** - Industry-standard data processing
    - **Matplotlib & Seaborn** - Statistical visualization
    - **Plotly** - Interactive charts and graphs
    - **Scikit-learn** - Machine learning preprocessing

    ---

    ### 🎯 Quick Start Guide:

    1. **📁 Upload Your Data** → Go to 'Data Upload' and select your CSV file
    2. **🧹 Clean Your Data** → Handle missing values and format issues
    3. **📊 Explore Statistics** → View comprehensive data summaries
    4. **📈 Create Visualizations** → Generate interactive charts
    5. **🔗 Find Correlations** → Discover relationships in your data
    6. **💾 Export Results** → Download your cleaned and analyzed data

    **Ready to unlock your data's potential? Start by uploading your CSV file! →**
    """)

    # Add some metrics for demonstration
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🚀 Features", "50+", delta="Advanced")
    with col2:
        st.metric("⚡ Processing", "Fast", delta="Optimized")
    with col3:
        st.metric("🎨 Charts", "6 Types", delta="Interactive")
    with col4:
        st.metric("💡 No Code", "100%", delta="User-Friendly")

def show_upload():
    st.header("📁 Data Upload Center")
    st.markdown("Upload your CSV file to begin your data analysis journey")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your CSV file to get started with professional data analysis",
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        try:
            # Show file details
            file_details = {
                "Filename": uploaded_file.name,
                "File Size": f"{uploaded_file.size / 1024:.2f} KB",
                "File Type": uploaded_file.type
            }

            # Display file info
            st.success("✅ File uploaded successfully!")

            # Read CSV file
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df.copy()
            st.session_state.original_data = df.copy()

            # Display basic metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("📊 Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("📋 Columns", df.shape[1])
            with col3:
                st.metric("❌ Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("💾 Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            with col5:
                st.metric("🔍 Duplicates", df.duplicated().sum())

            # Show data preview
            st.subheader("📋 Data Preview")
            st.markdown("First 10 rows of your dataset:")
            st.dataframe(df.head(10), use_container_width=True, height=350)

            # Show column information
            st.subheader("📊 Column Analysis")
            col_info = pd.DataFrame({
                'Column Name': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(2),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True, height=300)

            # Data quality assessment
            st.subheader("🎯 Data Quality Assessment")

            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            quality_score = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0

            col1, col2 = st.columns(2)
            with col1:
                st.metric("📈 Data Completeness", f"{quality_score:.1f}%")
                if quality_score >= 90:
                    st.success("🟢 Excellent data quality!")
                elif quality_score >= 70:
                    st.warning("🟡 Good data quality with minor issues")
                else:
                    st.error("🔴 Data quality needs improvement")

            with col2:
                # Show data types distribution
                dtype_counts = df.dtypes.value_counts()
                fig_pie = px.pie(
                    values=dtype_counts.values, 
                    names=dtype_counts.index, 
                    title="Data Types Distribution"
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure your file is a valid CSV format with proper encoding (UTF-8 recommended)")

    else:
        st.info("👆 Please upload a CSV file to get started with data analysis")

        # Show sample data format
        st.subheader("📝 Expected CSV Format")
        st.markdown("Your CSV should look something like this:")

        sample_data = pd.DataFrame({
            'Employee_ID': [1001, 1002, 1003, 1004],
            'Name': ['John Smith', 'Jane Doe', 'Bob Johnson', 'Alice Brown'],
            'Department': ['IT', 'HR', 'Finance', 'Marketing'],
            'Age': [28, 34, 31, 29],
            'Salary': [55000, 75000, 68000, 62000],
            'Performance_Score': [85, 92, 78, 90]
        })
        st.dataframe(sample_data, use_container_width=True)

def show_cleaning():
    st.header("🧹 Data Cleaning & Preprocessing")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload a CSV file first in the Data Upload section!")
        return

    df = st.session_state.data.copy()

    st.markdown("Transform and clean your data with professional preprocessing tools")

    # Data cleaning options
    st.subheader("🛠️ Cleaning Operations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔧 Missing Values Treatment**")
        missing_action = st.radio(
            "How would you like to handle missing values?",
            [
                "Keep as is", 
                "Drop rows with missing values", 
                "Fill with mean/mode", 
                "Fill with median", 
                "Fill with custom value",
                "Forward fill",
                "Backward fill"
            ]
        )

        if missing_action == "Fill with custom value":
            fill_value = st.text_input("Enter fill value:", "Unknown")

        # Show missing value summary
        if df.isnull().sum().sum() > 0:
            st.markdown("**Missing Values Summary:**")
            missing_summary = df.isnull().sum()
            missing_summary = missing_summary[missing_summary > 0]
            for col, count in missing_summary.items():
                st.write(f"• {col}: {count} missing ({count/len(df)*100:.1f}%)")
        else:
            st.success("✅ No missing values found!")

    with col2:
        st.markdown("**🔄 Data Type Conversion**")

        # Date column conversion
        date_conversion = st.checkbox("🗓️ Convert date columns")
        date_columns = []
        if date_conversion:
            date_columns = st.multiselect(
                "Select columns to convert to datetime:",
                [col for col in df.columns if df[col].dtype == 'object'],
                help="Select columns that contain date/time information"
            )

        # Numeric column conversion
        numeric_conversion = st.checkbox("🔢 Convert numeric columns")
        numeric_columns = []
        if numeric_conversion:
            numeric_columns = st.multiselect(
                "Select columns to convert to numeric:",
                [col for col in df.columns if df[col].dtype == 'object'],
                help="Select columns that contain numeric data stored as text"
            )

    # Additional cleaning options
    st.markdown("**🔧 Additional Operations**")
    col3, col4 = st.columns(2)

    with col3:
        remove_duplicates = st.checkbox("🗑️ Remove duplicate rows")
        if remove_duplicates:
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                st.info(f"Found {dup_count} duplicate rows")
            else:
                st.success("No duplicate rows found")

    with col4:
        standardize_text = st.checkbox("📝 Standardize text columns")
        if standardize_text:
            text_operation = st.selectbox(
                "Text standardization:",
                ["lowercase", "uppercase", "title case", "trim whitespace"]
            )

    # Apply cleaning button
    if st.button("🚀 Apply All Cleaning Operations", type="primary", use_container_width=True):
        try:
            original_shape = df.shape

            with st.spinner("🔄 Processing your data..."):

                # Handle missing values
                if missing_action == "Drop rows with missing values":
                    df = df.dropna()
                    st.success(f"✅ Removed {original_shape[0] - df.shape[0]} rows with missing values")

                elif missing_action == "Fill with mean/mode":
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            df[col].fillna(df[col].mean(), inplace=True)
                        else:
                            mode_val = df[col].mode()
                            df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown', inplace=True)
                    st.success("✅ Filled missing values with mean/mode")

                elif missing_action == "Fill with median":
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            df[col].fillna(df[col].median(), inplace=True)
                        else:
                            mode_val = df[col].mode()
                            df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown', inplace=True)
                    st.success("✅ Filled missing values with median/mode")

                elif missing_action == "Fill with custom value":
                    df.fillna(fill_value, inplace=True)
                    st.success(f"✅ Filled missing values with '{fill_value}'")

                elif missing_action == "Forward fill":
                    df.fillna(method='ffill', inplace=True)
                    st.success("✅ Applied forward fill to missing values")

                elif missing_action == "Backward fill":
                    df.fillna(method='bfill', inplace=True)
                    st.success("✅ Applied backward fill to missing values")

                # Convert data types
                if date_columns:
                    for col in date_columns:
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            st.success(f"✅ Converted {col} to datetime format")
                        except Exception as e:
                            st.error(f"❌ Could not convert {col} to datetime: {str(e)}")

                if numeric_columns:
                    for col in numeric_columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            st.success(f"✅ Converted {col} to numeric format")
                        except Exception as e:
                            st.error(f"❌ Could not convert {col} to numeric: {str(e)}")

                # Remove duplicates
                if remove_duplicates:
                    before_dup = len(df)
                    df = df.drop_duplicates()
                    after_dup = len(df)
                    if before_dup > after_dup:
                        st.success(f"✅ Removed {before_dup - after_dup} duplicate rows")

                # Standardize text
                if standardize_text:
                    text_cols = df.select_dtypes(include=['object']).columns
                    for col in text_cols:
                        if text_operation == "lowercase":
                            df[col] = df[col].astype(str).str.lower()
                        elif text_operation == "uppercase":
                            df[col] = df[col].astype(str).str.upper()
                        elif text_operation == "title case":
                            df[col] = df[col].astype(str).str.title()
                        elif text_operation == "trim whitespace":
                            df[col] = df[col].astype(str).str.strip()
                    st.success(f"✅ Applied {text_operation} to text columns")

                # Update session state
                st.session_state.data = df

                st.balloons()
                st.success("🎉 Data cleaning completed successfully!")

        except Exception as e:
            st.error(f"❌ Error during cleaning: {str(e)}")

    # Show data summary
    st.subheader("📊 Current Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("📋 Columns", df.shape[1])
    with col3:
        st.metric("❌ Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("🗑️ Duplicates", df.duplicated().sum())

    # Preview cleaned data
    st.subheader("👁️ Data Preview")
    st.dataframe(df.head(10), use_container_width=True, height=300)

def show_visualization():
    st.header("📈 Interactive Data Visualization")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload a CSV file first!")
        return

    df = st.session_state.data

    st.markdown("Create stunning interactive visualizations from your data")

    # Chart type selection
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("🎨 Chart Configuration")

        chart_type = st.selectbox(
            "Select Visualization Type:",
            ["📊 Bar Chart", "📈 Line Chart", "🎯 Scatter Plot", "🥧 Pie Chart", "📊 Histogram", "📦 Box Plot"],
            help="Choose the type of visualization that best represents your data"
        )

        # Get column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        all_cols = df.columns.tolist()

        # Column selection based on chart type
        if chart_type in ["📊 Bar Chart", "🥧 Pie Chart"]:
            if categorical_cols:
                x_col = st.selectbox("🎯 Category Column (X-axis):", categorical_cols)
                if chart_type == "📊 Bar Chart" and numeric_cols:
                    y_col = st.selectbox("📊 Value Column (Y-axis):", numeric_cols + [None])
                else:
                    y_col = None
            else:
                st.error("❌ No categorical columns found for this chart type")
                return

        elif chart_type in ["📈 Line Chart", "🎯 Scatter Plot"]:
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("➡️ X-axis Column:", numeric_cols + datetime_cols)
                y_col = st.selectbox("⬆️ Y-axis Column:", [col for col in numeric_cols if col != x_col])

                if chart_type == "🎯 Scatter Plot":
                    color_col = st.selectbox("🎨 Color by (optional):", [None] + categorical_cols)
                    size_col = st.selectbox("📏 Size by (optional):", [None] + numeric_cols)
            else:
                st.error("❌ Need at least 2 numeric columns for this chart type")
                return

        elif chart_type in ["📊 Histogram", "📦 Box Plot"]:
            if numeric_cols:
                x_col = st.selectbox("📊 Column to Analyze:", numeric_cols)
                y_col = None

                if chart_type == "📦 Box Plot":
                    group_col = st.selectbox("🎨 Group by (optional):", [None] + categorical_cols)
            else:
                st.error("❌ No numeric columns found for this chart type")
                return

        # Chart customization
        st.subheader("⚙️ Customization")
        chart_title = st.text_input("📝 Chart Title:", value=f"{chart_type} - {x_col}")
        chart_height = st.slider("📏 Chart Height:", 300, 800, 500)

        # Color scheme
        color_scheme = st.selectbox(
            "🎨 Color Scheme:",
            ["plotly", "viridis", "plasma", "inferno", "magma", "cividis"]
        )

    with col2:
        st.subheader(f"📊 {chart_title}")

        try:
            # Create visualizations
            if chart_type == "📊 Bar Chart" and x_col:
                if y_col:
                    grouped_data = df.groupby(x_col)[y_col].agg(['sum', 'mean', 'count']).reset_index()
                    agg_type = st.selectbox("Aggregation:", ["sum", "mean", "count"])
                    fig = px.bar(
                        grouped_data, 
                        x=x_col, 
                        y=agg_type, 
                        title=chart_title,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                else:
                    value_counts = df[x_col].value_counts()
                    fig = px.bar(
                        x=value_counts.index, 
                        y=value_counts.values, 
                        title=chart_title,
                        labels={'x': x_col, 'y': 'Count'}
                    )

            elif chart_type == "📈 Line Chart" and x_col and y_col:
                if x_col in datetime_cols:
                    df_sorted = df.sort_values(x_col)
                else:
                    df_sorted = df.sort_values(x_col)

                fig = px.line(
                    df_sorted, 
                    x=x_col, 
                    y=y_col, 
                    title=chart_title,
                    markers=True
                )

            elif chart_type == "🎯 Scatter Plot" and x_col and y_col:
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col if 'color_col' in locals() and color_col else None,
                    size=size_col if 'size_col' in locals() and size_col else None,
                    title=chart_title,
                    hover_data=categorical_cols[:3] if categorical_cols else None
                )

            elif chart_type == "🥧 Pie Chart" and x_col:
                value_counts = df[x_col].value_counts().head(10)  # Limit to top 10
                fig = px.pie(
                    values=value_counts.values, 
                    names=value_counts.index, 
                    title=chart_title,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )

            elif chart_type == "📊 Histogram" and x_col:
                fig = px.histogram(
                    df, 
                    x=x_col, 
                    title=chart_title,
                    nbins=st.slider("Number of bins:", 10, 100, 30),
                    color_discrete_sequence=['#1f77b4']
                )

            elif chart_type == "📦 Box Plot" and x_col:
                fig = px.box(
                    df, 
                    y=x_col, 
                    x=group_col if 'group_col' in locals() and group_col else None,
                    title=chart_title,
                    color=group_col if 'group_col' in locals() and group_col else None
                )

            # Update layout
            fig.update_layout(
                height=chart_height,
                showlegend=True,
                hovermode='closest',
                template='plotly_white'
            )

            # Display the chart
            st.plotly_chart(fig, use_container_width=True)

            # Chart statistics
            if chart_type in ["📊 Bar Chart", "📈 Line Chart", "🎯 Scatter Plot"] and y_col:
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("📊 Mean", f"{df[y_col].mean():.2f}")
                with col_stats2:
                    st.metric("📈 Max", f"{df[y_col].max():.2f}")
                with col_stats3:
                    st.metric("📉 Min", f"{df[y_col].min():.2f}")

        except Exception as e:
            st.error(f"❌ Error creating visualization: {str(e)}")
            st.info("Please check your data and column selections")

def show_correlation():
    st.header("🔗 Correlation Analysis")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload a CSV file first!")
        return

    df = st.session_state.data
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) < 2:
        st.error("❌ Need at least 2 numeric columns for correlation analysis")
        st.info("Please ensure your data has numerical columns or use the Data Cleaning section to convert text columns to numbers")
        return

    st.markdown("Discover hidden relationships and patterns in your numerical data")

    # Correlation matrix
    corr_matrix = numeric_df.corr()

    # Display correlation heatmap
    st.subheader("🔥 Correlation Heatmap")

    col1, col2 = st.columns([3, 1])

    with col2:
        # Correlation settings
        st.markdown("**⚙️ Settings**")
        annotation = st.checkbox("Show values", value=True)
        color_scale = st.selectbox(
            "Color scheme:",
            ["RdBu_r", "viridis", "plasma", "coolwarm", "seismic"]
        )

    with col1:
        # Create enhanced heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=annotation,
            aspect="auto",
            title="Correlation Matrix - Discover Data Relationships",
            color_continuous_scale=color_scale,
            zmin=-1, zmax=1
        )

        fig.update_layout(
            height=500,
            title_font_size=16
        )

        st.plotly_chart(fig, use_container_width=True)

    # Strong correlations analysis
    st.subheader("💪 Strong Correlations Discovery")

    correlation_threshold = st.slider(
        "🎯 Correlation Strength Threshold:", 
        0.1, 1.0, 0.7, 0.05,
        help="Values closer to 1.0 indicate stronger relationships"
    )

    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= correlation_threshold:
                strength = "Very Strong" if abs(corr_val) >= 0.9 else "Strong" if abs(corr_val) >= 0.7 else "Moderate"
                direction = "Positive" if corr_val > 0 else "Negative"

                strong_corr.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': round(corr_val, 3),
                    'Strength': strength,
                    'Direction': direction
                })

    if strong_corr:
        strong_corr_df = pd.DataFrame(strong_corr)
        strong_corr_df = strong_corr_df.sort_values('Correlation', key=abs, ascending=False)

        st.success(f"🎉 Found {len(strong_corr)} strong correlations!")
        st.dataframe(strong_corr_df, use_container_width=True, height=300)

        # Highlight the strongest correlation
        if len(strong_corr_df) > 0:
            strongest = strong_corr_df.iloc[0]
            st.info(
                f"🏆 **Strongest Relationship**: {strongest['Variable 1']} and {strongest['Variable 2']} "
                f"have a {strongest['Strength'].lower()} {strongest['Direction'].lower()} correlation "
                f"of {strongest['Correlation']}"
            )
    else:
        st.info(f"🔍 No correlations found above {correlation_threshold} threshold. Try lowering the threshold to discover weaker relationships.")

    # Pairwise comparison tool
    st.subheader("🎯 Pairwise Variable Analysis")

    col1, col2 = st.columns(2)

    with col1:
        var1 = st.selectbox("📊 Select First Variable:", numeric_df.columns)
    with col2:
        var2 = st.selectbox("📊 Select Second Variable:", [col for col in numeric_df.columns if col != var1])

    if var1 and var2:
        correlation = numeric_df[var1].corr(numeric_df[var2])

        # Correlation strength interpretation
        if abs(correlation) >= 0.9:
            strength_desc = "Very Strong"
            strength_color = "🟢"
        elif abs(correlation) >= 0.7:
            strength_desc = "Strong"
            strength_color = "🟡"
        elif abs(correlation) >= 0.5:
            strength_desc = "Moderate"
            strength_color = "🟠"
        elif abs(correlation) >= 0.3:
            strength_desc = "Weak"
            strength_color = "🔴"
        else:
            strength_desc = "Very Weak"
            strength_color = "⚫"

        direction = "Positive" if correlation > 0 else "Negative"

        st.metric(
            f"🔗 Correlation between {var1} and {var2}", 
            f"{correlation:.3f}",
            delta=f"{strength_color} {strength_desc} {direction}"
        )

        # Scatter plot with trend line
        fig = px.scatter(
            df, 
            x=var1, 
            y=var2, 
            title=f"{var1} vs {var2} (Correlation: {correlation:.3f})",
            trendline="ols",
            hover_data=df.select_dtypes(include=['object']).columns[:2].tolist()
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Statistical insights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"📊 {var1} Mean", f"{df[var1].mean():.2f}")
        with col2:
            st.metric(f"📊 {var2} Mean", f"{df[var2].mean():.2f}")
        with col3:
            st.metric("📊 R² Score", f"{correlation**2:.3f}")

def show_statistics():
    st.header("📊 Comprehensive Statistical Analysis")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload a CSV file first!")
        return

    df = st.session_state.data

    st.markdown("Deep dive into your data with professional statistical insights")

    # Overview metrics
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("📊 Total Rows", f"{len(df):,}")
    with col2:
        st.metric("📋 Total Columns", len(df.columns))
    with col3:
        st.metric("💾 Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    with col4:
        st.metric("🎯 Data Completeness", f"{((df.size - df.isnull().sum().sum()) / df.size * 100):.1f}%")
    with col5:
        st.metric("🗑️ Duplicate Rows", f"{df.duplicated().sum():,}")

    # Descriptive statistics
    st.subheader("📈 Descriptive Statistics")

    # Separate numeric and non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(include=['object', 'category'])

    if not numeric_df.empty:
        st.markdown("**🔢 Numerical Columns:**")

        # Enhanced descriptive statistics
        desc_stats = numeric_df.describe().round(2)

        # Add additional statistics
        additional_stats = pd.DataFrame({
            col: {
                'variance': numeric_df[col].var(),
                'skewness': numeric_df[col].skew(),
                'kurtosis': numeric_df[col].kurtosis()
            } for col in numeric_df.columns
        }).round(3)

        # Combine statistics
        full_stats = pd.concat([desc_stats, additional_stats])
        st.dataframe(full_stats, use_container_width=True, height=400)

        # Statistical insights
        st.markdown("**🎯 Key Insights:**")
        for col in numeric_df.columns:
            skew = numeric_df[col].skew()
            if abs(skew) > 1:
                skew_desc = "highly skewed"
            elif abs(skew) > 0.5:
                skew_desc = "moderately skewed"
            else:
                skew_desc = "approximately normal"

            st.write(f"• **{col}**: {skew_desc} (skewness: {skew:.2f})")

    if not categorical_df.empty:
        st.markdown("**📝 Categorical Columns:**")
        cat_stats = pd.DataFrame({
            'Column': categorical_df.columns,
            'Unique Values': [categorical_df[col].nunique() for col in categorical_df.columns],
            'Most Frequent': [categorical_df[col].mode().iloc[0] if not categorical_df[col].mode().empty else 'N/A' 
                            for col in categorical_df.columns],
            'Frequency': [categorical_df[col].value_counts().iloc[0] if len(categorical_df[col].value_counts()) > 0 else 0 
                         for col in categorical_df.columns]
        })
        st.dataframe(cat_stats, use_container_width=True)

    # Data quality analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Data Types Distribution")
        dtype_counts = df.dtypes.value_counts()
        fig_dtype = px.pie(
            values=dtype_counts.values,
            names=[str(dtype) for dtype in dtype_counts.index],
            title="Distribution of Data Types"
        )
        st.plotly_chart(fig_dtype, use_container_width=True)

    with col2:
        st.subheader("❌ Missing Values Analysis")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

        if not missing_df.empty:
            fig_missing = px.bar(
                missing_df,
                x='Column',
                y='Missing Percentage',
                title="Missing Values by Column (%)"
            )
            fig_missing.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("🎉 No missing values found in your dataset!")

    # Detailed column analysis
    st.subheader("🔍 Detailed Column Analysis")

    column_analysis = []
    for col in df.columns:
        col_info = {
            'Column': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Null %': round(df[col].isnull().sum() / len(df) * 100, 2),
            'Unique Values': df[col].nunique(),
            'Memory Usage (KB)': round(df[col].memory_usage(deep=True) / 1024, 2)
        }

        if df[col].dtype in ['int64', 'float64']:
            col_info.update({
                'Min': df[col].min(),
                'Max': df[col].max(),
                'Mean': round(df[col].mean(), 2),
                'Std': round(df[col].std(), 2)
            })
        else:
            col_info.update({
                'Most Frequent': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                'Most Frequent Count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            })

        column_analysis.append(col_info)

    analysis_df = pd.DataFrame(column_analysis)
    st.dataframe(analysis_df, use_container_width=True, height=400)

    # Export options
    st.subheader("💾 Export Your Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Download Statistical Summary", use_container_width=True):
            if not numeric_df.empty:
                csv = full_stats.to_csv()
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name="statistical_summary.csv",
                    mime="text/csv"
                )

    with col2:
        if st.button("📋 Download Column Analysis", use_container_width=True):
            csv = analysis_df.to_csv(index=False)
            st.download_button(
                label="📋 Download CSV",
                data=csv,
                file_name="column_analysis.csv",
                mime="text/csv"
            )

    with col3:
        if st.button("🔧 Download Cleaned Data", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="🔧 Download CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()


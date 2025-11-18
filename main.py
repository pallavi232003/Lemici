# ====================================================
# Streamlit + Cohere AI Data Assistant with Dashboards
# ====================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cohere
import fitz  # PyMuPDF
import plotly.express as px
import sqlite3

# -------------------------------
# Setup Cohere
# -------------------------------
COHERE_API_KEY = "hU3sWTKYF6nBPPjIIXHidDx2BqyWNVl0LyLzjPzQ" 
co = cohere.Client(COHERE_API_KEY)

# -------------------------------
# Cohere Insight Function
# -------------------------------
def generate_llm_insights(data):
    if isinstance(data, pd.DataFrame):
        sample_data = data.head(10).to_string()
        prompt = f"""
You are a business data analyst. Analyze this dataset and provide executive-level insights.
Dataset Sample:
{sample_data}

Please provide:
1. Business summary of the dataset
2. Key trends
3. Potential anomalies
4. Suggestions for decision-making
"""
    elif isinstance(data, str):
        prompt = f"""
You are a data analyst. Analyze this unstructured text data and provide insights.
Data:
{data}

Please provide a summary and any key insights or themes you find.
"""
    else:
        return "Unsupported data type for analysis."

    response = co.chat(
        model="command-r-plus-08-2024",
        message=prompt,
        max_tokens=800,
        temperature=0.7
    )
    return response.text

# -------------------------------
# Text-to-SQL Function
# -------------------------------
def natural_language_to_sql(question, df):
    schema_description = "\n".join([f"{col} ({str(df[col].dtype)})" for col in df.columns])
    sample_data = df.head(10).to_string(index=False)

    prompt = f"""
You are an expert SQL generator for SQLite.
Your task is to convert natural language questions into VALID SQL queries
for the dataset provided.

Dataset Schema:
{schema_description}

Sample Data:
{sample_data}

Rules:
- Table name is 'data'
- Use ONLY the columns provided
- Return ONLY raw SQL (no markdown, no explanation, no comments)
- Must use SQLite syntax (use LIMIT not TOP)
- If aggregation is needed, include GROUP BY

Question: {question}

SQL Query:
"""

    response = co.chat(
        model="command-r-plus-08-2024",
        message=prompt,
        temperature=0
    )

    sql_query = response.text.strip()
    if "```" in sql_query:
        sql_query = sql_query.split("```")[1]
    if sql_query.lower().startswith("sql"):
        sql_query = sql_query[3:].strip()

    return sql_query

# -------------------------------
# Execute SQL on Pandas via SQLite
# -------------------------------
def run_sql_on_df(df, sql_query):
    conn = sqlite3.connect(":memory:")
    df.to_sql("data", conn, index=False, if_exists="replace")
    try:
        result = pd.read_sql_query(sql_query, conn)
    except Exception as e:
        result = pd.DataFrame({"Error": [str(e)]})
    conn.close()
    return result

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title=" AI Business Data Assistant", layout="wide")
st.title(" AI-Powered Business Data Assistant")
st.write("Upload a dataset (CSV, Excel, PDF, or TXT) to explore insights, dashboards, and automation.")

uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls", "txt", "pdf"])

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()

    if ext in ["csv", "xlsx", "xls"]:
        df = pd.read_csv(uploaded_file) if ext == "csv" else pd.read_excel(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # =====================
        # KPI Cards
        # =====================
        st.subheader(" Key Performance Indicators (KPIs)")
        numeric_cols = df.select_dtypes(include="number").columns
        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.metric("Total Records", len(df))
        if len(numeric_cols) > 0:
            with kpi2:
                st.metric(f"Mean of {numeric_cols[0]}", round(df[numeric_cols[0]].mean(), 2))
            with kpi3:
                st.metric(f"Max of {numeric_cols[0]}", round(df[numeric_cols[0]].max(), 2))

        # =====================
        # Category Breakdown
        # =====================
        cat_cols = df.select_dtypes(include="object").columns
        if len(cat_cols) > 0:
            st.subheader(" Category Breakdown")
            col_choice = st.selectbox("Select a categorical column", cat_cols)

            cat_counts = df[col_choice].value_counts().reset_index()
            cat_counts.columns = [col_choice, "count"]

            chart_type = st.radio("Choose chart type", ["Bar", "Pie"], horizontal=True)

            if chart_type == "Bar":
                fig = px.bar(cat_counts, x=col_choice, y="count", title=f"Count of {col_choice}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.pie(cat_counts, names=col_choice, values="count", title=f"Distribution of {col_choice}")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("🛠 View Chart Code"):
                st.code("""
cat_counts = df[col_choice].value_counts().reset_index()
cat_counts.columns = [col_choice, "count"]
fig = px.bar(cat_counts, x=col_choice, y="count")
                """, language="python")

        # =====================
        # Correlation Heatmap
        # =====================
        if len(numeric_cols) > 1:
            st.subheader(" Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            with st.expander("🛠 View Heatmap Code"):
                st.code("""
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
                """, language="python")

        # =====================
        # AI Insights
        # =====================
        st.subheader("AI Insights from Cohere")
        with st.spinner("Analyzing dataset..."):
            insights = generate_llm_insights(df)
            st.write(insights)

        # =====================
        # Text-to-SQL Search
        # =====================
        st.subheader(" Ask Questions about your Data")
        user_question = st.text_input("Type a question (e.g., 'What is the average sales by region?')")

        if user_question:
            sql_query = natural_language_to_sql(user_question, df)
            result = run_sql_on_df(df, sql_query)

            if "Error" in result.columns:
                st.error(f" SQL Execution Failed: {result['Error'][0]}")
            else:
                st.write(" Query Results")
                st.dataframe(result)

                # Try auto-visualization
                if result.shape[1] >= 2:
                    try:
                        fig = px.bar(result, x=result.columns[0], y=result.columns[1], title="Query Visualization")
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.write(" Could not generate visualization for this query.")

                # Expandable backend code
                with st.expander(" View Backend SQL Code"):
                    st.code(sql_query, language="sql")

    elif ext == "txt":
        text_data = uploaded_file.read().decode("utf-8")
        st.subheader("Text Data Preview")
        st.text(text_data[:500])

        st.subheader(" AI Insights from Cohere")
        insights = generate_llm_insights(text_data)
        st.write(insights)

    elif ext == "pdf":
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text_data = ""
        for page in pdf_document:
            text_data += page.get_text()
        pdf_document.close()

        st.subheader("PDF Preview")
        st.text(text_data[:500])

        st.subheader(" AI Insights from Cohere")
        insights = generate_llm_insights(text_data)
        st.write(insights)

    else:
        st.error("Unsupported file type")


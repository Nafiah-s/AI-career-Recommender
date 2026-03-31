import streamlit as st
import pandas as pd
from model import load_data, recommend_careers, get_unique_skills

st.set_page_config(
    page_title="AI Career Recommendation System",
    page_icon="🎯",
    layout="wide"
)

# Load data
df = load_data()

st.title("🎯 AI Career Recommendation System")
st.write("Get personalized career recommendations based on your skills and interests.")

st.sidebar.header("User Profile Input")

all_skills = get_unique_skills(df)

selected_skills = st.sidebar.multiselect(
    "Select Your Skills",
    all_skills
)

interests = st.sidebar.text_input(
    "Enter Your Interests",
    placeholder="Example: data analysis, business, coding, design"
)

education = st.sidebar.selectbox(
    "Education Level",
    ["High School", "Diploma", "Undergraduate", "Postgraduate"]
)

preferred_domain = st.sidebar.selectbox(
    "Preferred Domain",
    ["Any", "Technology", "Business", "Design", "Marketing", "Finance", "Analytics"]
)

submit = st.sidebar.button("Get Recommendation")

st.markdown("---")

if submit:
    if not selected_skills and not interests.strip():
        st.warning("Please provide at least skills or interests.")
    else:
        recommendations = recommend_careers(
            df=df,
            user_skills=selected_skills,
            user_interests=interests,
            education=education,
            preferred_domain=preferred_domain
        )

        if recommendations.empty:
            st.error("No matching careers found. Try adding more skills or interests.")
        else:
            st.subheader("✅ Recommended Career Paths")

            for index, row in recommendations.iterrows():
                with st.container():
                    st.markdown(f"## {row['Career Path']}")
                    st.progress(min(int(row['Match Score']), 100))
                    st.write(f"**Match Score:** {row['Match Score']}%")
                    st.write(f"**Domain:** {row['Domain']}")
                    st.write(f"**Suggested Job Roles:** {row['Job Roles']}")
                    st.write(f"**Recommended Courses:** {row['Courses']}")
                    st.write(f"**Important Skills Needed:** {row['Skills']}")
                    st.write(f"**Missing Skills for You:** {row['Missing Skills']}")
                    st.markdown("---")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.info("### How it works\n- Select your skills\n- Add your interests\n- Choose education and domain\n- Get top career recommendations")

    with col2:
        st.success("### Output includes\n- Career Path\n- Job Roles\n- Recommended Courses\n- Missing Skills")

st.markdown("### 📊 Career Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)
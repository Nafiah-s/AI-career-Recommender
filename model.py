import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    df = pd.read_csv("careers.csv")

    # Fill missing values safely
    for col in ["Skills", "Interests", "Domain", "Job Roles", "Courses", "Career Path"]:
        df[col] = df[col].fillna("").astype(str)

    # Combined text for recommendation
    df["combined"] = (
        df["Skills"] + " " +
        df["Interests"] + " " +
        df["Domain"] + " " +
        df["Career Path"]
    )

    return df


def get_unique_skills(df):
    skills_set = set()

    for skills in df["Skills"]:
        split_skills = [skill.strip() for skill in skills.split(",")]
        for skill in split_skills:
            if skill:
                skills_set.add(skill)

    return sorted(skills_set)


def recommend_careers(df, user_skills, user_interests, education, preferred_domain):
    user_skills_text = " ".join(user_skills)
    user_profile = f"{user_skills_text} {user_interests}"

    # Domain filter if selected
    filtered_df = df.copy()
    if preferred_domain != "Any":
        filtered_df = filtered_df[
            filtered_df["Domain"].str.lower().str.contains(preferred_domain.lower(), na=False)
        ]

    if filtered_df.empty:
        return pd.DataFrame()

    # TF-IDF similarity
    corpus = filtered_df["combined"].tolist() + [user_profile]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    user_vector = tfidf_matrix[-1]
    career_vectors = tfidf_matrix[:-1]

    similarity_scores = cosine_similarity(user_vector, career_vectors).flatten()

    filtered_df = filtered_df.copy()
    filtered_df["Similarity"] = similarity_scores

    # Additional skill match scoring
    def skill_match_score(career_skills):
        career_skill_list = [s.strip().lower() for s in career_skills.split(",")]
        user_skill_list = [s.strip().lower() for s in user_skills]

        if not career_skill_list or not user_skill_list:
            return 0

        matched = set(career_skill_list).intersection(set(user_skill_list))
        return len(matched) / max(len(career_skill_list), 1)

    filtered_df["Skill Match"] = filtered_df["Skills"].apply(skill_match_score)

    # Final score
    filtered_df["Final Score"] = (
        filtered_df["Similarity"] * 0.6 +
        filtered_df["Skill Match"] * 0.4
    )

    filtered_df["Match Score"] = (filtered_df["Final Score"] * 100).round(2)

    # Missing skills
    def missing_skills(career_skills):
        career_skill_list = [s.strip() for s in career_skills.split(",") if s.strip()]
        user_skill_list = [s.strip().lower() for s in user_skills]

        missing = [skill for skill in career_skill_list if skill.lower() not in user_skill_list]
        return ", ".join(missing) if missing else "None"

    filtered_df["Missing Skills"] = filtered_df["Skills"].apply(missing_skills)

    recommendations = filtered_df.sort_values(by="Final Score", ascending=False).head(5)

    return recommendations[[
        "Career Path",
        "Domain",
        "Job Roles",
        "Courses",
        "Skills",
        "Missing Skills",
        "Match Score"
    ]]
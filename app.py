import os
import re
import json
import uuid
from datetime import datetime

import requests
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import fitz
from dotenv import load_dotenv
from fpdf import FPDF


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv(
    "HF_MODEL",
    "Qwen/Qwen2.5-7B-Instruct"
)

HF_URL = "https://router.huggingface.co/v1/chat/completions"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Mock Interviewer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# DARK PRODUCTION UI
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #0b0f19;
        color: #f5f7fb;
    }

    .main {
        background: #0b0f19;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }

    section[data-testid="stSidebar"] * {
        color: #f5f7fb !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    p, label, span, div {
        color: inherit;
    }

    div[data-baseweb="input"] {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="input"] input {
        color: #ffffff !important;
        background-color: transparent !important;
    }

    textarea {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #374151 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        color: #ffffff !important;
        border-color: #374151 !important;
    }

    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }

    div[data-baseweb="select"] input {
        color: #ffffff !important;
    }

    div[role="listbox"] {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #374151 !important;
    }

    div[role="option"] {
        background-color: #111827 !important;
        color: #ffffff !important;
    }

    div[role="option"] * {
        color: #ffffff !important;
    }

    div[role="option"]:hover {
        background-color: #1f2937 !important;
        color: #ffffff !important;
    }

    div[role="option"][aria-selected="true"] {
        background-color: #374151 !important;
        color: #ffffff !important;
    }

    .stButton > button {
        background: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1rem !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        background: #1d4ed8 !important;
        color: #ffffff !important;
    }

    div[data-testid="stFileUploader"] {
        background: #111827 !important;
        border: 1px dashed #374151 !important;
        border-radius: 10px !important;
    }

    div[data-testid="stFileUploader"] * {
        color: #ffffff !important;
    }

    .card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
    }

    .question-card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 25px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px !important;
    }

    div[data-testid="stProgressBar"] > div > div {
        background-color: #2563eb !important;
    }

    div[data-testid="stRadio"] label,
    div[data-testid="stCheckbox"] label {
        color: #ffffff !important;
    }

    .stDownloadButton > button {
        background: #059669 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stDownloadButton > button:hover {
        background: #047857 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# ROLES / LEVELS
# =========================================================

ROLES = [
    "Software Engineer",
    "Data Scientist",
    "Data Analyst",
    "Product Manager",
    "Frontend Developer",
    "Backend Developer",
    "DevOps Engineer",
    "Machine Learning Engineer",
    "Business Analyst",
    "Full Stack Developer",
    "UI/UX Designer",
    "Cloud Architect"
]

LEVELS = [
    "Intern",
    "Junior (0-2 yrs)",
    "Mid-Level (2-5 yrs)",
    "Senior (5+ yrs)",
    "Lead / Manager"
]


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are an expert senior technical interviewer at a top tech company.

Your job:
1. Ask highly relevant interview questions based on role, level, resume AND job description
2. Reference specific resume details when asking
3. When JD is provided, align questions to required skills in the JD
4. Evaluate answers critically and fairly
5. Score on:
   - Technical Accuracy
   - Clarity
   - Communication
6. Scores must be from 0 to 10.

ALWAYS respond in this EXACT JSON format.
Do not use markdown fences.
Do not add any text outside JSON.

{
  "response_type": "question",
  "message": "Your question or response",
  "followup_question": null,
  "scores": {
    "technical_accuracy": null,
    "clarity": null,
    "communication": null
  },
  "feedback": null,
  "strong_points": [],
  "weak_points": [],
  "ideal_answer": null
}

For evaluations:
- fill scores
- fill feedback
- fill strong_points
- fill weak_points
- fill ideal_answer

For questions:
- keep scores null
- keep feedback null
- strong_points should be []
- weak_points should be []
- ideal_answer should be null

followup_question:
Set ONLY if the answer is weak, incomplete, unclear, or needs deeper explanation.
"""


# =========================================================
# SESSION STATE
# =========================================================

def initialize_session():

    defaults = {
        "phase": "setup",
        "candidate_name": "",
        "role": "",
        "level": "",
        "resume_text": "",
        "jd_text": "",
        "skill_gap": None,
        "questions": [],
        "evaluations": [],
        "current_question": "",
        "current_q_num": 0,
        "answer_text": "",
        "started_at": None,
        "finished_at": None,
        "final_report": "",
        "report_pdf": None,
        "certificate_pdf": None,
        "session_id": str(uuid.uuid4())
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialize_session()


# =========================================================
# HELPER - SAFE STRING
# =========================================================

def clean_string(value):

    if value is None:
        return ""

    return str(value).strip()


# =========================================================
# HUGGING FACE API
# =========================================================

def call_huggingface(
    messages,
    temperature=0.4,
    max_tokens=1200
):

    if not HF_TOKEN:

        raise RuntimeError(
            "HF_TOKEN is missing. Please add HF_TOKEN to your .env file."
        )

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(
        HF_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"Model error: {response.status_code} - {response.text}"
        )

    data = response.json()

    try:

        return data["choices"][0]["message"]["content"]

    except Exception:

        raise RuntimeError(
            f"Unexpected Hugging Face response: {data}"
        )


# =========================================================
# JSON PARSER
# =========================================================

def parse_json_response(text):

    if not text:
        return {}

    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:
        return json.loads(text)

    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        json_text = text[start:end + 1]

        try:
            return json.loads(json_text)

        except Exception:
            pass

    return {
        "response_type": "question",
        "message": text,
        "followup_question": None,
        "scores": {
            "technical_accuracy": None,
            "clarity": None,
            "communication": None
        },
        "feedback": None,
        "strong_points": [],
        "weak_points": [],
        "ideal_answer": None
    }


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    if uploaded_file is None:
        return ""

    try:

        pdf_bytes = uploaded_file.read()

        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        pages = []

        for page in document:
            pages.append(page.get_text())

        document.close()

        return "\n".join(pages).strip()

    except Exception as e:

        st.error(
            f"Could not read PDF: {e}"
        )

        return ""


# =========================================================
# CANDIDATE CONTEXT
# =========================================================

def build_candidate_context():

    return f"""
Candidate Name:
{st.session_state.candidate_name}

Role:
{st.session_state.role}

Experience Level:
{st.session_state.level}

RESUME:
{st.session_state.resume_text[:12000]}

JOB DESCRIPTION:
{st.session_state.jd_text[:12000]}
"""


# =========================================================
# GENERATE QUESTION
# =========================================================

def generate_question():

    context = build_candidate_context()

    question_number = (
        len(st.session_state.questions) + 1
    )

    previous_questions = "\n".join(
        [
            f"{i + 1}. {q}"
            for i, q in enumerate(
                st.session_state.questions
            )
        ]
    )

    user_prompt = f"""
You are conducting a professional mock interview.

{context}

This is question number {question_number}.

Previous questions:
{previous_questions}

Generate the next interview question.

Requirements:
- Make it relevant to the selected role.
- Match the candidate's experience level.
- Use resume details when possible.
- Use JD requirements when available.
- Avoid repeating previous questions.
- Make it realistic for an actual interview.
- Return ONLY the required JSON format.
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    raw = call_huggingface(
        messages,
        temperature=0.5,
        max_tokens=900
    )

    result = parse_json_response(raw)

    question = clean_string(
        result.get("message")
    )

    if not question:

        question = (
            "Please describe your approach to solving "
            "a challenging problem in your field."
        )

    return result, question


# =========================================================
# EVALUATE ANSWER
# =========================================================

def evaluate_answer(
    question,
    answer
):

    context = build_candidate_context()

    user_prompt = f"""
Evaluate this interview answer.

{context}

QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

Evaluate strictly but fairly.

Give:
1. Technical Accuracy score from 0-10
2. Clarity score from 0-10
3. Communication score from 0-10
4. Specific feedback
5. Strong points
6. Weak points
7. Ideal answer
8. Follow-up question only if the answer is weak or incomplete

Return ONLY JSON using the required format.
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    raw = call_huggingface(
        messages,
        temperature=0.3,
        max_tokens=1500
    )

    return parse_json_response(raw)


# =========================================================
# SKILL GAP ANALYSIS
# =========================================================

def generate_skill_gap():

    context = build_candidate_context()

    user_prompt = f"""
Analyze the candidate's skill gap for the selected role.

{context}

Return ONLY JSON in this structure:

{{
    "required_skills": [],
    "matched_skills": [],
    "missing_skills": [],
    "recommendations": []
}}
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert technical career analyst. "
                "Return valid JSON only."
            )
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    raw = call_huggingface(
        messages,
        temperature=0.3,
        max_tokens=1200
    )

    result = parse_json_response(raw)

    if not isinstance(result, dict):
        result = {}

    return result


# =========================================================
# FINAL REPORT
# =========================================================

def generate_final_report():

    evaluations = st.session_state.evaluations

    evaluation_text = ""

    for i, item in enumerate(evaluations):

        evaluation_text += f"""
Question {i + 1}:
{item.get("question", "")}

Candidate Answer:
{item.get("answer", "")}

Technical Accuracy:
{item.get("technical_accuracy", 0)}

Clarity:
{item.get("clarity", 0)}

Communication:
{item.get("communication", 0)}

Feedback:
{item.get("feedback", "")}

Strong Points:
{", ".join(item.get("strong_points", []))}

Weak Points:
{", ".join(item.get("weak_points", []))}

Ideal Answer:
{item.get("ideal_answer", "")}

--------------------------------
"""

    user_prompt = f"""
Create a professional final interview performance report.

Candidate:
{st.session_state.candidate_name}

Role:
{st.session_state.role}

Level:
{st.session_state.level}

Interview evaluations:
{evaluation_text}

The report must include:

1. Overall performance
2. Technical performance
3. Clarity performance
4. Communication performance
5. Strong points
6. Weak areas
7. Skill gap observations
8. Recommended learning areas
9. Final interview readiness summary

Keep it professional and useful.

Return plain text.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior technical interview evaluator."
            )
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    return call_huggingface(
        messages,
        temperature=0.4,
        max_tokens=2000
    )


# =========================================================
# SCORE COLOR
# =========================================================

def score_color(score):

    try:
        score = float(score)
    except Exception:
        score = 0

    if score >= 8:
        return "🟢"

    if score >= 5:
        return "🟡"

    return "🔴"


# =========================================================
# SKILL GAP RADAR
# =========================================================

def show_skill_gap_radar(skill_gap):

    required = skill_gap.get(
        "required_skills",
        []
    )

    if not required:

        required = [
            "Python",
            "Data Analysis",
            "Machine Learning",
            "Communication",
            "Problem Solving"
        ]

    matched = skill_gap.get(
        "matched_skills",
        []
    )

    values = []

    for skill in required:

        if skill in matched:
            values.append(100)
        else:
            values.append(25)

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=required,
            fill="toself",
            name="Skill Coverage"
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            ),
            bgcolor="#111827"
        ),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        margin=dict(
            l=40,
            r=40,
            t=40,
            b=40
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PERFORMANCE RADAR
# =========================================================

def show_performance_radar(evaluations):

    if not evaluations:
        return

    technical = [
        float(
            x.get(
                "technical_accuracy",
                0
            ) or 0
        )
        for x in evaluations
    ]

    clarity = [
        float(
            x.get(
                "clarity",
                0
            ) or 0
        )
        for x in evaluations
    ]

    communication = [
        float(
            x.get(
                "communication",
                0
            ) or 0
        )
        for x in evaluations
    ]

    values = [
        sum(technical) / len(technical),
        sum(clarity) / len(clarity),
        sum(communication) / len(communication)
    ]

    labels = [
        "Technical Accuracy",
        "Clarity",
        "Communication"
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            name="Performance"
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            ),
            bgcolor="#111827"
        ),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        margin=dict(
            l=40,
            r=40,
            t=40,
            b=40
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# NEW PERFORMANCE DASHBOARD
# =========================================================

def show_performance_dashboard(evaluations):

    if not evaluations:
        return

    technical_scores = [
        float(x.get("technical_accuracy", 0) or 0)
        for x in evaluations
    ]

    clarity_scores = [
        float(x.get("clarity", 0) or 0)
        for x in evaluations
    ]

    communication_scores = [
        float(x.get("communication", 0) or 0)
        for x in evaluations
    ]

    technical_avg = sum(technical_scores) / len(technical_scores)
    clarity_avg = sum(clarity_scores) / len(clarity_scores)
    communication_avg = sum(communication_scores) / len(communication_scores)

    scores = {
        "Technical Accuracy": technical_avg,
        "Answer Clarity": clarity_avg,
        "Communication": communication_avg
    }

    overall = (
        technical_avg + clarity_avg + communication_avg
    ) / 3

    strongest_skill = max(scores, key=scores.get)
    weakest_skill = min(scores, key=scores.get)
    questions_evaluated = len(evaluations)

    if overall >= 8:
        status = "Excellent Performance"
        status_icon = "🟢"
    elif overall >= 6:
        status = "Good Performance"
        status_icon = "🟡"
    else:
        status = "Needs Improvement"
        status_icon = "🔴"

    # =====================================================
    # OVERALL PERFORMANCE
    # =====================================================

    st.markdown("## 🏆 Performance Dashboard")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Overall Score", f"{overall:.1f}/10")

    with m2:
        st.metric("Performance Status", f"{status_icon} {status}")

    with m3:
        st.metric("Questions Evaluated", str(questions_evaluated))

    # =====================================================
    # SKILL PERFORMANCE
    # =====================================================

    st.markdown("### 📊 Skill Performance")

    skill_labels = {
        "Technical Accuracy": "Technical Accuracy",
        "Answer Clarity": "Answer Clarity",
        "Communication": "Communication"
    }

    for skill, score in scores.items():

        if score >= 8:
            icon = "🟢"
            label = "Excellent"
        elif score >= 6:
            icon = "🟡"
            label = "Good"
        else:
            icon = "🔴"
            label = "Needs Improvement"

        st.markdown(
            f"**{skill_labels[skill]} — {score:.1f}/10 · {label}**"
        )

        st.progress(
            max(0.0, min(score / 10.0, 1.0))
        )

    # =====================================================
    # PERFORMANCE INSIGHTS
    # =====================================================

    st.markdown("### 🎯 Performance Insights")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**💪 Strongest Skill**")
        st.markdown(f"### {strongest_skill}")
        st.write(f"Score: {scores[strongest_skill]:.1f}/10")

    with c2:
        st.markdown("**🎯 Needs Improvement**")
        st.markdown(f"### {weakest_skill}")
        st.write(f"Score: {scores[weakest_skill]:.1f}/10")

    # =====================================================
    # QUESTION-WISE PERFORMANCE
    # =====================================================

    st.markdown("### 📋 Question-wise Performance")

    for i, item in enumerate(evaluations):

        technical = float(item.get("technical_accuracy", 0) or 0)
        clarity = float(item.get("clarity", 0) or 0)
        communication = float(item.get("communication", 0) or 0)

        question_score = (
            technical + clarity + communication
        ) / 3

        if question_score >= 8:
            question_status = "🟢 Excellent"
        elif question_score >= 6:
            question_status = "🟡 Good"
        else:
            question_status = "🔴 Needs Improvement"

        with st.expander(
            f"Question {i + 1}   •   {question_status}   •   {question_score:.1f}/10"
        ):

            q1, q2, q3 = st.columns(3)

            q1.metric(
                "Technical Accuracy",
                f"{technical:.1f}/10"
            )

            q2.metric(
                "Answer Clarity",
                f"{clarity:.1f}/10"
            )

            q3.metric(
                "Communication",
                f"{communication:.1f}/10"
            )

            st.write("**Feedback:**")
            st.write(item.get("feedback", ""))


# =========================================================
# QUESTION BREAKDOWN
# =========================================================

def show_question_breakdown(evaluations):

    if not evaluations:
        return

    st.subheader(
        "📋 Detailed Question Review"
    )

    for i, item in enumerate(evaluations):

        with st.expander(
            f"Question {i + 1} - Detailed Review"
        ):

            st.write(
                "**Question:**"
            )

            st.write(
                item.get(
                    "question",
                    ""
                )
            )

            st.write(
                "**Your Answer:**"
            )

            st.write(
                item.get(
                    "answer",
                    ""
                )
            )

            c1, c2, c3 = st.columns(3)

            technical = float(
                item.get(
                    "technical_accuracy",
                    0
                ) or 0
            )

            clarity = float(
                item.get(
                    "clarity",
                    0
                ) or 0
            )

            communication = float(
                item.get(
                    "communication",
                    0
                ) or 0
            )

            c1.metric(
                "Technical Accuracy",
                f"{technical:.1f}/10"
            )

            c2.metric(
                "Clarity",
                f"{clarity:.1f}/10"
            )

            c3.metric(
                "Communication",
                f"{communication:.1f}/10"
            )

            st.write(
                "**Feedback:**"
            )

            st.write(
                item.get(
                    "feedback",
                    ""
                )
            )

            strong = item.get(
                "strong_points",
                []
            )

            if strong:

                st.write(
                    "**Strong Points:**"
                )

                for point in strong:

                    st.write(
                        f"• {point}"
                    )

            weak = item.get(
                "weak_points",
                []
            )

            if weak:

                st.write(
                    "**Weak Points:**"
                )

                for point in weak:

                    st.write(
                        f"• {point}"
                    )

            ideal_answer = item.get(
                "ideal_answer",
                ""
            )

            if ideal_answer:

                st.write(
                    "**Ideal Answer:**"
                )

                st.write(
                    ideal_answer
                )


# =========================================================
# PDF SAFE TEXT
# =========================================================

def pdf_safe_text(value):

    if value is None:
        return ""

    text = str(value)

    replacements = {

        "\u2013": "-",
        "\u2014": "-",

        "\u2018": "'",
        "\u2019": "'",

        "\u201c": '"',
        "\u201d": '"',

        "\u2022": "-",

        "\u2713": "-",

        "\u2192": "->",
        "\u2190": "<-",

        "\u2026": "...",

        "\u00a0": " ",

        "\u200b": "",
        "\u200c": "",
        "\u200d": "",

        "\ufeff": "",
        "\u00ad": ""
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    text = "".join(
        ch
        for ch in text
        if ch in "\n\r\t" or ord(ch) >= 32
    )

    text = (
        text
        .encode(
            "latin-1",
            "replace"
        )
        .decode(
            "latin-1"
        )
    )

    return text


# =========================================================
# PDF WRAPPED TEXT
# =========================================================

def pdf_write_wrapped(
    pdf,
    value,
    font="Helvetica",
    style="",
    size=10,
    height=6
):

    text_value = pdf_safe_text(value)

    pdf.set_font(
        font,
        style,
        size
    )

    text_value = (
        text_value
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    lines = text_value.split("\n")

    for raw_line in lines:

        line = raw_line.replace(
            "\t",
            "    "
        ).strip()

        if not line:

            pdf.ln(height)
            continue

        while len(line) > 70:

            cut = line.rfind(
                " ",
                0,
                70
            )

            if cut < 1:
                cut = 70

            chunk = line[:cut].strip()
            line = line[cut:].strip()

            if chunk:

                pdf.set_x(
                    pdf.l_margin
                )

                pdf.multi_cell(
                    pdf.epw,
                    height,
                    chunk
                )

        if line:

            pdf.set_x(
                pdf.l_margin
            )

            pdf.multi_cell(
                pdf.epw,
                height,
                line
            )


# =========================================================
# GENERATE PDF REPORT
# =========================================================

def generate_pdf_report(
    candidate_name,
    role,
    level,
    avg,
    evaluations,
    report_text
):

    pdf = FPDF()

    pdf.set_auto_page_break(
        auto=True,
        margin=15
    )

    pdf.set_margins(
        15,
        15,
        15
    )

    pdf.add_page()

    pdf_write_wrapped(
        pdf,
        "AI Mock Interview - Performance Report",
        font="Helvetica",
        style="B",
        size=18,
        height=8
    )

    pdf.ln(4)

    pdf_write_wrapped(
        pdf,
        f"Candidate: {candidate_name}",
        style="B",
        size=11
    )

    pdf_write_wrapped(
        pdf,
        f"Role: {role}",
        size=10
    )

    pdf_write_wrapped(
        pdf,
        f"Level: {level}",
        size=10
    )

    pdf_write_wrapped(
        pdf,
        f"Interview Date: {datetime.now().strftime('%d-%m-%Y')}",
        size=10
    )

    pdf_write_wrapped(
        pdf,
        f"Overall Average Score: {avg:.2f}/10",
        style="B",
        size=11
    )

    pdf.ln(5)

    pdf_write_wrapped(
        pdf,
        "AI FINAL REPORT",
        style="B",
        size=14,
        height=7
    )

    pdf.ln(2)

    pdf_write_wrapped(
        pdf,
        report_text,
        size=10,
        height=6
    )

    pdf.ln(6)

    pdf_write_wrapped(
        pdf,
        "QUESTION-WISE EVALUATION",
        style="B",
        size=14,
        height=7
    )

    for i, item in enumerate(evaluations):

        pdf.ln(4)

        pdf_write_wrapped(
            pdf,
            f"Question {i + 1}",
            style="B",
            size=12
        )

        pdf_write_wrapped(
            pdf,
            f"Question: {item.get('question', '')}",
            size=10
        )

        pdf_write_wrapped(
            pdf,
            f"Candidate Answer: {item.get('answer', '')}",
            size=10
        )

        technical = float(
            item.get(
                "technical_accuracy",
                0
            ) or 0
        )

        clarity = float(
            item.get(
                "clarity",
                0
            ) or 0
        )

        communication = float(
            item.get(
                "communication",
                0
            ) or 0
        )

        pdf_write_wrapped(
            pdf,
            f"Technical Accuracy: {technical:.1f}/10"
        )

        pdf_write_wrapped(
            pdf,
            f"Clarity: {clarity:.1f}/10"
        )

        pdf_write_wrapped(
            pdf,
            f"Communication: {communication:.1f}/10"
        )

        pdf_write_wrapped(
            pdf,
            f"Feedback: {item.get('feedback', '')}"
        )

        strong_points = item.get(
            "strong_points",
            []
        )

        if strong_points:

            pdf_write_wrapped(
                pdf,
                "Strong Points:",
                style="B"
            )

            for point in strong_points:

                pdf_write_wrapped(
                    pdf,
                    f"- {point}"
                )

        weak_points = item.get(
            "weak_points",
            []
        )

        if weak_points:

            pdf_write_wrapped(
                pdf,
                "Weak Points:",
                style="B"
            )

            for point in weak_points:

                pdf_write_wrapped(
                    pdf,
                    f"- {point}"
                )

        ideal_answer = item.get(
            "ideal_answer",
            ""
        )

        if ideal_answer:

            pdf_write_wrapped(
                pdf,
                f"Ideal Answer: {ideal_answer}"
            )

    pdf.ln(8)

    pdf_write_wrapped(
        pdf,
        "Powered by AI Mock Interviewer",
        style="I",
        size=9
    )

    output = pdf.output()

    if isinstance(output, bytearray):
        return bytes(output)

    if isinstance(output, bytes):
        return output

    return bytes(output)


# =========================================================
# PREMIUM CERTIFICATE
# =========================================================

def generate_certificate(
    candidate_name,
    role,
    level,
    avg
):

    pdf = FPDF(
        orientation="L",
        unit="mm",
        format="A4"
    )

    pdf.set_auto_page_break(False)
    pdf.add_page()

    # =====================================================
    # COLORS
    # =====================================================

    navy = (7, 22, 45)
    navy2 = (12, 35, 68)
    gold = (212, 170, 72)
    light_gold = (244, 218, 145)
    white = (255, 255, 255)
    soft_white = (218, 226, 238)
    line_gold = (176, 135, 45)

    # =====================================================
    # BACKGROUND
    # =====================================================

    pdf.set_fill_color(*navy)
    pdf.rect(0, 0, 297, 210, style="F")

    pdf.set_fill_color(*navy2)
    pdf.rect(18, 18, 261, 174, style="F")

    # =====================================================
    # BORDER
    # =====================================================

    pdf.set_draw_color(*gold)
    pdf.set_line_width(1.0)
    pdf.rect(8, 8, 281, 194)

    pdf.set_draw_color(*line_gold)
    pdf.set_line_width(0.35)
    pdf.rect(13, 13, 271, 184)

    # =====================================================
    # SIDE DECORATION
    # =====================================================

    pdf.set_fill_color(5, 18, 42)
    pdf.rect(18, 18, 14, 174, style="F")
    pdf.rect(265, 18, 14, 174, style="F")

    pdf.set_fill_color(*gold)
    pdf.rect(32, 18, 1.2, 174, style="F")
    pdf.rect(263.8, 18, 1.2, 174, style="F")

    # =====================================================
    # BRAND
    # =====================================================

    pdf.set_text_color(*soft_white)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_xy(33, 25)
    pdf.cell(231, 6, "AI MOCK INTERVIEWER", align="C")

    # =====================================================
    # TITLE
    # =====================================================

    pdf.set_text_color(*gold)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(33, 37)
    pdf.cell(231, 11, "CERTIFICATE OF COMPLETION", align="C")

    # Straight decorative lines only - no circle/ellipse
    pdf.set_draw_color(*gold)
    pdf.set_line_width(0.45)
    pdf.line(93, 52, 204, 52)

    # =====================================================
    # PRESENTED TO
    # =====================================================

    pdf.set_text_color(*soft_white)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(33, 57)
    pdf.cell(231, 6, "This certificate is proudly presented to", align="C")

    # =====================================================
    # CANDIDATE NAME
    # =====================================================

    safe_name = pdf_safe_text(candidate_name.upper())

    pdf.set_text_color(*white)
    pdf.set_font("Helvetica", "B", 25)
    pdf.set_xy(33, 66)
    pdf.cell(231, 12, safe_name, align="C")

    pdf.set_draw_color(*gold)
    pdf.set_line_width(0.5)
    pdf.line(105, 81, 192, 81)

    # =====================================================
    # COMPLETION TEXT
    # =====================================================

    pdf.set_text_color(*soft_white)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(33, 86)
    pdf.cell(231, 6, "for successfully completing the", align="C")

    pdf.set_text_color(*light_gold)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_xy(33, 94)
    pdf.cell(231, 7, "AI Mock Interview Assessment", align="C")

    # =====================================================
    # ROLE
    # =====================================================

    pdf.set_text_color(*white)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(33, 105)
    pdf.cell(231, 8, pdf_safe_text(role).upper(), align="C")

    # =====================================================
    # LEVEL
    # =====================================================

    pdf.set_text_color(*soft_white)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_xy(33, 114)
    pdf.cell(231, 6, pdf_safe_text(level).upper(), align="C")

    # =====================================================
    # SCORE + DATE BOXES
    # =====================================================

    box_y = 127
    box_w = 78
    box_h = 31
    gap = 12
    total_w = (box_w * 2) + gap
    left_x = (297 - total_w) / 2
    right_x = left_x + box_w + gap

    pdf.set_fill_color(15, 42, 78)
    pdf.set_draw_color(*gold)
    pdf.set_line_width(0.65)

    pdf.rect(left_x, box_y, box_w, box_h, style="DF")
    pdf.rect(right_x, box_y, box_w, box_h, style="DF")

    # Score box
    pdf.set_text_color(*gold)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_xy(left_x, box_y + 5)
    pdf.cell(box_w, 5, "OVERALL SCORE", align="C")

    pdf.set_text_color(*white)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_xy(left_x, box_y + 14)
    pdf.cell(box_w, 9, f"{float(avg):.1f} / 10", align="C")

    # Date box
    pdf.set_text_color(*gold)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_xy(right_x, box_y + 5)
    pdf.cell(box_w, 5, "INTERVIEW DATE", align="C")

    pdf.set_text_color(*white)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_xy(right_x, box_y + 14)
    pdf.cell(
        box_w,
        9,
        datetime.now().strftime("%d %B %Y"),
        align="C"
    )

    # =====================================================
    # SIGNATURE LINES
    # =====================================================

    signature_y = 176
    sig_w = 55
    left_sig_x = 76
    right_sig_x = 166

    pdf.set_draw_color(*soft_white)
    pdf.set_line_width(0.4)

    pdf.line(left_sig_x, signature_y, left_sig_x + sig_w, signature_y)
    pdf.line(right_sig_x, signature_y, right_sig_x + sig_w, signature_y)

    pdf.set_text_color(*soft_white)
    pdf.set_font("Helvetica", "", 8.5)

    pdf.set_xy(left_sig_x, signature_y + 2)
    pdf.cell(sig_w, 5, "AI Interviewer", align="C")

    pdf.set_xy(right_sig_x, signature_y + 2)
    pdf.cell(sig_w, 5, "Assessment", align="C")

    # =====================================================
    # FOOTER
    # =====================================================

    pdf.set_text_color(*soft_white)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_xy(33, 190)
    pdf.cell(231, 4, "AI Mock Interviewer  -  Professional Interview Assessment", align="C")

    # =====================================================
    # OUTPUT
    # =====================================================

    output = pdf.output()

    if isinstance(output, bytearray):
        return bytes(output)

    if isinstance(output, bytes):
        return output

    return bytes(output)


# HEADER
# =========================================================

st.title(
    "🎯 AI Mock Interviewer"
)

st.caption(
    "Resume-aware · JD-aligned · Certificate-ready"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## 🎯 Interview Progress"
    )

    current_question_number = (
        st.session_state.current_q_num
    )

    progress = min(
        current_question_number / 10,
        1.0
    )

    st.progress(progress)

    st.caption(
        f"Question {current_question_number} / 10"
    )

    if st.session_state.role:

        st.markdown(
            f"**Role:** {st.session_state.role}"
        )

    if st.session_state.level:

        st.markdown(
            f"**Level:** {st.session_state.level}"
        )

    if st.session_state.resume_text:

        st.success(
            "Resume-aware"
        )

    if st.session_state.jd_text:

        st.success(
            "JD-aligned"
        )


# =========================================================
# PHASE 1 - SETUP
# =========================================================

if st.session_state.phase == "setup":

    st.subheader(
        "📝 Interview Setup"
    )

    col1, col2 = st.columns(2)

    with col1:

        candidate_name = st.text_input(
            "Candidate Name",
            value=st.session_state.candidate_name,
            placeholder="Enter your name"
        )

        role = st.selectbox(
            "Role",
            ROLES,
            index=(
                ROLES.index(
                    st.session_state.role
                )
                if st.session_state.role in ROLES
                else 0
            )
        )

    with col2:

        level = st.selectbox(
            "Experience Level",
            LEVELS,
            index=(
                LEVELS.index(
                    st.session_state.level
                )
                if st.session_state.level in LEVELS
                else 0
            )
        )

    st.markdown(
        "### 📄 Resume"
    )

    resume_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )

    st.markdown(
        "### 💼 Job Description"
    )

    jd_text_input = st.text_area(
        "Paste Job Description",
        height=180,
        placeholder="Paste the job description here..."
    )

    jd_file = st.file_uploader(
        "Or upload Job Description PDF",
        type=["pdf"],
        key="jd_pdf"
    )

    start_button = st.button(
        "🚀 Start Interview",
        use_container_width=True
    )

    if start_button:

        if not candidate_name.strip():

            st.warning(
                "Please enter candidate name."
            )

        elif resume_file is None:

            st.warning(
                "Please upload your Resume PDF."
            )

        else:

            with st.spinner(
                "Reading resume and preparing your interview..."
            ):

                st.session_state.candidate_name = (
                    candidate_name.strip()
                )

                st.session_state.role = role

                st.session_state.level = level

                st.session_state.resume_text = (
                    extract_pdf_text(
                        resume_file
                    )
                )

                pasted_jd = (
                    jd_text_input or ""
                ).strip()

                uploaded_jd = ""

                if jd_file is not None:

                    uploaded_jd = extract_pdf_text(
                        jd_file
                    )

                st.session_state.jd_text = (
                    pasted_jd
                    if pasted_jd
                    else uploaded_jd
                )

                if not st.session_state.resume_text:

                    st.error(
                        "Could not extract text from the resume PDF."
                    )

                else:

                    try:

                        st.session_state.skill_gap = (
                            generate_skill_gap()
                        )

                        st.session_state.questions = []

                        st.session_state.evaluations = []

                        st.session_state.current_q_num = 0

                        st.session_state.started_at = (
                            datetime.now()
                        )

                        st.session_state.phase = (
                            "skill_gap"
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Setup failed: {e}"
                        )


# =========================================================
# PHASE 2 - SKILL GAP
# =========================================================

elif st.session_state.phase == "skill_gap":

    st.subheader(
        "🔎 Skill Gap Analysis"
    )

    skill_gap = (
        st.session_state.skill_gap
        or {}
    )

    required = skill_gap.get(
        "required_skills",
        []
    )

    matched = skill_gap.get(
        "matched_skills",
        []
    )

    missing = skill_gap.get(
        "missing_skills",
        []
    )

    recommendations = skill_gap.get(
        "recommendations",
        []
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Required Skills"
        )

        if required:

            for skill in required:

                st.write(
                    f"• {skill}"
                )

        else:

            st.write(
                "No specific required skills detected."
            )

    with col2:

        st.markdown(
            "### Matched Skills"
        )

        if matched:

            for skill in matched:

                st.write(
                    f"🟢 {skill}"
                )

        else:

            st.write(
                "No matched skills detected."
            )

    st.markdown(
        "### Missing / Development Areas"
    )

    if missing:

        for skill in missing:

            st.write(
                f"🔴 {skill}"
            )

    else:

        st.write(
            "No major skill gaps detected."
        )

    st.markdown(
        "### Skill Gap Radar"
    )

    show_skill_gap_radar(
        skill_gap
    )

    if recommendations:

        st.markdown(
            "### Recommendations"
        )

        for recommendation in recommendations:

            st.write(
                f"• {recommendation}"
            )

    continue_button = st.button(
        "➡️ Continue to Interview",
        use_container_width=True
    )

    if continue_button:

        st.session_state.phase = (
            "generating"
        )

        st.rerun()


# =========================================================
# PHASE 3 - GENERATING QUESTION
# =========================================================

elif st.session_state.phase == "generating":

    st.subheader(
        "🤖 Preparing Your Question..."
    )

    with st.spinner(
        "AI is creating a personalized interview question..."
    ):

        try:

            result, question = (
                generate_question()
            )

            st.session_state.current_question = (
                question
            )

            st.session_state.questions.append(
                question
            )

            st.session_state.current_q_num = (
                len(
                    st.session_state.questions
                )
            )

            st.session_state.answer_text = ""

            st.session_state.phase = (
                "interview"
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Question generation failed: {e}"
            )

            retry = st.button(
                "🔄 Retry"
            )

            if retry:
                st.rerun()


# =========================================================
# PHASE 4 - INTERVIEW
# =========================================================

elif st.session_state.phase == "interview":

    q_num = (
        st.session_state.current_q_num
    )

    st.subheader(
        f"🎤 Interview Question {q_num}"
    )

    st.progress(
        min(q_num / 10, 1.0)
    )

    components.html(
        """
        <div style="
            background:#111827;
            border:1px solid #263244;
            border-radius:12px;
            padding:12px 18px;
            text-align:center;
            margin-bottom:15px;
            font-family:Arial,sans-serif;
        ">

            <div style="
                color:#9ca3af;
                font-size:13px;
                margin-bottom:4px;
            ">
                ⏱️ Question Timer
            </div>

            <div id="timer" style="
                color:#ffffff;
                font-size:26px;
                font-weight:700;
            ">
                02:00
            </div>

        </div>

        <script>

            let timeLeft = 120;

            const timerElement =
                document.getElementById("timer");

            const timerInterval =
                setInterval(function () {

                    let minutes =
                        Math.floor(timeLeft / 60);

                    let seconds =
                        String(timeLeft % 60).padStart(2, "0");

                    if (timerElement) {

                        timerElement.innerText =
                            minutes + ":" + seconds;

                    }

                    if (timeLeft <= 0) {

                        clearInterval(timerInterval);

                        if (timerElement) {

                            timerElement.innerText =
                                "Time Up";

                        }

                    } else {

                        timeLeft--;

                    }

                }, 1000);

        </script>
        """,
        height=105,
        scrolling=False
    )

    st.markdown(
        f"""
        <div class="question-card">

        <div style="
            color:#9ca3af;
            font-size:13px;
            margin-bottom:10px;
        ">
        QUESTION {q_num}
        </div>

        <div style="
            color:#ffffff;
            font-size:20px;
            line-height:1.6;
            font-weight:600;
        ">
        {st.session_state.current_question}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    answer = st.text_area(
        "Your Answer",
        value=st.session_state.answer_text,
        height=220,
        placeholder="Type your answer here..."
    )

    col1, col2 = st.columns(2)

    with col1:

        submit_answer = st.button(
            "📤 Submit Answer",
            use_container_width=True
        )

    with col2:

        finish_interview = st.button(
            "🏁 Finish Interview",
            use_container_width=True
        )

    if submit_answer:

        if not answer.strip():

            st.warning(
                "Please enter your answer."
            )

        else:

            with st.spinner(
                "AI is evaluating your answer..."
            ):

                try:

                    result = evaluate_answer(
                        st.session_state.current_question,
                        answer.strip()
                    )

                    scores = result.get(
                        "scores",
                        {}
                    )

                    evaluation = {

                        "question":
                            st.session_state.current_question,

                        "answer":
                            answer.strip(),

                        "technical_accuracy":
                            scores.get(
                                "technical_accuracy",
                                0
                            ) or 0,

                        "clarity":
                            scores.get(
                                "clarity",
                                0
                            ) or 0,

                        "communication":
                            scores.get(
                                "communication",
                                0
                            ) or 0,

                        "feedback":
                            result.get(
                                "feedback",
                                ""
                            ),

                        "strong_points":
                            result.get(
                                "strong_points",
                                []
                            ),

                        "weak_points":
                            result.get(
                                "weak_points",
                                []
                            ),

                        "ideal_answer":
                            result.get(
                                "ideal_answer",
                                ""
                            ),

                        "followup_question":
                            result.get(
                                "followup_question",
                                None
                            )
                    }

                    st.session_state.evaluations.append(
                        evaluation
                    )

                    st.session_state.answer_text = ""

                    st.session_state.phase = (
                        "feedback"
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"Answer evaluation failed: {e}"
                    )

    if finish_interview:

        if st.session_state.evaluations:

            st.session_state.finished_at = (
                datetime.now()
            )

            st.session_state.phase = (
                "report_gen"
            )

            st.rerun()

        else:

            st.warning(
                "Please answer at least one question before finishing."
            )


# =========================================================
# PHASE 5 - FEEDBACK
# =========================================================

elif st.session_state.phase == "feedback":

    if not st.session_state.evaluations:

        st.session_state.phase = (
            "generating"
        )

        st.rerun()

    latest = (
        st.session_state.evaluations[-1]
    )

    st.subheader(
        "📊 AI Feedback"
    )

    c1, c2, c3 = st.columns(3)

    technical = float(
        latest.get(
            "technical_accuracy",
            0
        ) or 0
    )

    clarity = float(
        latest.get(
            "clarity",
            0
        ) or 0
    )

    communication = float(
        latest.get(
            "communication",
            0
        ) or 0
    )

    c1.metric(
        f"{score_color(technical)} Technical Accuracy",
        f"{technical:.1f}/10"
    )

    c2.metric(
        f"{score_color(clarity)} Clarity",
        f"{clarity:.1f}/10"
    )

    c3.metric(
        f"{score_color(communication)} Communication",
        f"{communication:.1f}/10"
    )

    st.markdown(
        "### 💬 Feedback"
    )

    st.info(
        latest.get(
            "feedback",
            "No feedback available."
        )
    )

    strong_points = latest.get(
        "strong_points",
        []
    )

    if strong_points:

        st.markdown(
            "### ✅ Strong Points"
        )

        for point in strong_points:

            st.write(
                f"• {point}"
            )

    weak_points = latest.get(
        "weak_points",
        []
    )

    if weak_points:

        st.markdown(
            "### ⚠️ Areas to Improve"
        )

        for point in weak_points:

            st.write(
                f"• {point}"
            )

    ideal_answer = latest.get(
        "ideal_answer",
        ""
    )

    if ideal_answer:

        with st.expander(
            "💡 View Ideal Answer"
        ):

            st.write(
                ideal_answer
            )

    followup = latest.get(
        "followup_question"
    )

    if followup:

        st.markdown(
            "### 🔄 Adaptive Follow-up"
        )

        st.info(
            followup
        )

    answered_count = len(
        st.session_state.evaluations
    )

    st.markdown(
        "### Live Session Stats"
    )

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Questions Answered",
        answered_count
    )

    all_scores = []

    for item in st.session_state.evaluations:

        t = float(
            item.get(
                "technical_accuracy",
                0
            ) or 0
        )

        c = float(
            item.get(
                "clarity",
                0
            ) or 0
        )

        com = float(
            item.get(
                "communication",
                0
            ) or 0
        )

        all_scores.append(
            (t + c + com) / 3
        )

    current_avg = (
        sum(all_scores) / len(all_scores)
        if all_scores
        else 0
    )

    s2.metric(
        "Average Score",
        f"{current_avg:.2f}/10"
    )

    s3.metric(
        "Questions Remaining",
        max(
            0,
            10 - answered_count
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        next_question = st.button(
            "➡️ Next Question",
            use_container_width=True
        )

    with col2:

        finish = st.button(
            "🏁 Finish Interview",
            use_container_width=True
        )

    if next_question:

        if answered_count >= 10:

            st.session_state.finished_at = (
                datetime.now()
            )

            st.session_state.phase = (
                "report_gen"
            )

            st.rerun()

        else:

            st.session_state.phase = (
                "generating"
            )

            st.rerun()

    if finish:

        st.session_state.finished_at = (
            datetime.now()
        )

        st.session_state.phase = (
            "report_gen"
        )

        st.rerun()


# =========================================================
# PHASE 6 - REPORT GENERATION
# =========================================================

elif st.session_state.phase == "report_gen":

    st.subheader(
        "📑 Generating Final Report..."
    )

    with st.spinner(
        "AI is preparing your final performance report..."
    ):

        try:

            report = generate_final_report()

            st.session_state.final_report = (
                report
            )

            evaluations = (
                st.session_state.evaluations
            )

            averages = []

            for item in evaluations:

                technical = float(
                    item.get(
                        "technical_accuracy",
                        0
                    ) or 0
                )

                clarity = float(
                    item.get(
                        "clarity",
                        0
                    ) or 0
                )

                communication = float(
                    item.get(
                        "communication",
                        0
                    ) or 0
                )

                averages.append(
                    (
                        technical +
                        clarity +
                        communication
                    ) / 3
                )

            avg = (
                sum(averages) /
                len(averages)
                if averages
                else 0
            )

            st.session_state.report_pdf = (
                generate_pdf_report(
                    st.session_state.candidate_name,
                    st.session_state.role,
                    st.session_state.level,
                    avg,
                    evaluations,
                    report
                )
            )

            st.session_state.certificate_pdf = (
                generate_certificate(
                    st.session_state.candidate_name,
                    st.session_state.role,
                    st.session_state.level,
                    avg
                )
            )

            st.session_state.phase = (
                "report"
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Final report generation failed: {e}"
            )

            st.write(
                "If the problem is related to PDF creation, the report text was generated but PDF creation failed."
            )

            retry = st.button(
                "🔄 Retry Report Generation"
            )

            if retry:
                st.rerun()


# =========================================================
# PHASE 7 - FINAL REPORT
# =========================================================

elif st.session_state.phase == "report":

    st.subheader(
        "🏆 Interview Complete"
    )

    evaluations = (
        st.session_state.evaluations
    )

    averages = []

    for item in evaluations:

        technical = float(
            item.get(
                "technical_accuracy",
                0
            ) or 0
        )

        clarity = float(
            item.get(
                "clarity",
                0
            ) or 0
        )

        communication = float(
            item.get(
                "communication",
                0
            ) or 0
        )

        averages.append(
            (
                technical +
                clarity +
                communication
            ) / 3
        )

    avg = (
        sum(averages) /
        len(averages)
        if averages
        else 0
    )

    # =====================================================
    # SUMMARY CARDS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Questions",
        len(evaluations)
    )

    c2.metric(
        "Overall Score",
        f"{avg:.2f}/10"
    )

    technical_values = [
        float(
            x.get(
                "technical_accuracy",
                0
            ) or 0
        )
        for x in evaluations
    ]

    clarity_values = [
        float(
            x.get(
                "clarity",
                0
            ) or 0
        )
        for x in evaluations
    ]

    communication_values = [
        float(
            x.get(
                "communication",
                0
            ) or 0
        )
        for x in evaluations
    ]

    technical_avg = (
        sum(technical_values) /
        len(technical_values)
        if technical_values
        else 0
    )

    clarity_avg = (
        sum(clarity_values) /
        len(clarity_values)
        if clarity_values
        else 0
    )

    communication_avg = (
        sum(communication_values) /
        len(communication_values)
        if communication_values
        else 0
    )

    c3.metric(
        "Technical",
        f"{technical_avg:.2f}/10"
    )

    c4.metric(
        "Communication",
        f"{communication_avg:.2f}/10"
    )

    # =====================================================
    # AI FINAL REPORT
    # =====================================================

    st.markdown(
        "## 🤖 AI Final Report"
    )

    st.markdown(
        f"""
        <div class="card">

        <h3>Candidate</h3>

        <p>
        <b>{st.session_state.candidate_name}</b>
        </p>

        <p>
        Role: {st.session_state.role}
        </p>

        <p>
        Level: {st.session_state.level}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        st.session_state.final_report
    )

    # =====================================================
    # PERFORMANCE RADAR
    # =====================================================

    st.markdown(
        "## 📡 Performance Radar"
    )

    show_performance_radar(
        evaluations
    )

    # =====================================================
    # NEW PERFORMANCE DASHBOARD
    # =====================================================

    show_performance_dashboard(
        evaluations
    )

    # =====================================================
    # DETAILED QUESTION REVIEW
    # =====================================================

    show_question_breakdown(
        evaluations
    )

    # =====================================================
    # DOWNLOADS
    # =====================================================

    st.markdown(
        "## 📥 Downloads"
    )

    d1, d2 = st.columns(2)

    with d1:

        if st.session_state.report_pdf:

            st.download_button(
                label="📄 Download PDF Report",
                data=st.session_state.report_pdf,
                file_name=(
                    "AI_Mock_Interview_Report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )

    with d2:

        if st.session_state.certificate_pdf:

            st.download_button(
                label="🏆 Download Certificate",
                data=st.session_state.certificate_pdf,
                file_name=(
                    "AI_Mock_Interview_Certificate.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )

    st.markdown(
        "---"
    )

    new_interview = st.button(
        "🔄 Start New Interview",
        use_container_width=True
    )

    if new_interview:

        keys_to_reset = [
            "phase",
            "candidate_name",
            "role",
            "level",
            "resume_text",
            "jd_text",
            "skill_gap",
            "questions",
            "evaluations",
            "current_question",
            "current_q_num",
            "answer_text",
            "started_at",
            "finished_at",
            "final_report",
            "report_pdf",
            "certificate_pdf",
            "session_id"
        ]

        for key in keys_to_reset:

            if key == "phase":

                st.session_state[key] = "setup"

            elif key == "questions":

                st.session_state[key] = []

            elif key == "evaluations":

                st.session_state[key] = []

            elif key == "current_q_num":

                st.session_state[key] = 0

            elif key == "session_id":

                st.session_state[key] = str(
                    uuid.uuid4()
                )

            else:

                st.session_state[key] = (
                    ""
                    if isinstance(
                        st.session_state.get(key),
                        str
                    )
                    else None
                )

        st.rerun()
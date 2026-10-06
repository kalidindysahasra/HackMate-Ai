import os
import uuid
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="HackMate AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# SESSION STATE
# ==================================================

DEFAULTS = {
    "problem_statement": "",
    "problem_analysis": "",
    "idea_review": "",
    "project_idea": "",
    "design_result": "",
    "build_result": "",
    "validation_result": "",
    "pitch_result": "",
    "document_processed": False,
    "document_name": "",
    "activity": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# HELPERS
# ==================================================

def add_activity(message):
    st.session_state.activity.insert(0, message)
    st.session_state.activity = st.session_state.activity[:8]


def ask_ai(prompt):
    response = ollama.chat(
        model="gemma3",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


@st.cache_resource

def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource

def get_chroma_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="hackmate_documents")


def create_chunks(text, chunk_size=200):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
        if words[i:i + chunk_size]
    ]


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n".join(pages)


def get_problem_context():
    return (
        f"Problem Statement:\n{st.session_state.problem_statement}\n\n"
        f"Problem Analysis:\n{st.session_state.problem_analysis}\n\n"
        f"Project Idea:\n{st.session_state.project_idea}"
    )


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.title("🚀 HackMate AI")
    st.caption("Hackathon Intelligence Platform")
    st.divider()

    st.caption("CURRENT PROJECT")
    st.write("🚀 HackMate AI")
    st.caption("Active project")
    st.divider()

    st.caption("NAVIGATION")
    page = st.radio(
        "Go to",
        [
            "📊 Dashboard",
            "📚 Knowledge Base",
            "💬 Ask HackMate",
            "🔍 Problem Analyzer",
            "💡 Idea Reviewer",
            "🏗️ Design",
            "🔨 Build",
            "🏆 Validate",
            "🎤 Pitch",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("AI ENGINE")
    st.success("● Local AI")
    st.caption("Gemma 3")
    st.caption("Powered by Ollama")

# ==================================================
# DASHBOARD
# ==================================================

if page == "📊 Dashboard":

    # --------------------------------------------------
    # DASHBOARD STYLING
    # --------------------------------------------------

    st.markdown("""
    <style>

    .dashboard-hero {
        padding: 10px 0 25px 0;
    }

    .dashboard-hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .dashboard-hero p {
        font-size: 16px;
        opacity: 0.7;
        margin-top: 0;
    }

    .stat-card {
        padding: 20px;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        min-height: 125px;
        background: rgba(128,128,128,0.05);
    }

    .stat-label {
        font-size: 13px;
        opacity: 0.65;
        margin-bottom: 8px;
    }

    .stat-value {
        font-size: 28px;
        font-weight: 700;
    }

    .stat-small {
        font-size: 13px;
        opacity: 0.65;
        margin-top: 5px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 650;
        margin-top: 28px;
        margin-bottom: 14px;
    }

    .stage-card {
        text-align: center;
        padding: 16px 8px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        min-height: 105px;
        background: rgba(128,128,128,0.04);
    }

    .stage-icon {
        font-size: 24px;
        margin-bottom: 7px;
    }

    .stage-name {
        font-weight: 600;
        font-size: 14px;
    }

    .stage-status {
        font-size: 12px;
        opacity: 0.65;
        margin-top: 5px;
    }

    .next-action {
        padding: 22px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.05);
    }

    .next-label {
        font-size: 12px;
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .next-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 6px;
    }

    .project-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        min-height: 160px;
        background: rgba(128,128,128,0.04);
    }

    .project-title {
        font-size: 17px;
        font-weight: 650;
        margin-bottom: 8px;
    }

    .project-description {
        font-size: 13px;
        opacity: 0.7;
        line-height: 1.5;
    }

    .activity-card {
        padding: 12px 15px;
        margin-bottom: 8px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.2);
        background: rgba(128,128,128,0.03);
        font-size: 14px;
    }

    </style>
    """, unsafe_allow_html=True)


    # --------------------------------------------------
    # CALCULATE STATUS
    # --------------------------------------------------

    stages = [
        ("Understand", "🔍", bool(st.session_state.problem_statement)),
        ("Ideate", "💡", bool(st.session_state.idea_review)),
        ("Design", "🏗️", bool(st.session_state.design_result)),
        ("Build", "🔨", bool(st.session_state.build_result)),
        ("Validate", "🏆", bool(st.session_state.validation_result)),
        ("Pitch", "🎤", bool(st.session_state.pitch_result)),
    ]

    completed = sum(done for _, _, done in stages)
    health_score = int((completed / len(stages)) * 100)

    problem_score = 100 if st.session_state.problem_statement else 0
    idea_score = 100 if st.session_state.idea_review else 0

    if not st.session_state.problem_statement:
        next_action = "Analyze Problem"
        next_icon = "🔍"
    elif not st.session_state.idea_review:
        next_action = "Review Your Idea"
        next_icon = "💡"
    elif not st.session_state.design_result:
        next_action = "Design Architecture"
        next_icon = "🏗️"
    elif not st.session_state.build_result:
        next_action = "Analyze Build"
        next_icon = "🔨"
    elif not st.session_state.validation_result:
        next_action = "Validate Project"
        next_icon = "🏆"
    else:
        next_action = "Prepare Final Pitch"
        next_icon = "🎤"


    # --------------------------------------------------
    # HERO
    # --------------------------------------------------

    st.markdown("""
    <div class="dashboard-hero">
        <h1>🚀 HackMate AI</h1>
        <p>Your AI workspace for turning a hackathon problem into a working project.</p>
    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------
    # TOP STATUS CARDS
    # --------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">PROJECT HEALTH</div>
            <div class="stat-value">{health_score}%</div>
            <div class="stat-small">{completed}/6 stages completed</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">PROBLEM</div>
            <div class="stat-value">{problem_score}%</div>
            <div class="stat-small">
                {"Problem analyzed" if problem_score else "Waiting for analysis"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">IDEA</div>
            <div class="stat-value">{idea_score}%</div>
            <div class="stat-small">
                {"Idea reviewed" if idea_score else "Waiting for review"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">AI ENGINE</div>
            <div class="stat-value">● Local</div>
            <div class="stat-small">Gemma 3 + Ollama</div>
        </div>
        """, unsafe_allow_html=True)


    # --------------------------------------------------
    # DEVELOPMENT JOURNEY
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">Development Journey</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(6)

    for col, (name, icon, done) in zip(cols, stages):

        with col:

            if done:
                status = "Completed"
                status_icon = "✓"
            else:
                status = "Pending"
                status_icon = "○"

            st.markdown(f"""
            <div class="stage-card">
                <div class="stage-icon">{icon}</div>
                <div class="stage-name">{name}</div>
                <div class="stage-status">
                    {status_icon} {status}
                </div>
            </div>
            """, unsafe_allow_html=True)


    # --------------------------------------------------
    # NEXT ACTION
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">What should you do next?</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([3, 1])

    with left:
        st.markdown(f"""
        <div class="next-action">
            <div class="next-label">Recommended next step</div>
            <div class="next-title">{next_icon} {next_action}</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.metric(
            "Progress",
            f"{completed}/6",
            f"{health_score}% complete"
        )


    # --------------------------------------------------
    # PROJECT WORKSPACE
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">Project Workspace</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        status = (
            "Completed"
            if st.session_state.problem_statement
            else "Not started"
        )

        st.markdown(f"""
        <div class="project-card">
            <div class="project-title">🔍 Understand</div>
            <div class="project-description">
                Break down the problem, users, requirements,
                constraints and judging criteria.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if status == "Completed":
            st.success("✓ Completed")
        else:
            st.info("○ Not started")


    with p2:

        status = (
            "Completed"
            if st.session_state.idea_review
            else "Not started"
        )

        st.markdown("""
        <div class="project-card">
            <div class="project-title">💡 Ideate</div>
            <div class="project-description">
                Test your idea against the actual problem
                before investing time in development.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if status == "Completed":
            st.success("✓ Completed")
        else:
            st.info("○ Not started")


    with p3:

        st.markdown("""
        <div class="project-card">
            <div class="project-title">🤖 AI Mentor</div>
            <div class="project-description">
                Get local AI assistance throughout the
                entire hackathon development process.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.success("● Online")


    # --------------------------------------------------
    # RECENT ACTIVITY
    # --------------------------------------------------

    if st.session_state.activity:

        st.markdown(
            '<div class="section-title">Recent Activity</div>',
            unsafe_allow_html=True
        )

        for item in st.session_state.activity[:5]:
            st.markdown(
                f"""
                <div class="activity-card">
                    ✓ &nbsp; {item}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.markdown(
            '<div class="section-title">Recent Activity</div>',
            unsafe_allow_html=True
        )

        st.info(
            "No activity yet. Start by analyzing your hackathon problem."
        )




# ==================================================
# KNOWLEDGE BASE
# ==================================================

elif page == "📚 Knowledge Base":
    st.title("📚 Knowledge Base")
    st.write("Upload hackathon rules, requirements and supporting documents.")

    uploaded_file = st.file_uploader("Upload hackathon PDF", type=["pdf"])

    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

        if st.button("⚙️ Process Document", use_container_width=True):
            with st.spinner("Reading and indexing document..."):
                text = extract_text_from_pdf(uploaded_file)
                chunks = create_chunks(text)

                if not chunks:
                    st.error("No readable text was found in this PDF.")
                else:
                    model = load_embedding_model()
                    embeddings = model.encode(chunks).tolist()
                    collection = get_chroma_collection()

                    ids = [str(uuid.uuid4()) for _ in chunks]
                    collection.add(
                        documents=chunks,
                        embeddings=embeddings,
                        ids=ids,
                        metadatas=[{"source": uploaded_file.name} for _ in chunks],
                    )

                    st.session_state.document_processed = True
                    st.session_state.document_name = uploaded_file.name
                    add_activity(f"Indexed {uploaded_file.name} ({len(chunks)} chunks)")
                    st.success(f"Document processed successfully — {len(chunks)} chunks stored.")

    try:
        collection = get_chroma_collection()
        st.metric("Stored Knowledge Chunks", collection.count())
    except Exception as e:
        st.error(f"Knowledge base error: {e}")


# ==================================================
# ASK HACKMATE
# ==================================================

elif page == "💬 Ask HackMate":
    st.title("💬 Ask HackMate")
    st.write("Ask questions about the documents in your knowledge base.")

    question = st.text_input(
        "Your question",
        placeholder="Example: What are the requirements for the social media post?",
    )

    if st.button("🤖 Ask HackMate", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            collection = get_chroma_collection()
            if collection.count() == 0:
                st.warning("Please upload and process a document first.")
            else:
                with st.spinner("HackMate is thinking..."):
                    model = load_embedding_model()
                    question_embedding = model.encode([question]).tolist()
                    results = collection.query(
                        query_embeddings=question_embedding,
                        n_results=min(5, collection.count()),
                    )
                    chunks = results.get("documents", [[]])[0]
                    context = "\n\n".join(chunks)

                    prompt = f"""
You are HackMate AI, an AI technical mentor for hackathons.

Answer the question using ONLY the supplied document context.
Do not invent facts. If the answer is not supported by the context,
say: "The uploaded document does not provide this information."

CONTEXT:
{context}

QUESTION:
{question}

Give a concise, useful answer.
"""
                    try:
                        answer = ask_ai(prompt)
                        st.subheader("🤖 HackMate")
                        st.write(answer)
                        add_activity(f"Asked HackMate: {question[:70]}")
                    except Exception as e:
                        st.error(f"Could not contact Ollama: {e}")


# ==================================================
# PROBLEM ANALYZER
# ==================================================

elif page == "🔍 Problem Analyzer":
    st.title("🔍 Problem Analyzer")
    st.write("Understand exactly what the hackathon problem is asking.")

    problem_input = st.text_area(
        "Hackathon Problem Statement",
        value=st.session_state.problem_statement,
        height=280,
        key="problem_input",
        placeholder="Paste the complete problem statement here...",
    )

    if st.button("🔍 Analyze Problem", use_container_width=True):
        if not problem_input.strip():
            st.warning("Please paste a problem statement first.")
        else:
            with st.spinner("Analyzing problem statement..."):
                prompt = f"""
You are HackMate AI, an AI technical mentor for hackathons.

Analyze ONLY the information explicitly present in the problem statement.
Do not invent requirements or judging criteria.

Use EXACTLY these sections:

## 🎯 Problem
## 👥 Target Users
## ⚙️ Required Functionality
## 🚧 Constraints
## 🏆 Judging Criteria
## 🔑 Important Requirements
## 💡 What the Solution Must Demonstrate

If a section is not mentioned, write:
"Not specified in the problem statement."

Problem Statement:
{problem_input}
"""
                try:
                    analysis = ask_ai(prompt)
                    st.session_state.problem_statement = problem_input
                    st.session_state.problem_analysis = analysis
                    add_activity("Analyzed the hackathon problem statement")
                    st.success("Problem analysis completed.")
                except Exception as e:
                    st.error(f"Could not contact Ollama: {e}")

    if st.session_state.problem_analysis:
        st.subheader("📋 Problem Analysis")
        st.markdown(st.session_state.problem_analysis)


# ==================================================
# IDEA REVIEWER
# ==================================================

elif page == "💡 Idea Reviewer":
    st.title("💡 Idea Reviewer")
    st.write("Challenge your idea before spending time building it.")

    if not st.session_state.problem_statement:
        st.warning("Analyze a problem statement first.")

    project_idea = st.text_area(
        "Your Project Idea",
        value=st.session_state.project_idea,
        height=240,
        key="idea_input",
        placeholder="Describe what you want to build...",
    )

    if st.button("💡 Review My Idea", use_container_width=True):
        if not st.session_state.problem_statement:
            st.warning("Please analyze a problem statement first.")
        elif not project_idea.strip():
            st.warning("Please enter your project idea.")
        else:
            with st.spinner("Reviewing your idea..."):
                prompt = f"""
You are HackMate AI, an AI technical mentor for hackathons.

Review the project idea against the problem statement.
Clearly separate explicit requirements from recommendations.
Never present a recommendation as a mandatory requirement.

Use EXACTLY this format:

## 💡 Idea Review
**Overall:** [short judgement]
**Score:** [score out of 10]

### 🎯 Does it solve the problem?
2-3 sentences.

### ✅ What's working
3-5 bullets.

### ⚠️ Biggest gaps
2-4 important gaps with brief explanations.

### 🔴 Must Fix
ONLY explicitly required items that are missing or unclear.
If none: "Nothing critical is missing."

### 🟡 Recommended
Useful improvements that are not explicit requirements.

### 🟢 Optional
Useful but non-essential enhancements.

### 🚀 Next Best Action
ONE specific action.

### 🔄 Suggested Flow
Use arrows.

PROBLEM:
{st.session_state.problem_statement}

ANALYSIS:
{st.session_state.problem_analysis}

PROJECT IDEA:
{project_idea}
"""
                try:
                    review = ask_ai(prompt)
                    st.session_state.project_idea = project_idea
                    st.session_state.idea_review = review
                    add_activity("Reviewed the project idea")
                    st.success("Idea review completed.")
                except Exception as e:
                    st.error(f"Could not contact Ollama: {e}")

    if st.session_state.idea_review:
        st.subheader("🧠 HackMate's Review")
        st.markdown(st.session_state.idea_review)


# ==================================================
# DESIGN
# ==================================================

elif page == "🏗️ Design":
    st.title("🏗️ Design")
    st.write("Turn your idea into a concrete technical architecture.")

    if not st.session_state.project_idea:
        st.warning("Review your project idea first.")

    design_notes = st.text_area(
        "Optional design preferences",
        height=140,
        placeholder="Example: Python backend, Streamlit frontend, local AI...",
    )

    if st.button("🏗️ Generate Architecture", use_container_width=True):
        if not st.session_state.project_idea:
            st.warning("Please complete Idea Reviewer first.")
        else:
            with st.spinner("Designing the architecture..."):
                prompt = f"""
You are HackMate AI, a senior software architect helping a hackathon team.

Create a practical architecture for this project.
Prefer technologies that are realistic for a student hackathon team.
Do not add unnecessary technologies.

Return EXACTLY:

## 🏗️ Architecture

### 1. Core Components
- component — purpose

### 2. Data Flow
Show the flow using arrows.

### 3. Technology Choices
Give technology + reason.

### 4. Input / Output
Explain the main inputs and outputs.

### 5. MVP
List only the minimum features needed for a working demo.

### 6. Future Improvements
List features that can wait until after the MVP.

PROJECT IDEA:
{st.session_state.project_idea}

PROBLEM:
{st.session_state.problem_statement}

DESIGN PREFERENCES:
{design_notes}
"""
                try:
                    result = ask_ai(prompt)
                    st.session_state.design_result = result
                    add_activity("Generated the project architecture")
                    st.success("Architecture generated.")
                except Exception as e:
                    st.error(f"Could not contact Ollama: {e}")

    if st.session_state.design_result:
        st.markdown(st.session_state.design_result)


# ==================================================
# BUILD
# ==================================================

elif page == "🔨 Build":
    st.title("🔨 Build Assistant")
    st.write("Paste code, an error, or a build problem and let HackMate help you fix it.")

    code_or_error = st.text_area(
        "Code / Error / Pipeline Output",
        height=300,
        placeholder="Paste the code, traceback, terminal error, or pipeline log here...",
    )

    build_question = st.text_input(
        "What do you want HackMate to help with?",
        placeholder="Example: Find the cause and tell me exactly what to change.",
    )

    if st.button("🔨 Analyze Build Problem", use_container_width=True):
        if not code_or_error.strip():
            st.warning("Paste some code, an error, or a log first.")
        else:
            with st.spinner("Analyzing your build problem..."):
                prompt = f"""
You are HackMate AI, a practical coding mentor.

Analyze the supplied code/error/log.
Do not invent errors that are not present.
Give beginner-friendly, actionable guidance.

Return:

## 🔨 Build Analysis

### What is wrong?

### Why is it happening?

### Exact fix
Give the smallest practical change.

### Updated code
Only if code needs changing.

### How to test it
Give exact commands or steps.

### Prevention
One or two ways to avoid this problem again.

PROJECT:
{st.session_state.project_idea}

USER QUESTION:
{build_question}

INPUT:
{code_or_error}
"""
                try:
                    result = ask_ai(prompt)
                    st.session_state.build_result = result
                    add_activity("Analyzed a build/code problem")
                    st.success("Build analysis completed.")
                except Exception as e:
                    st.error(f"Could not contact Ollama: {e}")

    if st.session_state.build_result:
        st.markdown(st.session_state.build_result)


# ==================================================
# VALIDATE
# ==================================================

elif page == "🏆 Validate":
    st.title("🏆 Validate")
    st.write("Check whether your project actually satisfies the problem requirements.")

    if not st.session_state.problem_statement:
        st.warning("Analyze the problem statement first.")

    validation_extra = st.text_area(
        "Optional final project details",
        height=180,
        placeholder="Describe what is currently implemented, tested, and demo-ready...",
    )

    if st.button("🏆 Validate Project", use_container_width=True):
        if not st.session_state.problem_statement:
            st.warning("Please analyze the problem statement first.")
        else:
            with st.spinner("Validating the project..."):
                prompt = f"""
You are HackMate AI acting as a strict hackathon evaluator.

Compare the project against ONLY the explicit requirements in the problem statement.
Do not turn recommendations into mandatory requirements.

Return EXACTLY:

## 🏆 Project Validation

### Requirement Checklist
For every explicit requirement:
- ✅ Satisfied
- ⚠️ Partially satisfied
- ❌ Missing

### Strong Points

### Critical Gaps

### Demo Risks

### Final Readiness Score
Give a score out of 10 and explain it briefly.

### Next Best Action
Give ONE highest-priority action.

PROBLEM:
{st.session_state.problem_statement}

PROBLEM ANALYSIS:
{st.session_state.problem_analysis}

IDEA:
{st.session_state.project_idea}

ARCHITECTURE:
{st.session_state.design_result}

BUILD STATUS:
{st.session_state.build_result}

FINAL PROJECT DETAILS:
{validation_extra}
"""
                try:
                    result = ask_ai(prompt)
                    st.session_state.validation_result = result
                    add_activity("Validated the project against requirements")
                    st.success("Project validation completed.")
                except Exception as e:
                    st.error(f"Could not contact Ollama: {e}")

    if st.session_state.validation_result:
        st.markdown(st.session_state.validation_result)


# ==================================================
# PITCH
# ==================================================

elif page == "🎤 Pitch":
    st.title("🎤 Pitch Coach")
    st.write("Prepare the final demo, pitch and judge Q&A.")

    pitch_time = st.selectbox(
        "Pitch duration",
        ["1 minute", "2 minutes", "3 minutes", "5 minutes"],
    )

    if st.button("🎤 Generate Pitch", use_container_width=True):
        if not st.session_state.project_idea:
            st.warning("Complete your project idea first.")
        else:
            with st.spinner("Preparing your pitch..."):
                prompt = f"""
You are HackMate AI acting as a hackathon pitch coach.

Prepare a practical final pitch for the project below.
Avoid exaggerated claims. Use only supported project details.

Pitch duration: {pitch_time}

Return EXACTLY:

## 🎤 Final Pitch

### 1. Opening Hook

### 2. Problem

### 3. Solution

### 4. How It Works

### 5. What Makes It Strong

### 6. Demo Flow
Numbered steps for the live demo.

### 7. Closing

### 8. Judge Questions
Give 8 likely technical/judging questions and short answer guidance.

### 9. Weaknesses to Prepare For
Give the 3 hardest questions the team may face.

PROJECT IDEA:
{st.session_state.project_idea}

PROBLEM:
{st.session_state.problem_statement}

ARCHITECTURE:
{st.session_state.design_result}

VALIDATION:
{st.session_state.validation_result}
"""
                try:
                    result = ask_ai(prompt)
                    st.session_state.pitch_result = result
                    add_activity("Generated final pitch and judge questions")
                    st.success("Pitch generated.")
                except Exception as e:
                    st.error(f"Could not contact Ollama: {e}")

    if st.session_state.pitch_result:
        st.markdown(st.session_state.pitch_result)


# ==================================================
# FOOTER
# ==================================================

st.divider()
st.caption(
    "HackMate AI • Local AI Technical Mentor • Gemma 3 + ChromaDB + Sentence Transformers"
)

import streamlit as st # type: ignore
import time

from src.agents.agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Researcher AI",
    page_icon="🔬",
    layout="wide",
)

# ============================================================
# SIMPLE CSS
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
    }

    .main-title {
        font-size: 48px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .small-text {
        color: #94a3b8;
        font-size: 14px;
    }

    .report-box {
        background-color: #111827;
        border: 1px solid #263449;
        border-radius: 12px;
        padding: 20px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 50px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔬 Researcher <span style="color:#60a5fa;">AI</span></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'A multi-agent AI system that searches, reads, writes and reviews research.'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# INPUT + PIPELINE
# ============================================================

left, right = st.columns([1.3, 1], gap="large")


# ============================================================
# LEFT SIDE
# ============================================================

with left:

    st.markdown(
        '<div class="section-title">📝 Research Topic</div>',
        unsafe_allow_html=True,
    )

    topic = st.text_input(
        "Enter your topic",
        placeholder="Example: Future of AI Agents in Customer Service",
        label_visibility="collapsed",
    )

    run_button = st.button(
        "🚀 Start Research",
        use_container_width=True,
        type="primary",
    )

    st.markdown("### 💡 Example Topics")

    examples = [
        "Future of AI Agents",
        "Latest developments in Generative AI",
        "AI in Customer Service",
        "Future of LangChain",
    ]

    for example in examples:
        if st.button(
            example,
            key=f"example_{example}",
            use_container_width=True,
        ):
            topic = example
            run_button = True


# ============================================================
# RIGHT SIDE - PIPELINE
# ============================================================

with right:

    st.markdown(
        '<div class="section-title">⚙️ Agent Pipeline</div>',
        unsafe_allow_html=True,
    )

    st.info("Your research passes through four AI stages.")

    st.markdown("### 1️⃣ Search Agent")
    st.caption("Finds recent and reliable information.")

    st.markdown("### 2️⃣ Reader Agent")
    st.caption("Selects and extracts useful content.")

    st.markdown("### 3️⃣ Writer Chain")
    st.caption("Creates the research report.")

    st.markdown("### 4️⃣ Critic Chain")
    st.caption("Reviews the generated report.")


# ============================================================
# RUN PIPELINE
# ============================================================

if run_button:

    if not topic or not topic.strip():

        st.warning("⚠️ Please enter a research topic.")

    else:

        st.divider()

        st.subheader(f"🔎 Researching: {topic}")

        # ====================================================
        # STEP 1 - SEARCH
        # ====================================================

        with st.status(
            "🔎 Step 1 — Search Agent",
            expanded=True,
        ) as search_status:

            st.write("Searching for recent and reliable information...")

            try:

                search_agent = build_search_agent()

                search_result = search_agent.invoke({
                    "messages": [
                        (
                            "user",
                            f"""
                            Find recent, reliable and detailed
                            information about: {topic}
                            """
                        )
                    ]
                })

                search_results = search_result["messages"][-1].content

                search_status.update(
                    label="✅ Step 1 — Search completed",
                    state="complete",
                )

            except Exception as e:

                search_status.update(
                    label="❌ Step 1 — Search failed",
                    state="error",
                )

                st.error(f"Search Agent error: {e}")
                st.stop()


        # ====================================================
        # STEP 2 - READER
        # ====================================================

        with st.status(
            "📖 Step 2 — Reader Agent",
            expanded=True,
        ) as reader_status:

            st.write("Selecting the most relevant source...")

            try:

                reader_agent = build_reader_agent()

                reader_result = reader_agent.invoke({
                    "messages": [
                        (
                            "user",
                            f"""
                            Based on the following search results
                            about '{topic}', pick the most relevant
                            URL and scrape it for deeper content.

                            Search Results:

                            {search_results[:800]}
                            """
                        )
                    ]
                })

                scraped_content = reader_result["messages"][-1].content

                reader_status.update(
                    label="✅ Step 2 — Reader completed",
                    state="complete",
                )

            except Exception as e:

                reader_status.update(
                    label="❌ Step 2 — Reader failed",
                    state="error",
                )

                st.error(f"Reader Agent error: {e}")
                st.stop()


        # ====================================================
        # STEP 3 - WRITER
        # ====================================================

        with st.status(
            "✍️ Step 3 — Writer Chain",
            expanded=True,
        ) as writer_status:

            st.write("Creating the research report...")

            try:

                research = (
                    f"SEARCH RESULTS:\n"
                    f"{search_results}\n\n"
                    f"DETAILED SCRAPED CONTENT:\n"
                    f"{scraped_content}"
                )

                report = writer_chain.invoke({
                    "topic": topic,
                    "research": research,
                })

                writer_status.update(
                    label="✅ Step 3 — Report generated",
                    state="complete",
                )

            except Exception as e:

                writer_status.update(
                    label="❌ Step 3 — Writer failed",
                    state="error",
                )

                st.error(f"Writer error: {e}")
                st.stop()


        # ====================================================
        # STEP 4 - CRITIC
        # ====================================================

        with st.status(
            "🧐 Step 4 — Critic Chain",
            expanded=True,
        ) as critic_status:

            st.write("Reviewing the research report...")

            try:

                feedback = critic_chain.invoke({
                    "report": report
                })

                critic_status.update(
                    label="✅ Step 4 — Review completed",
                    state="complete",
                )

            except Exception as e:

                critic_status.update(
                    label="❌ Step 4 — Critic failed",
                    state="error",
                )

                st.error(f"Critic error: {e}")
                st.stop()


        # ====================================================
        # FINAL RESULTS
        # ====================================================

        st.divider()

        st.header("📊 Research Results")


        # ====================================================
        # FINAL REPORT
        # ====================================================

        st.subheader("📝 Final Research Report")

        with st.container(border=True):

            st.markdown(report)


        # Download button

        st.download_button(
            label="⬇️ Download Report",
            data=str(report),
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
        )


        # ====================================================
        # CRITIC
        # ====================================================

        st.subheader("🧐 Critic Review")

        with st.container(border=True):

            st.markdown(feedback)


        # ====================================================
        # RAW AGENT OUTPUTS
        # ====================================================

        st.subheader("🔍 Agent Details")

        with st.expander("🔎 View Search Agent Output"):

            st.write(search_results)

        with st.expander("📖 View Reader Agent Output"):

            st.write(scraped_content)


        # ====================================================
        # SUCCESS
        # ====================================================

        st.success(
            "🎉 Research pipeline completed successfully!"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Researcher AI • LangChain Multi-Agent Pipeline • Streamlit"
)
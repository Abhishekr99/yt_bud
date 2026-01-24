import pandas as pd
import streamlit as st

from knowledge_gaps import (
    evaluate_summaries_with_judge,
    fetch_gap_contexts,
    generate_enriched_notes,
    identify_knowledge_gaps,
)
from utility import (
    cache_transcript,
    extract_video_id,
    generate_notes,
    get_important_topics,
    get_or_create_vector_store,
    get_transcript,
    load_cached_transcript,
    rag_answer,
    translate_transcript,
)


@st.cache_data
def load_languages():
    df = pd.read_csv("language_code.csv")
    return dict(zip(df["Language"], df["Language_Code"]))


languages = load_languages()

# --- Sidebar (Inputs) ---
with st.sidebar:
    st.title("dYZ YT Buddy")
    st.markdown("Your AI-powered YouTube Companion")
    st.markdown("---")
    st.markdown("### Input Details")

    youtube_url = st.text_input(
        "YouTube URL", placeholder="https://www.youtube.com/watch-v=..."
    )

    selected_language_name = st.selectbox(
        "Video Language",
        options=sorted(list(languages.keys())),
        index=sorted(list(languages.keys())).index("English"),
    )
    language = languages[selected_language_name]

    task_option = st.radio(
        "Choose what you want to generate:", ["Chat with Video", "Notes For You"]
    )

    # Chat memory configuration
    memory_default = st.session_state.get("memory_turns", 2)
    memory_turns = (
        st.slider(
            "Chat memory (turns to remember)",
            min_value=0,
            max_value=8,
            value=memory_default,
            help="Number of previous chat turns (user + assistant) to carry into each answer.",
        )
        if task_option == "Chat with Video"
        else memory_default
    )
    st.session_state["memory_turns"] = memory_turns

    submit_button = st.button("Start Processing")
    st.markdown("---")

# --- Main Page ---
st.title("YouTube Content Synthesizer")
st.markdown("Paste a video link and select a task from the sidebar.")

# --- Processing Flow ---
if submit_button:
    if youtube_url and language:
        video_id = extract_video_id(youtube_url)
        if video_id:
            vectorstore = get_or_create_vector_store(video_id)
            full_transcript = load_cached_transcript(video_id)

            if not full_transcript:
                with st.spinner("Step 1/3: Fetching transcript..."):
                    full_transcript = get_transcript(video_id, language)

                    if language != "en":
                        with st.spinner(
                            "Step 1.5/3: Translating transcript to English..."
                        ):
                            full_transcript = translate_transcript(full_transcript)

                if full_transcript:
                    # Cache the normalized (English) transcript alongside the vector store.
                    cache_transcript(video_id, full_transcript)

            if full_transcript and not vectorstore:
                with st.spinner("Step 2/3: Preparing persistent video index..."):
                    vectorstore = get_or_create_vector_store(
                        video_id, transcript=full_transcript
                    )

            if vectorstore:
                st.session_state.vector_store = vectorstore
            if not full_transcript:
                st.error("Transcript unavailable. Please try again.")
                st.stop()

            if task_option == "Notes For You":
                with st.spinner("Step 2/5: Building a quick transcript summary..."):
                    base_summary = generate_notes(full_transcript)
                    st.subheader("Transcript Summary")
                    st.write(base_summary)
                    st.markdown("---")

                with st.spinner("Step 3/5: Extracting important topics..."):
                    important_topics = get_important_topics(full_transcript)
                    st.subheader("Important Topics")
                    st.write(important_topics)
                    st.markdown("---")

                with st.spinner("Step 4/5: Spotting knowledge gaps and fetching context..."):
                    gaps = identify_knowledge_gaps(full_transcript)
                    gap_contexts = fetch_gap_contexts(gaps)
                    st.subheader("Knowledge Gaps & Context")
                    if gap_contexts:
                        for item in gap_contexts:
                            st.markdown(f"**{item['term']}** - {item['reason']}")
                            st.caption(item["context"])
                        st.markdown("---")
                    else:
                        st.info("No knowledge gaps detected for this transcript.")

                with st.spinner("Step 5/5: Generating enriched notes..."):
                    notes = generate_enriched_notes(full_transcript, gap_contexts)
                    st.subheader("Enriched Notes")
                    st.write(notes)

                with st.spinner("Scoring summaries with LLM-as-judge..."):
                    eval_scores = evaluate_summaries_with_judge(
                        full_transcript, base_summary, notes
                    )
                    st.subheader("Summary Quality Check (LLM Judge)")
                    base_scores = eval_scores.get("base", {})
                    enriched_scores = eval_scores.get("enriched", {})

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Base Summary**")
                        if base_scores:
                            for metric, data in base_scores.items():
                                st.write(f"{metric.title()}: {data.get('score', '-')}")
                                st.caption(data.get("notes", ""))
                        else:
                            st.write("No scores available.")
                    with col2:
                        st.markdown("**Enriched Summary**")
                        if enriched_scores:
                            for metric, data in enriched_scores.items():
                                st.write(f"{metric.replace('_', ' ').title()}: {data.get('score', '-')}")
                                st.caption(data.get("notes", ""))
                        else:
                            st.write("No scores available.")

                st.success("Enriched summary, notes, and evaluation generated.")

            if task_option == "Chat with Video":
                if "vector_store" in st.session_state:
                    st.session_state.messages = []
                    st.success("Video is ready for chat.")
                else:
                    st.error("Vector store is not ready. Please try again.")

# chatbot session
if task_option == "Chat with Video" and "vector_store" in st.session_state:
    st.divider()
    st.subheader("Chat with Video")

    # Display the entire history
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # user_input
    prompt = st.chat_input("Ask me anything about the video.")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response = rag_answer(
                prompt,
                st.session_state.vector_store,
                chat_history=st.session_state.messages,
                history_turns=st.session_state.get("memory_turns", 2),
            )
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

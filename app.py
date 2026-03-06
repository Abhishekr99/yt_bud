import time

import pandas as pd
import streamlit as st

from knowledge_gaps import (
    evaluate_summaries_with_judge,
    fetch_gap_contexts,
    generate_enriched_notes,
    identify_knowledge_gaps,
)
from graph_rag import (
    compare_video_concepts,
    compare_videos_detailed,
    compare_with_reference,
    ensure_video_graph,
    graph_enabled,
    graph_rag_answer,
)
from experiment_logger import build_row, write_row
from rag_metrics import evaluate_rag_answers
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
    st.title("YT Bud")
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

    rag_mode = "Vanilla RAG"
    run_comparison_eval = False
    if task_option == "Chat with Video":
        rag_mode = st.selectbox(
            "Retrieval Mode",
            ["Vanilla RAG", "Graph RAG", "Compare (Both)"],
        )
        if rag_mode == "Compare (Both)":
            run_comparison_eval = st.checkbox(
                "Score answers with LLM judge", value=True
            )

    enable_graph_analysis = False
    reference_terms = ""
    compare_video_url = ""
    comparison_mode = "Detailed (flow + prereq + bridge)"
    log_metrics = False
    csv_path = "metrics_log.csv"
    if task_option == "Notes For You":
        enable_graph_analysis = st.checkbox("Enable Graph Gap Analysis (Neo4j)")
        if enable_graph_analysis:
            reference_terms = st.text_area(
                "Reference concepts (one per line, optional)",
                height=120,
            )
            comparison_mode = st.selectbox(
                "Comparison Mode",
                ["Detailed (flow + prereq + bridge)", "Simple (set difference)"],
            )
            compare_video_url = st.text_input(
                "Second video URL for graph gap analysis (optional)",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            log_metrics = st.checkbox("Log comparison metrics to CSV")
            if log_metrics:
                csv_path = st.text_input(
                    "CSV path for metrics log",
                    value="metrics_log.csv",
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

            st.session_state.video_id = video_id
            st.session_state.transcript = full_transcript

            graph_needed = (
                (task_option == "Chat with Video" and rag_mode != "Vanilla RAG")
                or (task_option == "Notes For You" and enable_graph_analysis)
            )
            if graph_needed:
                if not graph_enabled():
                    st.error(
                        "Neo4j is not configured. Set NEO4J_URI, NEO4J_USERNAME, "
                        "and NEO4J_PASSWORD in your .env."
                    )
                    st.session_state.graph_ready = False
                else:
                    with st.spinner("Preparing Neo4j graph for this video..."):
                        try:
                            graph_status = ensure_video_graph(
                                video_id, full_transcript, language
                            )
                        except Exception as exc:
                            st.error(f"Graph build failed: {exc}")
                            st.session_state.graph_ready = False
                        else:
                            st.session_state.graph_ready = bool(
                                graph_status.get("ready")
                            )
                            st.session_state.graph_video_id = video_id

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

                if enable_graph_analysis:
                    st.markdown("---")
                    st.subheader("Graph Gap Analysis (Neo4j)")

                    if not graph_enabled():
                        st.error(
                            "Neo4j is not configured. Set NEO4J_URI, NEO4J_USERNAME, "
                            "and NEO4J_PASSWORD in your .env."
                        )
                    elif not st.session_state.get("graph_ready", False):
                        st.info("Graph is not ready. Re-run with Neo4j configured.")
                    else:
                        reference_list = [
                            line.strip()
                            for line in reference_terms.splitlines()
                            if line.strip()
                        ]
                        if reference_list:
                            ref_result = compare_with_reference(video_id, reference_list)
                            st.markdown("**Reference Graph Coverage**")
                            st.write(
                                f"Missing concepts: {len(ref_result['missing'])} | "
                                f"Covered: {len(ref_result['covered'])}"
                            )
                            if ref_result["missing"]:
                                st.markdown("Missing from video")
                                st.write(ref_result["missing"])
                            if ref_result["covered"]:
                                st.markdown("Covered in video")
                                st.write(ref_result["covered"])

                        if compare_video_url:
                            other_video_id = extract_video_id(compare_video_url)
                            if not other_video_id:
                                st.error("Second video URL is invalid.")
                            else:
                                other_transcript = load_cached_transcript(other_video_id)
                                if not other_transcript:
                                    with st.spinner(
                                        "Fetching transcript for comparison video..."
                                    ):
                                        other_transcript = get_transcript(
                                            other_video_id, language
                                        )
                                        if language != "en" and other_transcript:
                                            other_transcript = translate_transcript(
                                                other_transcript
                                            )
                                        if other_transcript:
                                            cache_transcript(
                                                other_video_id, other_transcript
                                            )
                                if not other_transcript:
                                    st.error(
                                        "Transcript unavailable for comparison video."
                                    )
                                else:
                                    with st.spinner(
                                        "Preparing Neo4j graph for comparison video..."
                                    ):
                                        ensure_video_graph(
                                            other_video_id,
                                            other_transcript,
                                            language,
                                        )
                                    st.markdown("**Graph Gap vs. Comparison Video**")
                                    if comparison_mode == "Simple (set difference)":
                                        comparison = compare_video_concepts(
                                            video_id, other_video_id
                                        )
                                        st.write(
                                            f"Only in current: {len(comparison['only_in_a'])} | "
                                            f"Only in comparison: {len(comparison['only_in_b'])} | "
                                            f"Shared: {len(comparison['shared'])}"
                                        )
                                        if comparison["only_in_a"]:
                                            st.markdown("Only in current video")
                                            st.write(comparison["only_in_a"])
                                        if comparison["only_in_b"]:
                                            st.markdown("Only in comparison video")
                                            st.write(comparison["only_in_b"])
                                        if log_metrics:
                                            st.info(
                                                "Metrics logging is available only in Detailed mode."
                                            )
                                    else:
                                        comparison = compare_videos_detailed(
                                            video_id, other_video_id
                                        )
                                        summary = comparison.get("summary", {})
                                        st.write(
                                            "Concepts A: "
                                            f"{summary.get('concepts_a', 0)} | "
                                            "Concepts B: "
                                            f"{summary.get('concepts_b', 0)} | "
                                            "Shared: "
                                            f"{summary.get('shared', 0)} | "
                                            "Jaccard: "
                                            f"{summary.get('jaccard', 0.0)}"
                                        )
                                        if summary.get("topic_shift_flag"):
                                            st.warning(
                                                summary.get(
                                                    "topic_shift_reason",
                                                    "Topic shift detected.",
                                                )
                                            )
                                        else:
                                            st.info(
                                                summary.get(
                                                    "topic_shift_reason",
                                                    "No topic shift detected.",
                                                )
                                            )

                                        with st.expander("Metrics"):
                                            st.write(
                                                "Learning Continuity Score (LCS): "
                                                f"{summary.get('LCS', 0.0)} "
                                                f"({summary.get('lcs_label', '')})"
                                            )
                                            st.write(
                                                "Sub-scores: "
                                                f"Concept={summary.get('S_concept', 0.0)}, "
                                                f"Bridge={summary.get('S_bridge', 0.0)}, "
                                                f"Prereq={summary.get('S_prereq', 0.0)}, "
                                                f"Sequence={summary.get('S_sequence', 0.0)}"
                                            )
                                            st.write(
                                                "Baselines: "
                                                f"TF-IDF={summary.get('tfidf_similarity_full', 0.0)}, "
                                                f"Embedding={summary.get('embedding_similarity_full', 0.0)}"
                                            )
                                            st.write(
                                                "Rates: "
                                                f"New={summary.get('new_concept_rate', 0.0)}, "
                                                f"PrereqGap={summary.get('prereq_gap_rate', 0.0)}, "
                                                f"OrderViolation={summary.get('order_violation_rate', 0.0)}"
                                            )

                                        with st.expander(
                                            "New concepts introduced in comparison video"
                                        ):
                                            new_in_b = comparison.get("new_in_b", [])
                                            if not new_in_b:
                                                st.write("No new concepts found.")
                                            else:
                                                st.table(
                                                    [
                                                        {
                                                            "concept": item["concept"],
                                                            "first_chunk": item[
                                                                "first_seen"
                                                            ].get("chunk_index"),
                                                            "importance": item[
                                                                "importance"
                                                            ],
                                                        }
                                                        for item in new_in_b
                                                    ]
                                                )
                                                for item in new_in_b[:12]:
                                                    first_seen = item.get(
                                                        "first_seen", {}
                                                    )
                                                    st.caption(
                                                        f"{item['concept']} | "
                                                        f"chunk {first_seen.get('chunk_index')}"
                                                    )
                                                    if first_seen.get("snippet"):
                                                        st.write(
                                                            first_seen.get("snippet")
                                                        )

                                        with st.expander("Prerequisite gaps"):
                                            prereq_gaps = comparison.get(
                                                "prereq_gaps", []
                                            )
                                            if not prereq_gaps:
                                                st.write("No prerequisite gaps found.")
                                            else:
                                                for gap in prereq_gaps:
                                                    st.markdown(
                                                        f"**{gap['advanced_concept']}**"
                                                    )
                                                    st.write(
                                                        "Missing prerequisites: "
                                                        + ", ".join(
                                                            gap.get(
                                                                "missing_prerequisites",
                                                                [],
                                                            )
                                                        )
                                                    )
                                                    st.caption(
                                                        gap.get("why_gap", "")
                                                    )
                                                    for ev in gap.get(
                                                        "evidence_in_b", []
                                                    ):
                                                        st.caption(
                                                            f"chunk {ev.get('chunk_index')}: "
                                                            f"{ev.get('snippet')}"
                                                        )

                                        with st.expander("Order violations in video B"):
                                            order_violations = comparison.get(
                                                "order_violations_b", []
                                            )
                                            if not order_violations:
                                                st.write(
                                                    "No order violations detected."
                                                )
                                            else:
                                                st.table(
                                                    [
                                                        {
                                                            "prerequisite": item[
                                                                "prerequisite"
                                                            ],
                                                            "dependent": item[
                                                                "dependent"
                                                            ],
                                                            "prereq_first": item[
                                                                "prereq_first_chunk"
                                                            ],
                                                            "dependent_first": item[
                                                                "dependent_first_chunk"
                                                            ],
                                                        }
                                                        for item in order_violations
                                                    ]
                                                )
                                                for item in order_violations[:8]:
                                                    if item.get("evidence"):
                                                        st.caption(
                                                            item.get("evidence")
                                                        )

                                        with st.expander("Bridge gaps / topic jump"):
                                            bridge_gaps = comparison.get(
                                                "bridge_gaps", []
                                            )
                                            if not bridge_gaps:
                                                st.write("No bridge gaps found.")
                                            else:
                                                for gap in bridge_gaps:
                                                    st.write(
                                                        "End of A topics: "
                                                        + ", ".join(
                                                            gap.get(
                                                                "from_video_end_topic",
                                                                [],
                                                            )
                                                        )
                                                    )
                                                    st.write(
                                                        "Start of B topics: "
                                                        + ", ".join(
                                                            gap.get(
                                                                "to_video_start_topic",
                                                                [],
                                                            )
                                                        )
                                                    )
                                                    if gap.get(
                                                        "missing_bridge_concepts"
                                                    ):
                                                        st.write(
                                                            "Missing bridges: "
                                                            + ", ".join(
                                                                gap.get(
                                                                    "missing_bridge_concepts",
                                                                    [],
                                                                )
                                                            )
                                                        )
                                                    if gap.get(
                                                        "suggested_bridge_query"
                                                    ):
                                                        st.caption(
                                                            gap.get(
                                                                "suggested_bridge_query"
                                                            )
                                                        )

                                        with st.expander("Relation mismatches"):
                                            mismatches = comparison.get(
                                                "relation_mismatches", []
                                            )
                                            if not mismatches:
                                                st.write(
                                                    "No relation mismatches found."
                                                )
                                            else:
                                                st.table(
                                                    [
                                                        {
                                                            "source": item["source"],
                                                            "relation": item[
                                                                "relation"
                                                            ],
                                                            "target": item["target"],
                                                            "present_in": item[
                                                                "present_in"
                                                            ],
                                                        }
                                                        for item in mismatches
                                                    ]
                                                )
                                                for item in mismatches[:10]:
                                                    if item.get("evidence_a"):
                                                        st.caption(
                                                            f"A: {item.get('evidence_a')}"
                                                        )
                                                    if item.get("evidence_b"):
                                                        st.caption(
                                                            f"B: {item.get('evidence_b')}"
                                                        )
                                        if log_metrics:
                                            try:
                                                row = build_row(
                                                    video_id, other_video_id, comparison
                                                )
                                                write_row(csv_path, row)
                                                st.caption(
                                                    f"Metrics logged to {csv_path}"
                                                )
                                            except Exception as exc:
                                                st.error(
                                                    f"Failed to log metrics: {exc}"
                                                )

                st.success("Enriched summary, notes, and evaluation generated.")

            if task_option == "Chat with Video":
                if "vector_store" in st.session_state:
                    st.session_state.messages = []
                    if rag_mode != "Vanilla RAG" and not st.session_state.get(
                        "graph_ready", False
                    ):
                        st.error("Neo4j graph is not ready. Check configuration.")
                    else:
                        st.success("Video is ready for chat.")
                else:
                    st.error("Vector store is not ready. Please try again.")

# chatbot session
if task_option == "Chat with Video" and "vector_store" in st.session_state:
    st.divider()
    st.subheader("Chat with Video")
    current_video_id = st.session_state.get("video_id")

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
            if rag_mode == "Vanilla RAG":
                start = time.perf_counter()
                response, context_docs = rag_answer(
                    prompt,
                    st.session_state.vector_store,
                    chat_history=st.session_state.messages,
                    history_turns=st.session_state.get("memory_turns", 2),
                    return_context=True,
                )
                latency = time.perf_counter() - start
                st.write(response)
                st.caption(
                    f"Vanilla RAG: {len(context_docs)} chunks | {latency:.2f}s"
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            elif rag_mode == "Graph RAG":
                if not st.session_state.get("graph_ready", False):
                    st.error("Neo4j graph is not ready for this video.")
                elif st.session_state.get("graph_video_id") != current_video_id:
                    st.error("Graph is out of sync. Re-run processing.")
                else:
                    start = time.perf_counter()
                    response, stats = graph_rag_answer(
                        prompt,
                        current_video_id,
                        chat_history=st.session_state.messages,
                        history_turns=st.session_state.get("memory_turns", 2),
                    )
                    latency = time.perf_counter() - start
                    st.write(response)
                    st.caption(
                        "Graph RAG: "
                        f"{stats.get('concepts', 0)} concepts | "
                        f"{stats.get('chunks', 0)} chunks | "
                        f"{latency:.2f}s"
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
            else:
                if not st.session_state.get("graph_ready", False):
                    st.error("Neo4j graph is not ready for this video.")
                elif st.session_state.get("graph_video_id") != current_video_id:
                    st.error("Graph is out of sync. Re-run processing.")
                else:
                    start = time.perf_counter()
                    vanilla_response, vanilla_docs = rag_answer(
                        prompt,
                        st.session_state.vector_store,
                        chat_history=st.session_state.messages,
                        history_turns=st.session_state.get("memory_turns", 2),
                        return_context=True,
                    )
                    vanilla_latency = time.perf_counter() - start

                    start = time.perf_counter()
                    graph_response, graph_stats = graph_rag_answer(
                        prompt,
                        current_video_id,
                        chat_history=st.session_state.messages,
                        history_turns=st.session_state.get("memory_turns", 2),
                    )
                    graph_latency = time.perf_counter() - start

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Vanilla RAG**")
                        st.write(vanilla_response)
                        st.caption(
                            f"{len(vanilla_docs)} chunks | {vanilla_latency:.2f}s"
                        )
                    with col2:
                        st.markdown("**Graph RAG**")
                        st.write(graph_response)
                        st.caption(
                            f"{graph_stats.get('concepts', 0)} concepts | "
                            f"{graph_stats.get('chunks', 0)} chunks | "
                            f"{graph_latency:.2f}s"
                        )

                    if run_comparison_eval and st.session_state.get("transcript"):
                        eval_scores = evaluate_rag_answers(
                            prompt,
                            st.session_state.transcript,
                            vanilla_response,
                            graph_response,
                        )
                        st.markdown("**Answer Evaluation**")
                        if "raw_response" in eval_scores:
                            st.write(eval_scores["raw_response"])
                        else:
                            rows = []
                            for metric in [
                                "groundedness",
                                "relevance",
                                "completeness",
                                "clarity",
                            ]:
                                rows.append(
                                    {
                                        "metric": metric,
                                        "vanilla": eval_scores["vanilla"][metric][
                                            "score"
                                        ],
                                        "graph": eval_scores["graph"][metric]["score"],
                                    }
                                )
                            st.table(rows)
                            st.caption(
                                f"Winner: {eval_scores.get('winner')} | "
                                f"{eval_scores.get('rationale')}"
                            )

                    combined = (
                        "[Vanilla RAG]\n"
                        f"{vanilla_response}\n\n"
                        "[Graph RAG]\n"
                        f"{graph_response}"
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": combined}
                    )

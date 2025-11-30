import streamlit as st
from streamlit import spinner
from streamlit.web.server.server import server_port_is_manually_set
import pandas as pd

from utility import (
     extract_video_id,
     get_transcript,
     translate_transcript,
     get_important_topics,
     create_chunks,
     create_vector_store,
     rag_answer
)
from knowledge_gaps import (
     identify_knowledge_gaps,
     fetch_gap_contexts,
     generate_enriched_notes
)

# Load language codes
@st.cache_data
def load_languages():
    df = pd.read_csv("language_code.csv")
    return dict(zip(df['Language'], df['Language_Code']))

languages = load_languages()

# --- Sidebar (Inputs) ---
with st.sidebar:
    st.title("🎬 YT Buddy")
    st.markdown("Your AI-powered YouTube Companion")
    st.markdown("---")
    #st.markdown("Transform any YouTube video into key topics, a podcast, or a chatbot.")
    st.markdown("### Input Details")

    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    
    selected_language_name = st.selectbox(
        "Video Language",
        options=sorted(list(languages.keys())),
        index=sorted(list(languages.keys())).index("English")
    )
    language = languages[selected_language_name]
    print('language:', language)
    task_option = st.radio(
        "Choose what you want to generate:",
        ["Chat with Video", "Notes For You"]
    )

    submit_button = st.button("✨ Start Processing")
    st.markdown("---")

# --- Main Page ---
st.title("YouTube Content Synthesizer")
st.markdown("Paste a video link and select a task from the sidebar.")

# --- Processing Flow ---
if submit_button:
    if youtube_url and language:
        video_id= extract_video_id(youtube_url)
        if video_id:
            with spinner("Step 1/3 : Fetching Transcript....."):
                full_transcript= get_transcript(video_id, language)

                if language!="en":
                    with spinner("Step 1.5/3 : Translating Transcript into English, This may take few moments......"):
                        full_transcript= translate_transcript(full_transcript)


            if task_option=="Notes For You":
                with spinner("Step 2/4: Extracting important Topics..."):
                    import_topics= get_important_topics(full_transcript)
                    st.subheader("Important Topics")
                    st.write(import_topics)
                    st.markdown("---")

                with spinner("Step 3/4: Spotting knowledge gaps and fetching context..."):
                    gaps = identify_knowledge_gaps(full_transcript)
                    print('gaps:', gaps)
                    gap_contexts = fetch_gap_contexts(gaps)
                    print('gap_contexts:', gap_contexts)
                    st.subheader("Knowledge Gaps & Context")
                    if gap_contexts:
                        for item in gap_contexts:
                            st.markdown(f"**{item['term']}** — {item['reason']}")
                            st.caption(item["context"])
                        st.markdown("---")
                    else:
                        st.info("No knowledge gaps detected for this transcript.")

                with spinner("Step 4/4 : Generating enriched notes for you."):
                    notes= generate_enriched_notes(full_transcript, gap_contexts)
                    st.subheader("Enriched Notes")
                    st.write(notes)

                st.success("Enriched summary and notes generated.")

            if task_option == "Chat with Video":
                with st.spinner("Step 2/3: Creating chunks and vector store...."):
                    chunks = create_chunks(full_transcript)
                    vectorstore = create_vector_store(chunks)
                    st.session_state.vector_store = vectorstore
                st.session_state.messages=[]
                st.success('Video is ready for chat.....')

# chatbot session
if task_option=="Chat with Video" and "vector_store" in st.session_state:
    st.divider()
    st.subheader("Chat with Video")

    # Display the entire history
    for message in st.session_state.get('messages',[]):
        with st.chat_message(message['role']):
            st.write(message['content'])

    # user_input
    prompt= st.chat_input("Ask me anything about the video.")
    if prompt:
        st.session_state.messages.append({'role':'user','content':prompt})
        with st.chat_message('user'):
            st.write(prompt)

        with st.chat_message('assistant'):
           response= rag_answer(prompt,st.session_state.vector_store)
           st.write(response)
        st.session_state.messages.append({'role': 'assistant', 'content':response})

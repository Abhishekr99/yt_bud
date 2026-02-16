import math
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

VECTOR_STORE_ROOT = Path(".vectorstore")
TRANSCRIPT_CACHE_FILENAME = "transcript.txt"
DIRECT_LLM_CHAR_LIMIT = 30000
SECTION_SUMMARY_CHAR_LIMIT = 12000
SECTION_SUMMARY_OVERLAP = 600

PREVIEW_PATTERNS = [
    "in this video",
    "today we will",
    "we will cover",
    "we are going to",
    "overview of",
]

EXPLANATION_PATTERNS = [
    "is defined as",
    "means",
    "refers to",
    "we call this",
    "the idea is",
    "works like",
    "let us understand",
    "for example",
    "in other words",
    "space complexity",
    "time complexity",
]

GENERIC_CONCEPTS = {
    "input",
    "output",
    "data",
    "file",
    "example",
    "implementation",
    "code",
    "program",
    "overview",
}


# Function to extract video ID from a YouTube URL (Helper Function)
def extract_video_id(url):
    """
    Extract the YouTube video ID from any valid YouTube URL.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    st.error("Invalid YouTube URL. Please enter a valid video link.")
    return None


# function to get transcript from the video.
def get_transcript(video_id, language):
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript = ytt_api.fetch(video_id, languages=[language])
        full_transcript = " ".join([i.text for i in transcript])
        time.sleep(10)  # To avoid hitting rate limits
        return full_transcript
    except Exception as e:
        st.error(f"Error fetching video: {e}")
        return ""


# initialize the gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)


def get_vector_store_dir(video_id: str) -> Path:
    return VECTOR_STORE_ROOT / video_id


def normalize_concept(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r"[^a-z0-9\s]", "", n)
    n = re.sub(r"\s+", " ", n)
    if len(n) > 3 and n.endswith("s"):
        n = n[:-1]
    return n


def is_intro_like(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in PREVIEW_PATTERNS)


def explanation_score(text: str) -> int:
    t = text.lower()
    return sum(1 for p in EXPLANATION_PATTERNS if p in t)


def is_generic(concept_key: str) -> bool:
    return concept_key in GENERIC_CONCEPTS


@st.cache_resource(show_spinner=False)
def _get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def _average_vector(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    length = len(vectors[0])
    sums = [0.0] * length
    for vec in vectors:
        for idx, val in enumerate(vec):
            sums[idx] += float(val)
    return [val / len(vectors) for val in sums]


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def embedding_similarity(texts_a: List[str], texts_b: List[str]) -> float:
    if not texts_a or not texts_b:
        return -1.0
    model = _get_embedding_model()
    try:
        embeddings = model.embed_documents(texts_a + texts_b)
    except Exception:
        return -1.0
    if not embeddings or len(embeddings) < len(texts_a) + len(texts_b):
        return -1.0
    vec_a = _average_vector(embeddings[: len(texts_a)])
    vec_b = _average_vector(embeddings[len(texts_a) :])
    return _cosine_similarity(vec_a, vec_b)


def get_transcript_cache_path(video_id: str) -> Path:
    return get_vector_store_dir(video_id) / TRANSCRIPT_CACHE_FILENAME


def load_cached_transcript(video_id: str) -> Optional[str]:
    cache_path = get_transcript_cache_path(video_id)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    return None


def cache_transcript(video_id: str, transcript: str) -> None:
    store_dir = get_vector_store_dir(video_id)
    store_dir.mkdir(parents=True, exist_ok=True)
    get_transcript_cache_path(video_id).write_text(transcript, encoding="utf-8")


def vector_store_exists(video_id: str) -> bool:
    store_dir = get_vector_store_dir(video_id)
    return (store_dir / "chroma.sqlite3").exists()


def load_vector_store(video_id: str) -> Chroma:
    embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    kwargs = {
        "persist_directory": str(get_vector_store_dir(video_id)),
        "embedding_function": embedding,
        "collection_name": video_id,
    }
    try:
        return Chroma(**kwargs)
    except TypeError:
        kwargs.pop("collection_name", None)
        return Chroma(**kwargs)


# function to translate the transcript into english.
def translate_transcript(transcript):
    try:
        prompt = ChatPromptTemplate.from_template(
            """
        You are an expert translator with deep cultural and linguistic knowledge.
        I will provide you with a transcript. Your task is to translate it into English with absolute accuracy, preserving:
        - Full meaning and context (no omissions, no additions).
        - Tone and style (formal/informal, emotional/neutral as in original).
        - Nuances, idioms, and cultural expressions (adapt appropriately while keeping intent).
        - Speaker's voice (same perspective, no rewriting into third-person).
        Do not summarize or simplify. The translation should read naturally in the target language but stay as close as possible to the original intent.

        Transcript:
        {transcript}
        """
        )

        chain = prompt | llm
        response = chain.invoke({"transcript": transcript})
        return response.content
    except Exception as e:
        st.error(f"Error translating video: {e}")
        return ""


# function to get important topics
def get_important_topics(transcript):
    try:
        if len(transcript) > DIRECT_LLM_CHAR_LIMIT:
            transcript = compress_transcript_for_topics(transcript)

        prompt = ChatPromptTemplate.from_template(
            """
               You are an assistant that extracts the 5 most important topics discussed in a video transcript or summary.

               Rules:
               - Summarize into exactly 5 major points.
               - Each point should represent a key topic or concept, not small details.
               - Keep wording concise and focused on the technical content.
               - Do not phrase them as questions or opinions.
               - Output should be a numbered list.
               - show only points that are discussed in the transcript.
               Here is the transcript:
               {transcript}
               """
        )

        chain = prompt | llm
        response = chain.invoke({"transcript": transcript})
        return response.content

    except Exception as e:
        st.error(f"Error fetching topics: {e}")
        return ""


def _select_chunk_params(text_length: int) -> Tuple[int, int]:
    if text_length > 300000:
        return 4000, 400
    if text_length > 150000:
        return 6000, 600
    if text_length > 80000:
        return 8000, 800
    return 10000, 1000


def _select_section_params(chunk_size: int) -> Tuple[int, int]:
    section_size = max(24000, min(chunk_size * 3, 40000))
    section_overlap = max(600, int(section_size * 0.05))
    return section_size, section_overlap


def _split_text_to_documents(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: Optional[dict] = None,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = splitter.create_documents([text])
    if metadata:
        for doc in docs:
            doc.metadata.update(metadata)
    return docs


def _hierarchical_reduce(
    transcript: str,
    section_prompt: ChatPromptTemplate,
    merge_prompt: ChatPromptTemplate,
) -> str:
    sections = _split_text_to_documents(
        transcript, SECTION_SUMMARY_CHAR_LIMIT, SECTION_SUMMARY_OVERLAP
    )
    if not sections:
        return transcript

    summaries = []
    chain = section_prompt | llm
    for idx, section in enumerate(sections, start=1):
        response = chain.invoke(
            {"section": section.page_content, "index": idx, "total": len(sections)}
        )
        summaries.append(f"Section {idx}:\n{response.content.strip()}")

    merged = (merge_prompt | llm).invoke({"summaries": "\n\n".join(summaries)})
    return merged.content.strip()


def compress_transcript_for_topics(transcript: str) -> str:
    if len(transcript) <= DIRECT_LLM_CHAR_LIMIT:
        return transcript

    section_prompt = ChatPromptTemplate.from_template(
        """
        Summarize this transcript section into concise bullet points.
        - Keep key terms, names, and concepts for retrieval.
        - Limit to 6-8 bullets.

        Section {index} of {total}:
        {section}
        """
    )
    merge_prompt = ChatPromptTemplate.from_template(
        """
        Combine the section bullets into a compact summary.
        - Keep it under 20 bullets.
        - Preserve key terms and major ideas.

        Section summaries:
        {summaries}
        """
    )
    return _hierarchical_reduce(transcript, section_prompt, merge_prompt)


def compress_transcript_for_gaps(transcript: str) -> str:
    if len(transcript) <= DIRECT_LLM_CHAR_LIMIT:
        return transcript

    section_prompt = ChatPromptTemplate.from_template(
        """
        Extract terms and concepts from this transcript section that may need explanation.
        - Focus on acronyms, proper nouns, jargon, and niche concepts.
        - Provide a short reason per item.
        - Limit to 6-8 bullets.

        Section {index} of {total}:
        {section}
        """
    )
    merge_prompt = ChatPromptTemplate.from_template(
        """
        Combine the section lists into a compact transcript overview focused on key terms.
        - Keep it concise and scannable.
        - Preserve term names and reasons.

        Section summaries:
        {summaries}
        """
    )
    return _hierarchical_reduce(transcript, section_prompt, merge_prompt)


# FUNCTION TO GET NOTES FROM THE VIDEO
def generate_notes(transcript):
    try:
        if len(transcript) > DIRECT_LLM_CHAR_LIMIT:
            section_prompt = ChatPromptTemplate.from_template(
                """
                Summarize this transcript section into concise notes.
                - Use bullet points with short sentences.
                - Keep the key facts, terms, and examples.
                - Do not introduce new information.

                Section {index} of {total}:
                {section}
                """
            )
            merge_prompt = ChatPromptTemplate.from_template(
                """
                Merge the section notes into clean, structured notes.
                - Group related bullets under short headings.
                - Remove duplication and keep wording concise.
                - Do not add information not present in the notes.

                Section notes:
                {summaries}
                """
            )
            return _hierarchical_reduce(transcript, section_prompt, merge_prompt)

        prompt = ChatPromptTemplate.from_template(
            """
                You are an AI note-taker. Your task is to read the following YouTube video transcript 
                and produce well-structured, concise notes.

                Requirements:
                - Present the output as **bulleted points**, grouped into clear sections.
                - Highlight key takeaways, important facts, and examples.
                - Use **short, clear sentences** (no long paragraphs).
                - If the transcript includes multiple themes, organize them under **subheadings**.
                - Do not add information that is not present in the transcript.

                Here is the transcript:
                {transcript}
                """
        )

        chain = prompt | llm
        response = chain.invoke({"transcript": transcript})
        return response.content

    except Exception as e:
        st.error(f"Error generating notes: {e}")
        return ""


# FUNCTION TO CREATE CHUNKS
def create_chunks(transcript):
    chunk_size, chunk_overlap = _select_chunk_params(len(transcript))
    return _split_text_to_documents(transcript, chunk_size, chunk_overlap)


# function to create embedding and store it into an vector space.
def create_vector_store(
    docs: List[Document],
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    try:
        vector_store = Chroma.from_documents(
            docs,
            embedding,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    except TypeError:
        vector_store = Chroma.from_documents(
            docs, embedding, persist_directory=persist_directory
        )
    if persist_directory and hasattr(vector_store, "persist"):
        vector_store.persist()
    return vector_store


def _summarize_sections_for_retrieval(
    transcript: str, section_size: int, section_overlap: int
) -> List[Document]:
    sections = _split_text_to_documents(
        transcript, section_size, section_overlap, metadata=None
    )
    if not sections:
        return []

    prompt = ChatPromptTemplate.from_template(
        """
        Create a compact outline of this transcript section for semantic search.
        - 4-6 bullet points.
        - Include key terms, names, and concepts.
        - Keep it concise and factual.

        Section {index} of {total}:
        {section}
        """
    )
    chain = prompt | llm
    summaries: List[Document] = []
    for idx, section in enumerate(sections, start=1):
        response = chain.invoke(
            {"section": section.page_content, "index": idx, "total": len(sections)}
        )
        summaries.append(
            Document(
                page_content=response.content.strip(),
                metadata={"level": "summary", "section_index": idx - 1},
            )
        )
    return summaries


def _build_hierarchical_documents(transcript: str) -> List[Document]:
    chunk_size, chunk_overlap = _select_chunk_params(len(transcript))
    section_size, section_overlap = _select_section_params(chunk_size)

    if len(transcript) <= section_size:
        return _split_text_to_documents(
            transcript,
            chunk_size,
            chunk_overlap,
            metadata={"level": "detail", "section_index": 0},
        )

    # Store a summary layer plus detailed chunks to enable two-stage retrieval.
    summary_docs = _summarize_sections_for_retrieval(
        transcript, section_size, section_overlap
    )
    detailed_docs: List[Document] = []
    sections = _split_text_to_documents(
        transcript, section_size, section_overlap, metadata=None
    )
    for idx, section in enumerate(sections):
        detailed_docs.extend(
            _split_text_to_documents(
                section.page_content,
                chunk_size,
                chunk_overlap,
                metadata={"level": "detail", "section_index": idx},
            )
        )

    return summary_docs + detailed_docs


def build_vector_store(video_id: str, transcript: str) -> Chroma:
    store_dir = get_vector_store_dir(video_id)
    store_dir.mkdir(parents=True, exist_ok=True)
    docs = _build_hierarchical_documents(transcript)
    return create_vector_store(
        docs,
        persist_directory=str(store_dir),
        collection_name=video_id,
    )


def get_or_create_vector_store(
    video_id: str, transcript: Optional[str] = None
) -> Optional[Chroma]:
    if vector_store_exists(video_id):
        return load_vector_store(video_id)
    if transcript:
        return build_vector_store(video_id, transcript)
    return None


def _retrieve_context(question: str, vectorstore: Chroma, k: int = 4) -> List[Document]:
    summary_hits: List[Document] = []
    try:
        summary_hits = vectorstore.similarity_search(
            question, k=2, filter={"level": "summary"}
        )
    except Exception:
        summary_hits = []

    if summary_hits:
        # First narrow to relevant sections, then retrieve detailed chunks.
        section_ids = sorted(
            {
                doc.metadata.get("section_index")
                for doc in summary_hits
                if doc.metadata.get("section_index") is not None
            }
        )
        if section_ids:
            try:
                detail_hits = vectorstore.similarity_search(
                    question,
                    k=k,
                    filter={"level": "detail", "section_index": {"$in": section_ids}},
                )
                if detail_hits:
                    return detail_hits
            except Exception:
                pass
        return summary_hits

    return vectorstore.similarity_search(question, k=k)


# RAG FUNCTION WITH CHAT HISTORY SUPPORT
def rag_answer(
    question, vectorstore, chat_history=None, history_turns=2, return_context=False
):
    """
    Answer user questions using retrieved transcript context and recent chat history.
    - chat_history: list of {"role": "...", "content": "..."} entries from the chat UI.
    - history_turns: how many prior exchanges (user + assistant) to include.
    """
    chat_history = chat_history or []
    turns_to_keep = max(history_turns, 0) * 2  # each turn is user + assistant
    recent_messages = chat_history[-turns_to_keep:] if turns_to_keep else []

    formatted_history = []
    for msg in recent_messages:
        role = msg.get("role", "").lower()
        if role not in ("user", "assistant"):
            continue
        speaker = "User" if role == "user" else "Assistant"
        formatted_history.append(f"{speaker}: {msg.get('content', '').strip()}")
    history_text = "\n".join(formatted_history).strip() or "No prior chat turns."

    results = _retrieve_context(question, vectorstore, k=4)
    context_text = "\n".join([i.page_content for i in results])

    prompt = ChatPromptTemplate.from_template(
        """
                You are a precise, helpful assistant for chatting about a video.
                - Use both the retrieved transcript context and the recent chat history to stay consistent.
                - Do NOT add greetings; reply directly to the user's request.
                - Answer ONLY using the provided context/history; avoid outside knowledge.
                - If the answer is not in the context, reply:
                  "I couldn't find that information in the transcript. Could you rephrase or ask something else?"
                - Keep responses concise and easy to skim.

                Recent Chat History:
                {history}

                Transcript Context:
                {context}

                User Question:
                {question}

                Answer:
                """
    )

    chain = prompt | llm
    response = chain.invoke(
        {"context": context_text, "question": question, "history": history_text}
    )

    if return_context:
        return response.content, results
    return response.content

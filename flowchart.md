```mermaid
flowchart TD
    %% Entry and setup
    A[User opens sidebar<br/>- Paste YouTube URL<br/>- Choose language<br/>- Pick task] --> B[Load language map from CSV<br/>(st.cache_data)]
    B --> C[Extract video_id from URL]
    C --> D[Fetch transcript via YouTubeTranscriptApi]
    D --> E{Transcript language<br/>is English?}
    E -- Yes --> F[Use transcript as-is]
    E -- No --> G[Translate transcript to English<br/>(Gemini LLM)]
    F --> H[Normalized transcript]
    G --> H
    H --> I{Task selection}

    %% Notes For You branch
    subgraph N[Notes For You]
        direction TB
        N1[LLM: generate_notes<br/>(quick summary)]
        N2[LLM: get_important_topics]
        N3[LLM: identify_knowledge_gaps]
        N4[Fetch gap context<br/>- Wikipedia REST<br/>- Fallback: LLM explain]
        N5[LLM: generate_enriched_notes<br/>with gap context]
        N6[LLM judge: score base vs enriched summaries]
        N1 --> N2 --> N3 --> N4 --> N5 --> N6
    end

    %% Chat with Video branch (RAG)
    subgraph C[Chat with Video]
        direction TB
        C1[Chunk transcript<br/>(RecursiveCharacterTextSplitter)]
        C2[Embed chunks<br/>(GoogleGenAI embeddings)]
        C3[Store in Chroma vector DB]
        C4[User chat prompt<br/>+ stored history]
        C5[Retrieve top-k chunks<br/>(similarity search)]
        C6[LLM answer using<br/>context + history only]
        C1 --> C2 --> C3
        C4 --> C5 --> C6
    end

    %% Branch wiring and outputs
    I -- Notes For You --> N1
    I -- Chat with Video --> C1
    N6 --> Z[Streamlit UI renders<br/>topics, gaps, notes, scores]
    C6 --> Z
```

```mermaid
flowchart TD
    %% ============ Input & Setup ============
    A[Start] --> B[User inputs YouTube URL<br/>+ selects language & task<br/>(Notes For You / Chat with Video)]
    B --> C[Load language map<br/>Extract video_id from URL]
    C --> D[Fetch transcript via YouTubeTranscriptApi]

    %% ============ Translation Decision ============
    D --> E{Is transcript<br/>already in English?}
    E -- "Yes" --> F[Use transcript as-is]
    E -- "No" --> G[Translate transcript to English<br/>via LLM]
    F --> H{Which task<br/>did user select?}
    G --> H

    %% ============ Notes For You Branch ============
    subgraph S1[Notes For You Flow]
        direction TB
        N1[LLM: get_important_topics<br/>(important topics from transcript)]
        N2[LLM: identify_knowledge_gaps<br/>(knowledge gaps per topic)]
        N3[Fetch external context for gaps<br/>via Wikipedia REST<br/>(fallback: LLM-only context)]
        N4[LLM: generate enriched notes<br/>weaving in gap contexts]
        N5[Display to user:<br/>• Important topics<br/>• Gaps + external context<br/>• Enriched notes]

        N1 --> N2 --> N3 --> N4 --> N5
    end

    %% ============ Chat with Video Branch ============
    subgraph S2[Chat with Video (RAG) Flow]
        direction TB
        C1[Chunk transcript<br/>(RecursiveCharacterTextSplitter)]
        C2[Embed chunks with<br/>GoogleGenerativeAIEmbeddings]
        C3[Store embeddings in<br/>Chroma vector store]
        C4[User chat question]
        C5[Retrieve k = 4 most<br/>relevant chunks from Chroma]
        C6[LLM answers using<br/>retrieved chunks as context only]

        C1 --> C2 --> C3
        C4 --> C5 --> C6
    end

    %% ============ Branch Wiring ============
    H -- "Notes For You" --> N1
    H -- "Chat with Video" --> C1

    N5 --> Z[End]
    C6 --> Z

    %% ============ Data Flow Summary (inline) ============
    %% URL → video_id → transcript → (translate?) →
    %%   Notes flow: topics → gaps → external context → enriched notes
    %%   Chat flow: chunks → embeddings → Chroma → RAG answers

```
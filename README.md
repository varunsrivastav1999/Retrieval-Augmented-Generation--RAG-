# 🚀 i-Tips RAG: Enterprise-Grade Production RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)

An advanced, **7-Layer Retrieval-Augmented Generation (RAG)** engine designed for high-stakes technical environments. This system handles **1,000,000+ pages** with millisecond search latency, deep contextual accuracy, and 100% offline privacy.

---

## 🌟 The "Amazing Concept": 7-Layer Intelligence
Unlike basic RAG systems, i-Tips RAG uses a multi-layered pipeline to ensure the AI never "hallucinates" and always has the full technical picture.

### 🏗 Architecture Diagram
```
                    ┌──────────────────────────────────────────┐
                    │          Enterprise RAG Engine           │
                    │                                          │
  PDF Ingest  ──────►  Layer 1: Smart OCR & Table Extraction   │
                    │  Layer 2: Recursive Character Chunking   │
  User Query  ──────►  Layer 3: Hybrid Search (HNSW + BM25)    │
                    │  Layer 4: Cross-Encoder Reranking        │
                    │  Layer 5: Max Marginal Relevance (MMR)   │
                    │  Layer 6: Contextual Window Expansion    │
                    │  Layer 7: Real-Time Token Streaming      │
                    │                                          │
                    └────┬──────────┬──────────┬───────────────┘
                         │          │          │
                    ┌────▼───┐ ┌───▼────┐ ┌──▼──────┐
                    │Postgres│ │ Redis  │ │ Ollama  │
                    │pgvector│ │ Cache  │ │ LLM     │
                    └────────┘ └────────┘ └─────────┘
```

---

## 🚀 Key Enterprise Features

### 1. **High-Scale Performance (1M+ Pages)**
*   **HNSW Indexing**: Uses Hierarchical Navigable Small Worlds for O(log n) search complexity.
*   **Semantic Caching**: Redis-backed SHA-256 caching bypasses the LLM for repeat queries.

### 2. **Contextual Window Expansion**
The system doesn't just find a "snippet"—it automatically pulls the **neighboring sections** from the document. This gives the AI a broader "vision," essential for technical manuals and complex data.

### 3. **Real-Time Token Streaming**
Experience instant interaction. The UI displays the AI's thoughts as they are generated, eliminating the "waiting" period common in other systems.

### 4. **Hardware Auto-Detection (Cross-Platform)**
*   **Apple Silicon**: Auto-selects **MPS** (Metal) for Mac M1-M5.
*   **NVIDIA**: Auto-selects **CUDA** for Linux/Windows servers.
*   **Fallback**: High-performance **CPU** mode for general hardware.

---

## 🛠 Deployment & Setup

### 🐳 Quick Start (Docker)
```bash
# Clone the repository
git clone https://github.com/varunsrivastav1999/Retrieval-Augmented-Generation--RAG-.git
cd Retrieval-Augmented-Generation--RAG-

# Start the full stack
docker-compose -f local.yml up --build
```

### ⚡ Native Mode (Max GPU Speed on Mac)
To bypass Docker limits and use your Mac's M-series GPU directly:
```bash
pip install -r requirements.txt
python app/main.py
```

---

## 🗺 Roadmap
- [ ] **Multi-Modal Support**: Retrieval of images and diagrams via CLIP.
- [ ] **Agentic Re-Ranking**: Use a small LLM to decide which chunks are "actually" useful.
- [ ] **Evaluations Suite**: Integrated RAGAS benchmarks for accuracy tracking.
- [ ] **Plugin System**: Connect to SharePoint, Google Drive, and Slack.

---

## 🤝 Contributing
We welcome contributions! This is an open-source project dedicated to pushing the boundaries of local, private RAG.
1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/AmazingNewLayer`).
3. Commit your changes (`git commit -m 'Add some AmazingNewLayer'`).
4. Push to the branch (`git push origin feature/AmazingNewLayer`).
5. Open a Pull Request.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

---
**Created by the community for high-performance, private AI.**

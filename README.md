# Local RAG Pipeline POC

A complete Retrieval-Augmented Generation (RAG) pipeline built entirely with free tools, featuring local LLM inference, semantic search, and automated evaluation.

## 🎯 Project Overview

This project demonstrates a fully functional RAG system that:
- Runs completely offline using local models
- Ingests and processes academic papers (PDF documents)
- Performs semantic search using FAISS and sentence-transformers
- Generates contextual responses using a local LLM (GPT4All)
- Evaluates response quality using RAGAS metrics

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │    Vector       │    │   Local LLM     │
│   Ingestion     │───▶│   Database      │───▶│   Generation    │
│                 │    │   (FAISS)       │    │   (GPT4All)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Papers    │    │  Embeddings     │    │  Augmented      │
│   (3 sources)   │    │ (384-dim)       │    │  Responses      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **LLM** | GPT4All (Mistral-7B) | Local text generation |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Semantic text encoding |
| **Vector DB** | FAISS | Similarity search and retrieval |
| **Evaluation** | RAGAS | RAG quality assessment |
| **Documents** | ArXiv Papers | Knowledge base content |

## 📋 Prerequisites

- Python 3.8+
- 8GB+ RAM (for LLM inference)
- 10GB+ disk space (for models and documents)
- Windows/Linux/macOS

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/rishi2628/local_RAG.git
cd local_RAG

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Run individual components or the full pipeline
python src/document_ingestion.py    # Download and process documents
python src/vector_database.py       # Build FAISS index
python src/rag_pipeline.py          # Generate responses
python src/ragas_evaluation.py      # Evaluate results
```

### 3. Interactive Usage

```python
from src.rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline()
rag.initialize()

# Ask questions
response = rag.query("What is the attention mechanism?")
print(f"Query: {response.query}")
print(f"Answer: {response.answer}")
print(f"Sources: {len(response.retrieved_contexts)} documents")
```

## 📊 Project Structure

```
local_RAG/
├── src/
│   ├── document_ingestion.py      # PDF download and text chunking
│   ├── vector_database.py         # FAISS indexing and search
│   ├── local_llm.py              # GPT4All wrapper
│   ├── rag_pipeline.py           # Main RAG implementation
│   └── ragas_evaluation.py       # Quality evaluation
├── documents/                     # Downloaded papers and chunks
├── models/                        # LLM and FAISS index files
├── evaluation/                    # Results and evaluation reports
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 📚 Knowledge Base

The system includes three academic papers:

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - 49 chunks, 46,883 characters
   - Transformer architecture and attention mechanisms

2. **BERT** (Devlin et al., 2018)
   - 81 chunks, 78,383 characters
   - Bidirectional transformer pre-training

3. **ResNet** (He et al., 2015)
   - 74 chunks, 71,789 characters
   - Deep residual learning for image recognition

**Total:** 204 document chunks, 197,055 characters

## 🔍 Sample Queries and Results

### Query 1: "What is the attention mechanism and how does it work?"

**Response:** The attention mechanism in the Transformer model works by allowing each position in the input sequence to attend over all positions in the input sequence. This is done through multi-head attention, which computes a weighted sum of value vectors based on similarity scores between queries and keys...

**Performance:**
- Retrieval Time: 0.400s
- Generation Time: 249.834s
- Documents Retrieved: 5
- Top Source: attention_is_all_you_need (Score: 0.8820)

### Query 2: "How does BERT differ from traditional language models?"

**Response:** Based on the provided context document, BERT differs from traditional language models in its unified architecture across different tasks. Unlike traditional language models that require separate models for each task, BERT has a multi-layer bidirectional Transformer encoder...

**Performance:**
- Retrieval Time: 0.643s
- Generation Time: 154.079s
- Documents Retrieved: 5
- Top Source: bert_paper (Score: 0.8039)

### Query 3: "What are residual connections and why are they important?"

**Response:** Residual connections are a type of connection in neural networks that allow the output of one layer to be passed directly to an input of another layer, without any further processing. This is done by adding a shortcut path between the two layers...

**Performance:**
- Retrieval Time: 0.050s
- Generation Time: 216.957s
- Documents Retrieved: 5
- Top Source: resnet_paper (Score: 0.8704)

## 📈 Evaluation Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Average Response Length** | 2,791 characters |
| **Average Retrieval Time** | 0.364 seconds |
| **Average Generation Time** | 206.957 seconds |
| **Average Total Time** | 207.327 seconds |
| **Context Utilization** | 5.0 docs/query |
| **Response Completeness** | 1.00 (100%) |
| **Query Coverage** | 0.61 (61%) |

### Key Findings

✅ **Strengths:**
- **Perfect retrieval accuracy:** All queries successfully retrieved relevant documents
- **Complete responses:** 100% of responses were complete and well-formed
- **Fast retrieval:** Sub-second semantic search performance
- **Relevant context:** High-quality document matching with FAISS

⚠️ **Areas for Improvement:**
- **Generation speed:** Average 3.5 minutes per response (could be optimized with smaller models)
- **Query coverage:** 61% overlap between query terms and response terms (could improve with better prompt engineering)

### Recommendations

1. **Performance Optimization:**
   - Consider using a smaller, quantized model for faster inference
   - Implement GPU acceleration if available
   - Add response caching for repeated queries

2. **Quality Enhancement:**
   - Improve prompt engineering for better query coverage
   - Implement context ranking and filtering
   - Add response post-processing and validation

3. **System Expansion:**
   - Increase document diversity and volume
   - Add multi-language support
   - Implement real-time document ingestion

## 🔧 Configuration

### Key Parameters

```python
# RAG Pipeline Configuration
LLM_MODEL = "mistral-7b-instruct-v0.1.Q4_0.gguf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 5
MAX_CONTEXT_LENGTH = 2000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

### Hardware Requirements

- **Minimum:** 8GB RAM, 4-core CPU
- **Recommended:** 16GB RAM, 8-core CPU
- **Storage:** 10GB for models and documents

## 🐛 Troubleshooting

### Common Issues

1. **CUDA/GPU Errors:**
   ```bash
   # Use CPU-only versions
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Memory Issues:**
   ```python
   # Reduce batch size and context length
   BATCH_SIZE = 16
   MAX_CONTEXT_LENGTH = 1000
   ```

3. **Slow Generation:**
   ```python
   # Use smaller model
   LLM_MODEL = "mistral-7b-openorca.gguf2.Q4_0.gguf"
   ```

## 📝 Dependencies

Core dependencies and their purposes:

```
sentence-transformers==2.7.0    # Text embeddings
faiss-cpu==1.7.4                # Vector similarity search
gpt4all==2.8.2                  # Local LLM inference
langchain==0.1.17               # Document processing
ragas==0.1.9                    # RAG evaluation metrics
PyPDF2==3.0.1                   # PDF text extraction
numpy==1.24.3                   # Numerical operations
pandas==2.0.3                   # Data manipulation
```

## 🔄 Development Workflow

1. **Document Ingestion:** Download PDFs → Extract text → Create chunks
2. **Index Building:** Generate embeddings → Build FAISS index → Save to disk
3. **Query Processing:** Embed query → Search index → Retrieve context
4. **Response Generation:** Format prompt → LLM inference → Return response
5. **Evaluation:** Collect responses → Run RAGAS metrics → Generate report

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with clear description

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Hugging Face** for sentence-transformers
- **Facebook Research** for FAISS
- **GPT4All** for local LLM inference
- **ArXiv** for open access papers
- **RAGAS** team for evaluation framework

## 📞 Contact

For questions or issues:
- Create an issue in this repository
- Contact: [Your contact information]

---

**Built with ❤️ using only free and open-source tools**
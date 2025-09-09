# Local RAG Pipeline

A complete Retrieval-Augmented Generation (RAG) pipeline using only free, open-source tools that runs entirely locally.

## 🎯 Project Overview

This project demonstrates how to build a production-ready RAG system using:
- **Local LLM**: GPT4All for response generation
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS for efficient similarity search
- **Evaluation**: Custom metrics and RAGAS framework
- **Document Processing**: Support for PDF, DOCX, and TXT files

## 🛠️ Features

- ✅ **Document Ingestion**: Automatic download and processing of sample documents
- ✅ **Semantic Chunking**: Intelligent text splitting with overlap
- ✅ **Vector Indexing**: FAISS-powered similarity search
- ✅ **Local LLM Integration**: GPT4All for private, offline generation
- ✅ **RAG Pipeline**: Complete retrieval + generation workflow
- ✅ **Comprehensive Evaluation**: Multiple metrics for quality assessment
- ✅ **Easy to Use**: Simple CLI interface and demo scripts

## 📋 Requirements

- Python 3.8+
- 8GB+ RAM (for LLM models)
- 5GB+ disk space (for models and data)
- Internet connection (for initial setup only)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd local_RAG
pip install -r requirements.txt
```

### 2. Run Simple Demo

```bash
python demo.py
```

### 3. Run Complete Pipeline

```bash
python main.py
```

## 📁 Project Structure

```
local_RAG/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── main.py                  # Complete pipeline demonstration
├── demo.py                  # Simple demo script
├── document_processor.py    # Document ingestion and FAISS indexing
├── llm_interface.py         # Local LLM integration and RAG pipeline
├── rag_evaluator.py         # Evaluation metrics and reporting
├── data/                    # Sample documents (auto-created)
├── models/                  # LLM models (auto-downloaded)
├── results/                 # Evaluation results and reports
└── faiss_index.*           # Vector index files
```

## 🔧 Core Components

### DocumentProcessor
Handles document ingestion, chunking, and FAISS indexing:

```python
from document_processor import DocumentProcessor

processor = DocumentProcessor()
files = processor.download_sample_documents()
processor.process_documents(files)
processor.create_embeddings()
processor.build_faiss_index()
processor.save_index()
```

### LocalLLM
Interface for local language models:

```python
from llm_interface import LocalLLM

llm = LocalLLM(model_type="gpt4all")
llm.load_model()
response = llm.generate_response("What is AI?")
```

### RAGPipeline
Complete retrieval + generation pipeline:

```python
from llm_interface import RAGPipeline

rag = RAGPipeline(processor, llm)
result = rag.query("Explain machine learning", k=3)
print(result['response'])
```

### RAGEvaluator
Comprehensive evaluation metrics:

```python
from rag_evaluator import evaluate_rag_pipeline

evaluation = evaluate_rag_pipeline(results)
```

## 📊 Sample Queries

The pipeline comes with three sample queries for demonstration:

1. **"What is artificial intelligence and how is it used today?"**
2. **"Explain machine learning and its main types."**
3. **"What are the advantages of Python programming language?"**

## 📈 Evaluation Metrics

The system provides comprehensive evaluation including:

### Retrieval Metrics
- Average number of documents retrieved
- Mean retrieval similarity scores
- Score distribution analysis

### Answer Quality
- Answer relevance (semantic similarity to question)
- Context relevance (how well context matches question)
- Completeness and informativeness scores
- Hallucination detection

### Example Evaluation Output
```
RAG PIPELINE EVALUATION REPORT
============================================================

📋 RETRIEVAL METRICS:
  • Average documents retrieved: 3.00
  • Average retrieval score: 0.7245
  • Retrieval score std dev: 0.1123

🎯 ANSWER RELEVANCE:
  • Mean relevance score: 0.6834
  • Standard deviation: 0.0892

📄 CONTEXT RELEVANCE:
  • Mean context relevance: 0.7456
  • Standard deviation: 0.0756

✨ ANSWER QUALITY:
  • Average answer length: 245.3 chars
  • Average word count: 42.7 words
  • Completeness score: 1.00
  • Informativeness score: 2.34

🚨 HALLUCINATION DETECTION:
  • Mean hallucination score: 0.2341 (lower is better)
  • Standard deviation: 0.0567
```

## 🔍 How It Works

1. **Document Ingestion**: Downloads and processes sample documents (AI, ML, Python guides)
2. **Text Chunking**: Splits documents into overlapping chunks for better context
3. **Embedding Creation**: Uses sentence-transformers to create semantic embeddings
4. **Index Building**: Creates FAISS index for efficient similarity search
5. **Query Processing**: For each query:
   - Embeds the query
   - Retrieves top-k most similar chunks
   - Constructs RAG prompt with context
   - Generates response using local LLM
6. **Evaluation**: Analyzes retrieval quality, answer relevance, and potential hallucinations

## ⚙️ Configuration

### Embedding Model
Change the embedding model in `DocumentProcessor`:
```python
processor = DocumentProcessor(embedding_model_name="all-mpnet-base-v2")
```

### LLM Model
Use different GPT4All models:
```python
llm = LocalLLM(model_name="orca-mini-3b-gguf2-q4_0.gguf")
```

### Chunking Parameters
Adjust chunking in `DocumentProcessor.chunk_text()`:
```python
chunks = processor.chunk_text(text, chunk_size=1000, overlap=100)
```

## 🎯 Use Cases

- **Research Assistant**: Query scientific papers and documentation
- **Customer Support**: Answer questions based on company knowledge base
- **Educational Tool**: Create interactive learning experiences
- **Content Analysis**: Analyze and summarize large document collections
- **Personal Knowledge Management**: Build your own searchable knowledge base

## 🔒 Privacy & Security

- **100% Local**: No data sent to external APIs
- **Offline Capable**: Works without internet after initial setup
- **Private**: Your documents and queries stay on your machine
- **Open Source**: Full transparency, no black boxes

## 🚀 Performance Tips

1. **Memory**: Ensure adequate RAM for model loading
2. **Storage**: Use SSD for faster index operations
3. **Batch Processing**: Process multiple documents together
4. **Model Selection**: Balance model size vs. quality needs
5. **Chunk Size**: Optimize chunk size for your use case

## 🛠️ Troubleshooting

### Common Issues

**Model download fails:**
- Check internet connection
- Ensure sufficient disk space
- Try smaller model (orca-mini-3b)

**Out of memory:**
- Close other applications
- Use smaller model
- Reduce batch size

**Poor retrieval quality:**
- Increase chunk overlap
- Try different embedding model
- Adjust chunk size

## 📚 Additional Resources

- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [GPT4All](https://gpt4all.io/)
- [RAGAS Framework](https://github.com/explodinggradients/ragas)

## 🤝 Contributing

Feel free to:
- Add support for more document types
- Implement additional LLM backends
- Enhance evaluation metrics
- Improve documentation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Happy RAG Building! 🚀**


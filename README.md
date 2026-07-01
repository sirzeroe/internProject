# IT Support AI Intern (Local RAG Pipeline)

This repository contains a fully local Retrieval-Augmented Generation (RAG) pipeline designed to act as an IT Support AI Intern. It ingests a dataset of historical IT support tickets to answer queries, troubleshoot common issues, and synthesize solutions based on past resolutions.

## Project Architecture
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Vector Database:** ChromaDB
* **Inference Engine:** Ollama 
* **Orchestration:** LangChain

## Model Benchmarking
To determine the most efficient model for IT support tasks, a custom benchmarking suite evaluates five local LLMs:
* `gemma:2b`
* `mistral:7b`
* `llama3.1`
* `phi3:mini`
* `qwen2.5`

The benchmark runs automated tests for grounding, hallucination traps, multi-ticket synthesis, edge cases, and multilingual capabilities. Results are exported to a lightweight JSON file and a styled HTML report for easy stakeholder review.

## File Structure
* `buildDB.py`: Ingests the raw CSV dataset, generates embeddings, and compiles the local ChromaDB vector store.
* `queryAgent.py`: The interactive CLI agent for querying the IT Support AI.
* `benchmark.py`: Runs the automated test suite across all models and generates the performance reports.
* `benchmark_report.html` / `benchmark_results.json`: The final outputs of the benchmarking suite.

## Quick Start
1. Ensure your Python environment is set up and [Ollama](https://ollama.com/) is running locally with your chosen models pulled.
2. Install the required dependencies:
   ```bash
   pip install pandas langchain-core langchain-community chromadb sentence-transformers
3. Run `python buildDB.py` to initialize the vector database from the provided dataset.
4. Run `python queryAgent.py` to launch the interactive assistant in your terminal.
5. Open `benchmark_report.html` in any web browser to view the model performance breakdown.

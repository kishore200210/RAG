# RBI Document Question Answering System (RAG)

A local Retrieval-Augmented Generation (RAG) system that answers questions using documents published by the Reserve Bank of India (RBI).

The system retrieves relevant document passages, uses them as context for a local language model, and generates answers grounded in the available evidence. It also includes an evaluation dataset, an evaluation harness, retrieval improvements, and refusal detection for questions that cannot be answered from the documents.

---

## 1. Project Overview

### Problem

Finding answers in lengthy RBI documents manually can take considerable time. A language model may also generate incorrect information when it does not have reliable supporting evidence.

### Solution

This project combines document retrieval with a locally running language model.

When a user asks a question, the system:

1. Receives the question.
2. Searches the indexed RBI documents for relevant passages.
3. Optionally reranks the retrieved passages.
4. Passes the question and selected passages to the language model.
5. Generates an answer using the retrieved evidence.
6. Refuses to answer when the system determines that the required information is not available.
7. Evaluates retrieval and answer quality using a question-answer benchmark.

### Main objectives

- Build a question-answering system over RBI documents.
- Run the language model locally.
- Create a hand-written evaluation dataset containing 40 questions.
- Measure baseline performance.
- Evaluate retrieval improvements using reranking.
- Test refusal detection for unanswerable questions.
- Record evaluation results and latency.
- Provide a demonstration through a user interface.

---

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Document source | RBI documents |
| Document processing | PDF/text extraction and chunking |
| Embedding model | `BAAI/bge-base-en-v1.5` |
| Vector search | FAISS |
| Optional reranking | `BAAI/bge-reranker-base` |
| Local language model | Ollama with the model configured for the project |
| Evaluation dataset | JSONL |
| Demo interface | Streamlit |
| Testing | pytest |
| Code quality | Ruff |
| Version control | Git |
| Repository hosting | GitHub |

**Note:** Check `config.yaml` and the Ollama model configuration to confirm the exact generation model used for your final evaluation. Do not assume that results produced with different models are directly comparable.

---

## 3. Project Structure

The main project structure is organised as follows. Some generated files and implementation details may differ depending on the current repository state.

```text
rag-system-assignment/
│
├── app.py
├── config.yaml
├── Makefile
├── README.md
├── requirements.txt
│
├── src/
│   ├── cli.py
│   ├── ingest.py
│   ├── evaluate.py
│   └── generation.py
│
├── scripts/
│   └── calculate_kappa.py
│
├── eval/
│   ├── qa_pairs.jsonl
│   ├── unanswerable_qa_pairs.jsonl
│   └── human_validation_20_template.csv
│
├── results/
│   ├── baseline.json
│   ├── baseline_qwen.json
│   ├── reranking.json
│   └── reranking_qwen.json
│
└── report/
    └── RAG_Evaluation_Report.pdf
```

The exact files may vary. Check the repository before relying on a filename.

### Important directories

- `src/` — core application, ingestion, retrieval, generation, and evaluation code.
- `scripts/` — supporting scripts, including Cohen's kappa calculation.
- `eval/` — evaluation questions and human-validation template.
- `results/` — saved evaluation outputs.
- `report/` — generated evaluation report.

The vector index and downloaded corpus may be stored in other configured directories.

---

## 4. Prerequisites

Before running the project, install or configure:

- Ubuntu/Linux or another supported operating system.
- Python compatible with the project's dependencies.
- Git.
- An available internet connection for initial dependency, model, and corpus downloads.
- Sufficient disk space for document files, embeddings, the vector index, and model weights.
- Ollama with the required local language model.
- The Python packages listed in `requirements.txt`.

GPU acceleration is optional if the configured models and hardware support CPU execution. CPU inference may be slower.

---

## 5. Step 1 — Open the Project

Open a terminal and move to the project directory.

```bash
cd ~/Desktop/rag-system-assignment
```

Confirm the project files are available:

```bash
ls
```

Inspect the main configuration:

```bash
cat config.yaml
```

Review the Python dependencies:

```bash
cat requirements.txt
```

---

## 6. Step 2 — Create the Python Environment

Create a virtual environment if one does not already exist.

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Verify Python:

```bash
python --version
```

Verify the installed packages:

```bash
pip check
```

**Important:** If `.venv` already exists and contains the working environment, activate it instead of recreating it.

---

## 7. Step 3 — Configure Ollama

Ollama runs the language model locally.

Check whether Ollama is installed:

```bash
ollama --version
```

List the models already available:

```bash
ollama list
```

Check the generation model specified by your project configuration.

If the required model is not installed, pull the exact model configured for your application. For example, if your configuration uses Qwen 2.5 7B:

```bash
ollama pull qwen2.5:7b
```

Test the model:

```bash
ollama run qwen2.5:7b
```

Enter a simple question to verify that the model responds. Exit the interactive session when finished.

The model name in this example must match your actual configuration. If your final experiment uses another model, use that model consistently when reproducing its results.

### Why use a local model?

A locally running model can generate answers without sending the question and retrieved document context to a hosted language-model API.

However, the application must be configured to use the local Ollama endpoint for this benefit to apply.

---

## 8. Step 4 — Download and Prepare the RBI Corpus

The corpus contains the documents used by the question-answering system.

The ingestion process typically involves:

1. Obtaining the configured RBI documents.
2. Reading PDF or text content.
3. Extracting document text.
4. Splitting long documents into smaller chunks.
5. Preserving useful source information and metadata.
6. Preparing the chunks for embedding and indexing.

The configured chunking parameters have included a chunk size of 512 tokens and an overlap of 64 tokens. Confirm the current values in the project configuration.

### Run ingestion

From the project root, activate the virtual environment and run:

```bash
python -m src.ingest
```

This is the project's ingestion entry point. Its exact download, extraction, and storage behaviour is determined by the implementation and configuration.

### Verify ingestion

Check the terminal output for errors and confirm that the expected corpus or processed files were created.

Do not assume the corpus is ready just because the command started. Wait for it to finish successfully.

---

## 9. Step 5 — Build the Vector Index

The vector index allows the system to search for passages based on semantic similarity rather than exact keyword matches alone.

The general process is:

1. Read the processed document chunks.
2. Convert each chunk into an embedding vector.
3. Store the vectors in FAISS.
4. Associate indexed vectors with the corresponding text and metadata.
5. Make the index available to the retrieval pipeline.

The project CLI supports index creation or rebuilding.

Run:

```bash
python -m src.cli index --rebuild
```

Use this command when you intentionally need to rebuild the index.

### What does `--rebuild` mean?

It requests a fresh index build rather than relying on an existing index.

Depending on the implementation, rebuilding can take time because documents may need to be processed and embedded again.

**Important:** Do not rebuild the index unnecessarily if the current index is already valid. A rebuild can consume CPU, memory, and disk space.

---

## 10. Step 6 — Ask a Question

Once the corpus and index are ready, use the CLI to ask a question.

```bash
python -m src.cli ask
```

Follow the prompts displayed by the application.

The question-answering pipeline works as follows:

1. Accept the user's question.
2. Convert the question into an embedding.
3. Search the FAISS index.
4. Retrieve the most relevant document chunks.
5. Optionally rerank those chunks.
6. Construct a prompt containing the question and selected evidence.
7. Generate an answer using the local language model.
8. Apply the configured refusal behaviour where appropriate.
9. Display the answer.

### Why is retrieval important?

The language model should use the retrieved RBI passages as evidence rather than answering only from its internal knowledge.

Retrieval quality affects the quality of the final answer. If the correct passage is not retrieved, the generator may be unable to answer correctly.

---

## 11. Step 7 — Understand the Baseline System

The baseline is the initial configuration used as a reference for later improvements.

It provides a way to measure whether a change improves retrieval, answer quality, or runtime.

The baseline evaluation uses the project's evaluation dataset and saves its output as a JSON file.

Example command:

```bash
python -m src.evaluate \
  --dataset eval/qa_pairs.jsonl \
  --output results/baseline_qwen.json
```

Use the actual generation model and options required by your current configuration.

### What does this command do?

- `python -m src.evaluate` runs the evaluation module.
- `--dataset` specifies the questions to evaluate.
- `--output` specifies where the results should be saved.

The evaluation harness runs the configured pipeline against the supplied questions and records the metrics supported by the implementation.

**Important:** Running this command may take considerable time because it can involve document retrieval, model inference, and judging. Do not start another full evaluation while an equivalent run is still active.

---

## 12. Step 8 — Understand the Evaluation Dataset

The main benchmark is stored in:

```text
eval/qa_pairs.jsonl
```

It contains 40 hand-written evaluation questions.

The benchmark includes different question types, such as:

- Factual questions.
- Multi-hop questions that require combining information.
- Table or numerical questions.
- Unanswerable questions.
- Ambiguous questions.

The dataset is used to evaluate system behaviour across different scenarios.

### Why keep the dataset?

A fixed dataset makes it possible to compare configurations using the same questions.

Do not delete `eval/qa_pairs.jsonl` before submission. It is needed to reproduce the benchmark results.

The targeted refusal test dataset is stored in:

```text
eval/unanswerable_qa_pairs.jsonl
```

Keep this file if you want to rerun the dedicated unanswerable-question evaluation.

---

## 13. Step 9 — Measure Baseline Performance

Run the baseline evaluation and save its output.

For example:

```bash
python -m src.evaluate \
  --dataset eval/qa_pairs.jsonl \
  --output results/baseline_qwen.json
```

Review the saved JSON file after the evaluation finishes.

The baseline provides the reference point for comparing improvements.

### Retrieval metrics

**Recall@5**

Measures how often a relevant document is retrieved within the top five results, according to the evaluation harness's relevance labels.

**Recall@10**

Measures retrieval coverage within the top ten results.

**MRR@10 — Mean Reciprocal Rank**

Measures how highly the first relevant result appears in the ranked list. Higher values indicate that relevant results tend to appear closer to the top.

### Answer metrics

**Correctness**

Measures whether the generated answer matches the expected answer or satisfies the evaluation criteria.

The project may use a heuristic or a language-model judge, depending on the evaluation configuration.

**Faithfulness**

Assesses whether the generated answer is supported by the supplied evidence.

**Refusal performance**

Measures whether the system correctly refuses questions that cannot be answered from the available documents.

### Latency metrics

**Average latency** is the average time taken per evaluated question.

**p50 latency** is the median latency: half the recorded requests finish at or below this value.

**p95 latency** is the 95th-percentile latency: approximately 95% of recorded requests finish at or below this value.

Always identify which part of the pipeline the timing covers. End-to-end latency must not be presented as generation-only latency unless generation time was measured separately.

---

## 14. Step 10 — Improvement One: Cross-Encoder Reranking

Initial vector retrieval ranks passages using embedding similarity.

However, a passage with a high embedding similarity score is not always the best passage for the exact question.

A cross-encoder reranker evaluates a question and candidate passage together to produce a more precise relevance ranking.

### Reranking workflow

1. Retrieve candidate passages using FAISS.
2. Score the candidates using the configured cross-encoder.
3. Sort the candidates by reranking score.
4. Select the highest-ranked passages.
5. Use the selected passages as context for answer generation.

The configured reranker is:

```text
BAAI/bge-reranker-base
```

### Run a retrieval-only reranking evaluation

```bash
python -m src.evaluate \
  --dataset eval/qa_pairs.jsonl \
  --output results/reranking.json \
  --rerank \
  --retrieval-only
```

### What does `--rerank` do?

It enables reranking for the evaluation, provided the implementation supports the configured reranker.

### What does `--retrieval-only` do?

It evaluates retrieval without running the complete answer-generation pipeline.

This is useful for isolating retrieval performance.

### Compare baseline and reranking results

Compare the following files:

```text
results/baseline.json
results/reranking.json
```

Focus on:

- Recall@5.
- Recall@10.
- MRR@10.
- Retrieval latency.

A previous recorded retrieval-only run reported:

| Metric | Baseline | Reranking |
|---|---:|---:|
| Recall@5 | 76.92% | 96.15% |
| Recall@10 | 88.46% | 96.15% |
| MRR@10 | 64.80% | 86.22% |

These are previously recorded results, not a guarantee that a new run will produce identical numbers. Reproduce the evaluations before presenting them as final.

**Interpretation:** The recorded results indicate improved retrieval coverage and ranking quality after reranking.

Because these are retrieval-only results, they do not by themselves prove that end-to-end answer correctness improved.

---

## 15. Step 11 — Evaluate Reranking With Answer Generation

Retrieval-only evaluation measures retrieval behaviour, but it does not measure the complete question-answering pipeline.

To evaluate the complete pipeline, run the evaluation without the retrieval-only option.

Example:

```bash
python -m src.evaluate \
  --dataset eval/qa_pairs.jsonl \
  --output results/reranking_full_qwen.json \
  --rerank
```

Use this command only if the current evaluator supports these options and the generation model is configured correctly.

Compare the full-RAG result with the baseline generated using the same dataset and compatible evaluation settings.

Check:

- Answer correctness.
- Faithfulness.
- Refusal behaviour.
- Average latency.
- p50 and p95 latency.
- Model errors or skipped evaluations.

Do not compare retrieval-only latency directly with full-RAG latency as though they measure the same work.

---

## 16. Step 12 — Improvement Two: Refusal Detection

A question-answering system should not confidently invent an answer when the required information is unavailable.

The project includes refusal detection to identify situations where the available evidence does not support an answer.

The configured refusal message has been:

```text
Not found in the provided documents.
```

The exact wording depends on the current implementation.

### Expected behaviour

**Answerable question**

The system retrieves relevant evidence and attempts to answer from it.

**Unanswerable question**

The system should refuse to provide an unsupported factual answer.

**Ambiguous question**

The system should avoid making an unjustified assumption. Depending on the implementation, it may request clarification or refuse to answer.

### Run the targeted refusal evaluation

```bash
python -m src.evaluate \
  --dataset eval/unanswerable_qa_pairs.jsonl \
  --output results/abstention_targeted.json
```

If the current evaluator supports a different command or requires an explicit abstention option, follow its actual CLI interface.

### Metrics to examine

**Refusal recall**

The proportion of unanswerable questions for which the system correctly refuses.

**Refusal precision**

The proportion of refusals that were appropriate.

A system that refuses every question may achieve high refusal recall but would not be useful for answerable questions. Therefore, both measures matter.

A previous targeted run over eight unanswerable questions recorded 8/8 correct refusals. This is encouraging but is a small test set, not proof of general reliability.

---

## 17. Step 13 — Run the Evaluation Harness

The evaluation harness provides a repeatable way to measure the system.

Example full evaluation:

```bash
python -m src.evaluate \
  --dataset eval/qa_pairs.jsonl \
  --output results/final_evaluation.json
```

The command uses the default evaluation configuration unless additional options are supplied.

### Recommended evaluation procedure

1. Verify the dataset.
2. Verify the generation model and retrieval configuration.
3. Run the baseline.
4. Save the baseline JSON output.
5. Enable reranking and run the retrieval-only evaluation.
6. Run the complete pipeline with reranking.
7. Evaluate refusal behaviour using the targeted dataset.
8. Inspect failures, skipped questions, and model errors.
9. Repeat important configurations where time and resources permit.
10. Record the configuration and results used in the final report.

### Why save JSON files?

Saved results preserve the measurements used in analysis and make it possible to compare different configurations.

Keep the main result files in the repository unless the project's submission policy requires another storage location.

---

## 18. Step 14 — LLM Judge and Human Validation

A language-model judge can help assess whether generated answers are correct or supported by retrieved evidence.

However, a judge is not automatically reliable.

It may incorrectly assess an answer, overlook evidence, or produce biased results when the same model is used for both generation and judging.

### Current validation template

The human-validation template is:

```text
eval/human_validation_20_template.csv
```

It contains 20 selected samples for independent human review.

The selection includes different question types, including factual, multi-hop, table/numerical, unanswerable, and ambiguous questions.

### How to complete human validation

1. Open the CSV in a spreadsheet application.
2. Read each question, generated answer, and relevant evidence.
3. Enter an independent human answerability label.
4. Enter a human correctness rating where applicable.
5. Save the completed CSV.
6. Run the kappa calculation script.

The answerability labels used by the template are:

```text
answerable
unanswerable
ambiguous
```

The correctness rating convention is:

| Value | Meaning |
|---|---|
| `1.0` | Correct or valid refusal |
| `0.5` | Partially correct |
| `0.0` | Incorrect |

Follow the template's instructions for ambiguous samples and any fields that are not applicable.

**Do not fill human-label columns using the system's own predictions.** Human ratings must be independent assessments.

### Calculate Cohen's kappa

Run:

```bash
python scripts/calculate_kappa.py
```

If using the project's virtual environment directly:

```bash
.venv/bin/python scripts/calculate_kappa.py
```

Cohen's kappa measures agreement between two sets of categorical labels while accounting for agreement that could occur by chance.

A kappa result should only be reported after the required independent labels have been supplied and the script successfully calculates it.

### Validation status

The previously checked template contained 0 of 20 human answerability labels and 0 of 20 human correctness labels.

Therefore, human validation and the associated kappa calculation were pending at that point. The missing labels must be completed before claiming that judge agreement has been validated.

---

## 19. Step 15 — Run Tests

Run the automated test suite from the project root.

```bash
python -m pytest
```

To obtain more detailed output:

```bash
python -m pytest -v
```

### Optional code-quality check

If Ruff is installed:

```bash
ruff check .
```

Review any reported issues. Distinguish pre-existing warnings from new issues introduced by your changes.

Do not claim that all tests pass unless the current run finishes successfully.

---

## 20. Step 16 — Launch the Streamlit Demo

The project contains a Streamlit application entry point:

```text
app.py
```

Activate the virtual environment, then run:

```bash
streamlit run app.py
```

The terminal should display the local address for the application.

Open that address in your browser.

### Demo checklist

1. Confirm the application starts successfully.
2. Ask a factual question that is covered by the RBI documents.
3. Verify that the answer is supported by retrieved evidence.
4. Ask a question that is not covered by the documents.
5. Check whether the system handles unsupported questions appropriately.
6. Review any displayed source passages or retrieval information.
7. Check the terminal for runtime errors.

Run this checklist before the live demonstration. A previous successful launch does not guarantee that the current environment is ready.

---

## 21. Step 17 — Review the Final Results

Before creating the final report, inspect the saved evaluation files.

For example:

```bash
ls results/
```

Review the relevant JSON outputs and confirm:

- The dataset used.
- The model used.
- Whether reranking was enabled.
- Whether the run included answer generation.
- Whether any questions were skipped.
- Whether model or evaluation errors occurred.
- Whether latency measures cover retrieval only or the complete pipeline.
- Whether the results can be reproduced.

### Interpretation guidelines

Use baseline results as the reference point.

Use retrieval-only reranking results to explain retrieval improvements.

Use full-RAG results to assess the impact on answer generation.

Use targeted unanswerable-question results to explain refusal behaviour.

Use human validation to assess the reliability of the LLM judge.

Do not combine measurements from incompatible runs into a single comparison without explaining the differences.

---

## 22. Step 18 — Prepare the Final Report

The report should summarise:

1. Project objective.
2. System architecture.
3. Models and technologies.
4. Evaluation dataset.
5. Baseline results.
6. Improvements implemented.
7. Measured results for each improvement.
8. Latency measurements.
9. Limitations.
10. Human-validation and judge-validation status.
11. Future improvements.

The existing report path is:

```text
report/RAG_Evaluation_Report.pdf
```

Confirm that the report exists and contains the latest verified measurements.

The assignment requires a one-page PDF report. Do not submit outdated measurements or describe pending work as completed.

---

## 23. Step 19 — Prepare the Git Repository

Review the project files before committing.

```bash
git status
```

Confirm that the repository includes the code, evaluation dataset, necessary configuration templates, relevant result files, and final report.

Check that sensitive files and local-only artefacts are not accidentally included.

In particular:

- Do not commit secrets or credentials.
- Do not commit a virtual environment.
- Do not upload unnecessarily large downloaded models.
- Keep the evaluation dataset and important result files needed to demonstrate the work.
- Check `.gitignore` before adding generated files.

Stage the intended submission files:

```bash
git add README.md src/ scripts/ eval/ results/ report/
```

If other required files are not already tracked, add them deliberately. Do not blindly stage local data, secrets, or unrelated files.

Review what is staged:

```bash
git diff --cached
```

Commit the work:

```bash
git commit -m "Complete RBI RAG evaluation and improvements"
```

---

## 24. Step 20 — Push to GitHub

First, check whether a remote repository is configured:

```bash
git remote -v
```

If `origin` is already configured, push the current branch:

```bash
git push -u origin main
```

If no remote exists, create an empty repository on GitHub and configure its actual repository URL:

```bash
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
```

Then push:

```bash
git push -u origin main
```

Replace the placeholder with your repository's real URL.

If GitHub requests authentication, use an approved authentication method, such as a personal access token or SSH authentication. Do not put credentials into the README or commit them to the repository.

---

## 25. Step 21 — Create the `v1.0` Tag

The assignment requires the repository to be tagged `v1.0`.

Before creating the tag, confirm that the final submission commit includes the required files and report.

Create the tag:

```bash
git tag -a v1.0 -m "RBI RAG System v1.0"
```

Verify the tag:

```bash
git tag
```

Push the branch and tag:

```bash
git push origin main
git push origin v1.0
```

If the tag already exists, inspect it before changing anything. Do not assume that an existing `v1.0` tag points to the latest submission commit.

---

## 26. Step 22 — Prepare the 15-Minute Demo

Use a simple, evidence-based demonstration.

| Time | Activity |
|---|---|
| 0–1 min | Introduce the problem and project goal. |
| 1–3 min | Explain the architecture and project structure. |
| 3–6 min | Ask an answerable question and demonstrate retrieval and answering. |
| 6–8 min | Test a question that is not covered by the documents. |
| 8–11 min | Explain baseline and reranking results. |
| 11–13 min | Show the evaluation harness and saved results. |
| 13–15 min | Explain limitations, validation status, and next steps. |

### Suggested introduction

> I built a local RAG question-answering system over RBI documents. The system retrieves relevant document passages and uses a local language model to generate evidence-based answers. I created a 40-question evaluation dataset, measured baseline performance, tested cross-encoder reranking, and implemented refusal detection for unsupported questions. The recorded retrieval results show improvements after reranking. Human validation of the language-model judge remains a separate validation task until the independent labels and agreement calculation are completed.

Only use claims that match the latest verified results.

---

## 27. Limitations

The project has several important limitations to consider.

- Retrieval quality depends on the available corpus and document-processing quality.
- A relevant passage may not always be retrieved.
- Reranking introduces additional computation.
- Local model inference may be slow, particularly on CPU.
- Generated answers can still be incorrect or insufficiently supported.
- Refusal detection may reject some questions that are actually answerable.
- LLM-judge results may be biased or incorrect.
- A small set of unanswerable questions cannot establish general refusal reliability.
- Human validation and Cohen's kappa must not be claimed complete without independent labels and a successful calculation.
- End-to-end latency and retrieval-only latency measure different workloads.

---

## 28. Troubleshooting

### Problem: Python package import errors

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

### Problem: Ollama model not found

List installed models:

```bash
ollama list
```

Pull the exact model required by the configuration.

### Problem: No documents are retrieved

Check that ingestion completed successfully, the index exists, and the application points to the correct index location.

### Problem: Evaluation takes a long time

Full evaluations may require many model-inference calls.

Check whether another evaluation is still running before starting a duplicate run. Do not terminate a process without confirming what it is doing.

### Problem: Cohen's kappa cannot be calculated

Open:

```text
eval/human_validation_20_template.csv
```

Complete the required independent human labels and rerun:

```bash
python scripts/calculate_kappa.py
```

A template with blank labels cannot produce a meaningful human-agreement result.

### Problem: Git push fails

Check the configured remote:

```bash
git remote -v
```

Check the branch:

```bash
git branch --show-current
```

Verify that your GitHub authentication method is configured correctly.

---

## 29. Final Submission Checklist

Before submission, verify every item.

- [ ] Environment setup is documented.
- [ ] The RBI corpus has been prepared successfully.
- [ ] The vector index can be built.
- [ ] The CLI can answer a question.
- [ ] The 40-question dataset is committed.
- [ ] The baseline evaluation has been run and saved.
- [ ] Reranking has been evaluated.
- [ ] End-to-end impact is reported only if measured.
- [ ] Refusal behaviour has been evaluated.
- [ ] Automated tests have been run.
- [ ] Important result files are preserved.
- [ ] Human-validation status is accurate.
- [ ] Cohen's kappa is reported only if successfully calculated.
- [ ] The final one-page PDF report is ready.
- [ ] The Streamlit demo has been tested.
- [ ] The README reflects the actual implementation.
- [ ] The repository has been pushed to GitHub.
- [ ] The `v1.0` tag points to the intended final commit.
- [ ] The 15-minute demonstration has been rehearsed.

---

## 30. Conclusion

This project demonstrates a local RAG pipeline for answering questions using RBI documents.

It combines document ingestion, embedding-based retrieval, FAISS indexing, optional cross-encoder reranking, local language-model generation, refusal detection, and quantitative evaluation.

The evaluation process provides a structured way to compare the baseline with improvements. The final submission should include reproducible code, the evaluation dataset, measured results, a clear report, and an honest account of remaining validation work.

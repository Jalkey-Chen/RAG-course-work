# RAG-course-work

Minimal Retrieval-Augmented Generation (RAG) system for answering questions about
my own machine learning course materials (lectures, homeworks, and final project).

The goal of this project is to build a small but complete **vector-based RAG**
pipeline and use it to critique how well the system can answer both **core ML
concept questions** and **project-specific questions**.

---

## Homework 7-1 Checklist

This repo is designed to satisfy the HW 7-1 requirements:

- ✅ **Use Vector RAG on your own set of documents**  
  - Indexes my own ML lecture notes, annotated slide PDFs, homework HTML exports,
    and my course final report.
- ✅ **Use document types other than just PDF**  
  - Mix of `*.pdf` and `*.html` (e.g., `ml_homework1.html`, `ml_homework4.html`).
- ✅ **Critique the results**  
  - See the **Evaluation & Critique** section below (15 questions, CSV + discussion).
- ✅ **Either a Streamlit app or a web service**  
  - Implemented as a **FastAPI web service** with `/api/ingest` and `/api/query`.
- ✅ **Vector RAG only (no Graph RAG)**  
  - Uses a FAISS index over embeddings; no graph-based retrieval.

---

## Project Structure

```text
RAG-course-work/
│
├── app/
│   ├── main.py                # FastAPI app entrypoint
│   ├── api/                   # HTTP routes (ingest, query, health)
│   ├── core/                  # embeddings, vector store, retrieval logic
│   ├── services/              # ingestion and query orchestration
│   ├── schemas/               # Pydantic request/response models
│   └── utils/                 # text splitting, file loaders, helpers
│
├── data/
│   ├── raw/                   # source docs: PDFs + HTML (lectures, HW, report)
│   └── vectorstore/           # FAISS index + metadata storage
│
├── scripts/
│   ├── eval_questions.json    # 15 benchmark questions (conceptual + project)
│   └── evaluate.py            # evaluation pipeline (CSV + annotated citations)
│
├── evaluation_results.csv     # saved answers + metadata for 15 questions
├── evaluation_results.md      # optional Markdown summary (if generated)
├── pyproject.toml
└── README.md
````

---

## Tech Stack

* **Web framework:** FastAPI
* **Environment:** managed with `uv`
* **Embeddings:** OpenAI embedding model (configurable via env)
* **Vector store:** FAISS (in `data/vectorstore`)
* **Documents:** mix of PDFs and HTML (lectures, annotated slides, homeworks, final report)
* **Evaluation:** custom `scripts/evaluate.py` that calls the `/api/query` endpoint

---

## Running the Service

### 1. Install dependencies

```bash
uv sync
```

### 2. Start the FastAPI server

```bash
uv run fastapi dev app/main.py
```

By default the API runs at `http://127.0.0.1:8000`.

### 3. Ingest documents

You can ingest either individual files or a directory. For the homework, I
point the API at the `data/raw` directory which contains:

* `Stanford_ML_Lecture_notes.pdf`
* Several `*-annotated.pdf` slide decks (KNN, kernels, bias–variance, etc.)
* `ml_homework1.html`, `ml_homework4.html`
* `ML_final_report.pdf` (my course project report)
* `clean_bills.csv` (data for my course project report)

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"paths": ["data/raw"]}'
```

This runs the loader → splitter → embedder → FAISS index build pipeline.

### 4. Query the RAG system

```bash
curl -X POST http://127.0.0.1:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain the bias-variance tradeoff and how it relates to underfitting/overfitting.", "k": 3}'
```

The response includes:

* `answer`: the generated answer
* `sources`: list of retrieved chunks with `metadata.source` and `score`

Answers also include inline citations like `[C1]`, `[C2]`.

---

## Evaluation Setup

To critically assess the system, I created **15 evaluation questions** in
`scripts/eval_questions.json`. They cover two broad groups:

1. **Core ML concepts**

   * Training vs. generalization error
   * Bias–variance tradeoff
   * Backpropagation and gradient descent
   * Cross-validation vs. single validation split
   * Kernel methods and non-linear decision boundaries
   * Linear vs. logistic regression vs. perceptron
   * Ridge regression vs. OLS
   * KNN advantages/limitations

2. **Project-specific questions** about my Illinois legislature project
   (based on `ML_final_report.pdf`)

   * Model comparison (Random Forest vs. XGBoost vs. Logistic Regression)
   * Role of cosponsorship and fiscal features
   * Feature selection for legislative text
   * Embedding-based similarity for bills
   * Interpretability needs in policy prediction
   * Best-performing model and why
   * How to extend to other states / generalization advice

Evaluation is done by calling the live FastAPI backend:

```bash
uv run python scripts/evaluate.py --k 3 --max-answer-chars 1500 --markdown
```

This script:

* Sends each question to `/api/query`
* Saves results to `evaluation_results.csv` with:

  * `answer` (raw answer with `[C1]` tags)
  * `answer_annotated` where `[C1]` is expanded to `[C1: filename.pdf]`
  * `source_files`, `source_map`, `source_map_with_scores`
  * Latency (`duration`), answer length (`chars`), and a simple hallucination flag

---

## Evaluation & Critique

### Quantitative overview

On these 15 questions:

* **Top-k:** `k = 3` retrieved chunks per query
* **Sources per answer:** always 3 (consistent retrieval)
* **Hallucination heuristic:** all entries are `0`
  → every answer includes at least one explicit citation `[C#]`.
* **Latency:** around **9–19 seconds** per request, roughly ~13 s on average
  → dominated by LLM generation + embedding calls, not FAISS lookup.
* **Answer length:** typically **600–2000 characters**
  → long, textbook-style explanations rather than short snippets.

### Retrieval quality

Retrieval is generally well-aligned with the question type:

* **Conceptual ML questions** mainly use:

  * `Stanford_ML_Lecture_notes.pdf`
  * Specific annotated slide decks such as
    `bias-variance-tradeoff-annotated.pdf`,
    `k-nearest-neighbors-annotated.pdf`,
    `kernels-annotated.pdf`,
    `ridge-regression-annotated.pdf`,
    `neural-networks-backpropogation-annotated.pdf`.
* **Project questions** consistently retrieve `ML_final_report.pdf`
  for questions about cosponsorship, fiscal data, embedding-based similarity, and
  model comparison.

This is exactly what I want: the system routes high-level ML questions to
lecture content and project-specific questions to my own report. The
`answer_annotated` column confirms that inline citations are correctly mapped
to the underlying files, e.g.:

* `[C1: Stanford_ML_Lecture_notes.pdf]` for training vs. generalization error
* `[C1: k-nearest-neighbors-annotated.pdf]` for the KNN question
* `[C1: ML_final_report.pdf]` for questions about legislative success prediction.

One limitation is that **most answers rely on a single document**, sometimes
repeating the same PDF as `C1`, `C2`, and `C3`. This suggests that with `k = 3`
the top chunks often come from the **same file**, so the diversity of evidence is
lower than it could be. A reranking step or a “one-chunk-per-document” policy
might encourage more varied citations.

### Answer quality

For these 15 questions, the answers are:

* **Factually consistent** with the source material.
  The explanations of bias–variance, ridge vs. OLS, kernel methods, and
  backpropagation match the lecture notes and slides.
* **Well-grounded in the project report** for Illinois legislature–related
  questions. The system correctly identifies:

  * XGBoost as the best-performing model in terms of recall on the minority
    class (35/55 vs. 28/55 for others).
  * Cosponsorship count as the strongest predictor of bill success.
  * PCA-based features and TF-IDF as important textual signals.
  * SHAP / interpretability considerations in the policy prediction setting.
* **Verbose and somewhat generic**.
  Many answers read like short essays: they are great for understanding, but
  not ideal if the goal is bullet-point summaries. This is a prompting issue,
  not a retrieval issue.

Importantly, I did not observe obvious hallucinations in the 15 answers:
whenever a document is cited, I can find corresponding content in the source
PDF/HTML that supports the claim.

### System limitations

From this mini-evaluation, several limitations of the current RAG setup are clear:

1. **Latency is high**
   ~13 s per answer is fine for homework, but too slow for an interactive app.
   This comes from calling a relatively large LLM and generating long answers.
   Potential fixes:

   * Use a smaller, cheaper model for answer synthesis.
   * Add a “short answer” mode with a different prompt.
   * Cache answers or reuse embeddings across runs.

2. **Limited source diversity**
   For each query, all three retrieved chunks often come from the same file.
   This is acceptable for narrow questions (e.g., about a single lecture) but
   may miss complementary perspectives (e.g., combining lecture + homework
   solutions). A future improvement is to:

   * Sample at most one chunk per document in the top-k, or
   * Do a second‐stage rerank that encourages document diversity.

3. **Chunking and context window usage**
   Chunks are relatively long, so each answer may only see a few large
   paragraphs instead of many small, focused ones. This can waste some context
   window and makes it harder to cite specific equations or definitions.
   Tuning the splitter to shorter chunks with overlap could help.

4. **No automatic grading or reference answers**
   This evaluation is descriptive, not quantitative in terms of “correctness”.
   There is no gold-standard answer to compare against. A more rigorous
   evaluation would:

   * Hand-write short reference answers.
   * Use LLM-as-judge or simple rubric scoring (e.g., 0–5 for correctness,
     grounding, completeness).

### Possible future improvements

If I were to extend this project beyond HW 7-1, I would consider:

* Adding a **reranker** (e.g., cross-encoder) on top of FAISS.
* Supporting **query-time filters** (e.g., “only use project report” vs.
  “only use conceptual lectures”).
* Providing both **concise** and **detailed** answer modes.
* Evaluating with more questions and doing **error categorization**
  (missing concept, wrong emphasis, shallow project connection, etc.).
* Experimenting with **hybrid retrieval** (sparse + dense) for better coverage.

Overall, the RAG system behaves sensibly on this small benchmark: it picks the
right documents, produces coherent answers, and surfaces citations in a way
that lets me quickly check whether the model is actually grounded in my own
materials.


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from datetime import datetime
import time
import json
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODELS = [
    "gemma:2b",
    "mistral",
    "llama3.1",
    "phi3:mini",
    "qwen2.5",
]

QUESTIONS = [
    # Grounding & Recall
    ("Grounding", "What steps were already taken before a ticket was opened about the MacBook Pro and Laser Printer issue?"),
    ("Grounding", "How does the billing cycle work and when are payments due?"),
    ("Grounding", "What smart home platforms do your products integrate with?"),
    # Hallucination Traps
    ("Hallucination Trap", "What is the exact price of the smart home integration product?"),
    ("Hallucination Trap", "Who was the IT technician assigned to resolve the account management portal outage?"),
    # Multi-Ticket Synthesis
    ("Synthesis", "What are the most common causes of system disruptions mentioned across tickets?"),
    ("Synthesis", "What types of security incidents have been reported and what was the recommended response?"),
    # Edge Cases
    ("Edge Case", "How do I integrate Microsoft SQL Server 2019 with the SaaS project management tool?"),
    ("Edge Case", "What third-party apps does the SaaS platform support for integration?"),
    # Multilingual
    ("Multilingual", "Was there a data breach involving medical records? What caused it?"),
]

HALLUCINATION_TRAP_IDS = {3, 4}  # 0-indexed positions of trap questions

template = """You are a Senior IT Support Agent. Use the following historical ticket logs to answer the user's question.
If the answer is not contained in the logs, you must say "I do not have enough information in the historical tickets to answer that."
Do not guess or hallucinate.

Historical Tickets:
{context}

User Question: {question}
Resolution:"""

# ── SETUP ───────────────────────────────────────────────────────────────────
print("Loading database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
prompt = PromptTemplate.from_template(template)

def ask_model(model_name, question):
    llm = Ollama(model=model_name)
    docs = vectorstore.similarity_search(question, k=6)
    context = "\n\n".join([doc.page_content for doc in docs])
    formatted_prompt = prompt.format(context=context, question=question)
    start = time.time()
    response = llm.invoke(formatted_prompt)
    elapsed = round(time.time() - start, 2)
    return response.strip(), elapsed

# ── RUN BENCHMARK ───────────────────────────────────────────────────────────
results = {}

for model in MODELS:
    print(f"\n{'='*50}")
    print(f"  Testing: {model}")
    print(f"{'='*50}")
    results[model] = []

    for i, (category, question) in enumerate(QUESTIONS):
        print(f"  Q{i+1}: {question[:60]}...")
        try:
            answer, elapsed = ask_model(model, question)
            results[model].append({
                "category": category,
                "question": question,
                "answer": answer,
                "elapsed": elapsed,
                "error": None
            })
            print(f"  ✓ Done in {elapsed}s")
        except Exception as e:
            results[model].append({
                "category": category,
                "question": question,
                "answer": None,
                "elapsed": None,
                "error": str(e)
            })
            print(f"  ✗ Error: {e}")

# ── SAVE RAW JSON ────────────────────────────────────────────────────────────
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nRaw results saved to benchmark_results.json")

# ── GENERATE HTML REPORT ─────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Grounding":          ("#1a3a5c", "#e8f0fe"),
    "Hallucination Trap": ("#5c1a1a", "#fdecea"),
    "Synthesis":          ("#1a4a2e", "#e8f5e9"),
    "Edge Case":          ("#3a2a00", "#fff8e1"),
    "Multilingual":       ("#2a1a5c", "#ede7f6"),
}

def refusal_detected(text):
    if text is None:
        return False
    phrases = [
        "i do not have enough information",
        "cannot answer",
        "not contained in",
        "context does not provide",
        "not provided in",
    ]
    return any(p in text.lower() for p in phrases)

# Per-model scoring (simple heuristic)
def score_model(model_results):
    score = 0
    notes = []
    for i, r in enumerate(model_results):
        if r["error"]:
            notes.append(f"Q{i+1}: Error")
            continue
        is_trap = i in HALLUCINATION_TRAP_IDS
        refused = refusal_detected(r["answer"])
        if is_trap:
            if refused:
                score += 2
                notes.append(f"Q{i+1}: ✅ Correctly refused")
            else:
                notes.append(f"Q{i+1}: ⚠️ Possible hallucination")
        else:
            if refused:
                notes.append(f"Q{i+1}: ⚠️ Over-refused (answer exists in data)")
            else:
                score += 1
                notes.append(f"Q{i+1}: ✅ Attempted answer")
    return score, notes

scores = {m: score_model(results[m]) for m in MODELS}
ranked = sorted(MODELS, key=lambda m: scores[m][0], reverse=True)

# Build HTML
rows_html = ""
for i, (category, question) in enumerate(QUESTIONS):
    bg, fg_bg = CATEGORY_COLORS.get(category, ("#333", "#f5f5f5"))
    is_trap = i in HALLUCINATION_TRAP_IDS
    trap_badge = '<span class="trap-badge">TRAP</span>' if is_trap else ""

    cells = f"""
    <td class="q-cell">
      <div class="q-num">Q{i+1} <span class="cat-pill" style="background:{bg};color:#fff">{category}</span>{trap_badge}</div>
      <div class="q-text">{question}</div>
    </td>"""

    for model in MODELS:
        r = results[model][i]
        if r["error"]:
            cell_content = f'<span class="error">⚠ {r["error"]}</span>'
            cell_class = "cell-error"
        else:
            refused = refusal_detected(r["answer"])
            if is_trap:
                cell_class = "cell-good" if refused else "cell-warn"
                badge = '<span class="verdict good">REFUSED ✓</span>' if refused else '<span class="verdict warn">ANSWERED ⚠</span>'
            else:
                cell_class = "cell-warn" if refused else "cell-neutral"
                badge = '<span class="verdict warn">OVER-REFUSED</span>' if refused else ""
            timer = f'<span class="timer">{r["elapsed"]}s</span>'
            cell_content = f'{badge}{timer}<div class="answer-text">{r["answer"]}</div>'
        cells += f'<td class="{cell_class}">{cell_content}</td>'

    rows_html += f"<tr>{cells}</tr>\n"

# Scoreboard
scoreboard_html = ""
for rank, model in enumerate(ranked, 1):
    s, notes = scores[model]
    medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"#{rank}"
    scoreboard_html += f"""
    <div class="score-card">
      <div class="score-rank">{medal}</div>
      <div class="score-model">{model}</div>
      <div class="score-val">{s} / {len(QUESTIONS) + len(HALLUCINATION_TRAP_IDS)} pts</div>
    </div>"""

# Avg response time
avg_times_html = ""
for model in MODELS:
    times = [r["elapsed"] for r in results[model] if r["elapsed"] is not None]
    avg = round(sum(times) / len(times), 2) if times else "N/A"
    avg_times_html += f"<div class='time-chip'><b>{model}</b>: {avg}s avg</div>"

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Intern Benchmark Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

  :root {{
    --bg: #0d0f14;
    --surface: #161920;
    --border: #252933;
    --text: #e2e8f0;
    --muted: #64748b;
    --accent: #38bdf8;
    --good: #22c55e;
    --warn: #f59e0b;
    --bad: #ef4444;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    padding: 24px;
  }}

  header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
    margin-bottom: 28px;
  }}

  header h1 {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    color: var(--accent);
    letter-spacing: -0.5px;
  }}

  header p {{ color: var(--muted); margin-top: 6px; font-size: 12px; }}

  .section-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    margin-bottom: 12px;
  }}

  /* Scoreboard */
  .scoreboard {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 32px;
  }}

  .score-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex: 1;
    min-width: 160px;
  }}

  .score-rank {{ font-size: 22px; }}
  .score-model {{ font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--accent); flex: 1; }}
  .score-val {{ font-weight: 700; font-size: 15px; }}

  /* Timing */
  .timing-row {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 32px;
  }}

  .time-chip {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 12px;
  }}

  /* Table */
  .table-wrap {{
    overflow-x: auto;
    border-radius: 10px;
    border: 1px solid var(--border);
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    min-width: 900px;
  }}

  thead th {{
    background: var(--surface);
    padding: 12px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    text-align: left;
    white-space: nowrap;
  }}

  thead th:first-child {{ color: var(--text); }}

  tr + tr td {{ border-top: 1px solid var(--border); }}

  .q-cell {{
    background: var(--surface);
    padding: 14px;
    min-width: 220px;
    max-width: 260px;
    vertical-align: top;
  }}

  .q-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }}

  .q-text {{ font-size: 12px; line-height: 1.6; color: var(--text); }}

  .cat-pill {{
    font-size: 9px;
    padding: 2px 7px;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }}

  .trap-badge {{
    background: var(--bad);
    color: #fff;
    font-size: 9px;
    padding: 2px 7px;
    border-radius: 20px;
    font-weight: 700;
    letter-spacing: 0.5px;
  }}

  td {{
    padding: 12px 14px;
    vertical-align: top;
    min-width: 160px;
    max-width: 240px;
  }}

  .cell-good {{ background: #0d1f12; }}
  .cell-warn {{ background: #1f180d; }}
  .cell-neutral {{ background: var(--bg); }}
  .cell-error {{ background: #1f0d0d; }}

  .verdict {{
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 6px;
    letter-spacing: 0.5px;
  }}

  .verdict.good {{ background: #14532d; color: var(--good); }}
  .verdict.warn {{ background: #451a03; color: var(--warn); }}

  .timer {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    margin-left: 8px;
    margin-bottom: 6px;
  }}

  .answer-text {{
    font-size: 12px;
    line-height: 1.6;
    color: #94a3b8;
    margin-top: 4px;
    white-space: pre-wrap;
    word-break: break-word;
  }}

  .error {{ color: var(--bad); font-size: 12px; }}

  footer {{
    margin-top: 32px;
    color: var(--muted);
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    border-top: 1px solid var(--border);
    padding-top: 16px;
  }}
</style>
</head>
<body>

<header>
  <h1>// INTERN BENCHMARK REPORT</h1>
  <p>Generated: {timestamp} &nbsp;|&nbsp; Models tested: {len(MODELS)} &nbsp;|&nbsp; Questions: {len(QUESTIONS)}</p>
</header>

<div class="section-title">Leaderboard</div>
<div class="scoreboard">
  {scoreboard_html}
</div>

<div class="section-title">Avg Response Time</div>
<div class="timing-row">
  {avg_times_html}
</div>

<div class="section-title">Full Results</div>
<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>Question</th>
        {''.join(f'<th>{m}</th>' for m in MODELS)}
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</div>

<footer>
  RAG config: k=6 &nbsp;|&nbsp; Embeddings: all-MiniLM-L6-v2 &nbsp;|&nbsp; Vector DB: ChromaDB
</footer>

</body>
</html>"""

report_path = "benchmark_report.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✅ Report saved to: {report_path}")
print("Open it in any browser.")
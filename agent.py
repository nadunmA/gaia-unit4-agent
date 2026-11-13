import os
import re
import time
import base64
import requests
import litellm
from smolagents import CodeAgent, LiteLLMModel, tool

litellm.drop_params = True


GEMINI_API_KEY = " "  
USERNAME = "nadunAlgoda"
AGENT_CODE_URL = "https://github.com/nadunmA/gaia-unit4-agent/tree/main"

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
BASE_URL = "https://agents-course-unit4-scoring.hf.space"


@tool
def search_wikipedia(query: str) -> str:
    """
    Searches English Wikipedia and returns matching article titles and snippets.
    Args:
        query: Search keywords.
    """
    try:
        res = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 5},
            headers={"User-Agent": "GAIA-Benchmark-Agent/1.0 (contact@example.com)"},
            timeout=10
        ).json()
        items = res.get("query", {}).get("search", [])
        if not items:
            return "No Wikipedia results found."
        out = []
        for item in items:
            snippet = re.sub(r"<.*?>", "", item["snippet"])
            out.append(f"Title: {item['title']}\nSnippet: {snippet}")
        return "\n\n".join(out)
    except Exception as e:
        return f"Wikipedia search error: {e}"

@tool
def get_wikipedia_page(title: str) -> str:
    """
    Fetches the full plain-text content (including tables/lists) of an English Wikipedia page by exact title.
    Args:
        title: Exact Wikipedia page title (e.g., 'Mercedes Sosa', '1928 Summer Olympics', 'Malko Competition').
    """
    try:
        res = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "page": title,
                "prop": "text",
                "format": "json",
                "redirects": 1
            },
            headers={"User-Agent": "GAIA-Benchmark-Agent/1.0 (contact@example.com)"},
            timeout=15
        ).json()
        html = res.get("parse", {}).get("text", {}).get("*", "")
        if not html:
            return f"Page '{title}' not found."
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "sup"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)[:25000]
    except Exception as e:
        return f"Error reading Wikipedia page '{title}': {e}"

@tool
def read_webpage_text(url: str) -> str:
    """
    Fetches readable text from any non-Wikipedia webpage URL using a browser User-Agent.
    Args:
        url: The full URL to visit.
    """
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=15
        )
        if resp.status_code != 200:
            return f"HTTP {resp.status_code} for {url}"
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return "\n".join([l.strip() for l in soup.get_text("\n").splitlines() if l.strip()])[:20000]
    except Exception as e:
        return f"Error fetching {url}: {e}"


@tool
def download_and_read_task_file(task_id: str, file_name: str) -> str:
    """
    Downloads and reads/analyzes the attached task file (supports .xlsx, .py, .csv, .pdf, .png, .jpg, .mp3).
    Args:
        task_id: The unique task ID.
        file_name: The attached file name.
    """
    candidate_urls = [
        f"{BASE_URL}/files/{task_id}",
        f"{BASE_URL}/files/{file_name}",
        f"https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/{file_name}"
    ]
    content_bytes = None
    for u in candidate_urls:
        try:
            r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if r.status_code == 200 and len(r.content) > 0:
                content_bytes = r.content
                break
        except Exception:
            continue

    if content_bytes is None:
        return "File could not be downloaded from server endpoints."

    local_path = f"/tmp/{file_name}"
    with open(local_path, "wb") as f:
        f.write(content_bytes)

    ext = os.path.splitext(file_name)[1].lower()
    try:
        if ext in [".txt", ".md", ".py", ".json", ".csv"]:
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()[:15000]
        elif ext in [".xlsx", ".xls"]:
            import pandas as pd
            excel_data = pd.read_excel(local_path, sheet_name=None)
            output = []
            for sheet, df in excel_data.items():
                output.append(f"--- Sheet: {sheet} ---\n{df.to_string()}")
            return "\n".join(output)[:15000]
        elif ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(local_path)
            return "\n".join([p.extract_text() or "" for p in reader.pages])[:15000]
        elif ext in [".png", ".jpg", ".jpeg", ".mp3", ".wav"]:
            mime = "image/png" if ext == ".png" else ("image/jpeg" if ext in [".jpg", ".jpeg"] else "audio/mp3")
            b64 = base64.b64encode(content_bytes).decode("utf-8")
            resp = litellm.completion(
                model="gemini/gemini-3.5-flash-lite",
                api_key=GEMINI_API_KEY,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract every detail, text, chess board FEN/position, or full spoken transcript from this file accurately."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                    ]
                }]
            )
            return f"File Analysis:\n{resp.choices[0].message.content}"
        return f"Saved to {local_path}"
    except Exception as e:
        return f"Saved to {local_path}, error: {e}"


model = LiteLLMModel(
    model_id="gemini/gemini-3.5-flash-lite",
    api_key=GEMINI_API_KEY,
    num_retries=3
)

agent = CodeAgent(
    tools=[search_wikipedia, get_wikipedia_page, read_webpage_text, download_and_read_task_file],
    model=model,
    additional_authorized_imports=["pandas", "numpy", "math", "datetime", "json", "re", "os", "csv", "subprocess", "requests", "bs4"],
    max_steps=8
)

SYSTEM_INSTRUCTION = """You are an expert AI assistant solving GAIA benchmark questions.
CRITICAL OUTPUT RULES:
1. Return ONLY the exact final answer—nothing else.
2. Do NOT include phrases like 'FINAL ANSWER', 'The answer is', explanations, or markdown formatting.
3. If the answer is a number, output only the digits (no units or commas unless explicitly asked).
4. If the answer is a name or string, output only the exact text with no trailing period.

Use `search_wikipedia` to find exact Wikipedia page titles, then `get_wikipedia_page` to read full tables/lists.

Question: {question}
{file_info}
"""

def clean_answer(raw_output) -> str:
    ans = str(raw_output).strip()
    if ans.startswith("```") or "import " in ans:
        return ""
    ans = re.sub(r"^(FINAL ANSWER\s*:?\s*|Answer\s*:?\s*)", "", ans, flags=re.IGNORECASE).strip()
    return ans.strip('"\'').rstrip(".")


print("Fetching GAIA Level 1 questions...")
questions = requests.get(f"{BASE_URL}/questions").json()
print(f"Total questions fetched: {len(questions)}\n")

answers_payload = []

for idx, q in enumerate(questions, 1):
    task_id = q["task_id"]
    question_text = q["question"]
    file_name = q.get("file_name", "")
    
    file_info = (
        f"Attached file: '{file_name}'. Call `download_and_read_task_file(task_id='{task_id}', file_name='{file_name}')` first."
        if file_name else "No files attached."
    )
    
    prompt = SYSTEM_INSTRUCTION.format(question=question_text, file_info=file_info)
    print(f"[{idx}/{len(questions)}] Running Task {task_id}...")
    
    final_ans = ""
    for attempt in range(2):
        try:
            raw_result = agent.run(prompt)
            final_ans = clean_answer(raw_result)
            break
        except Exception as e:
            print(f"Attempt {attempt+1} error on {task_id}: {e}")
            time.sleep(5)
            
    print(f" -> Submitted Answer: {final_ans}\n")
    answers_payload.append({
        "task_id": task_id,
        "submitted_answer": final_ans
    })
    time.sleep(2)


submission = {
    "username": USERNAME,
    "agent_code": AGENT_CODE_URL,
    "answers": answers_payload
}

print("Submitting answers to evaluation server...")
submit_res = requests.post(f"{BASE_URL}/submit", json=submission)
print("Status Code:", submit_res.status_code)
print("Response:", submit_res.json())

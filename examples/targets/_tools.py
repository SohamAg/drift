"""Shared mock tools for the target systems.

Every tool here is deterministic. That's deliberate: drift measures how a
multi-agent system behaves under perturbed *state*, so any nondeterminism
in the tool layer is noise that makes divergence detection harder to read.
Real network calls (search, HTTP) would also make runs cost money and
break reproducibility.

The LLM calls are still nondeterministic — that's the thing under test.
"""
from __future__ import annotations


# ---- math -----------------------------------------------------------------

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b if b else float("nan")


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


# ---- research / retrieval -------------------------------------------------

def web_search(query: str) -> str:
    """Mocked web search. Deterministic."""
    return f"(mock web result for query: {query!r})"


def read_file(filename: str) -> str:
    """Mocked file reader."""
    return f"(mock file content for: {filename})"


def fetch_url(url: str) -> str:
    """Mocked URL fetch."""
    return f"(mock page body from {url}, 480 words)"


# ---- authoring ------------------------------------------------------------

def write_code(language: str, intent: str) -> str:
    """Mocked code writer — returns a one-line stub."""
    return f"# {language} stub for: {intent}\nresult = None"


def make_chart(data_csv: str, chart_type: str) -> str:
    """Mocked chart generator."""
    return f"(mock {chart_type} chart rendered from {len(data_csv)} chars of CSV)"


def analyze_data(table_csv: str, instruction: str) -> str:
    """Mocked data analysis."""
    return f"(mock analysis of {len(table_csv)} chars: {instruction})"


def summarize(text: str) -> str:
    """Mocked summarizer."""
    return f"(summary of {len(text)} chars)"


def draft_document(topic: str, sections: str) -> str:
    """Mocked long-form drafter."""
    return f"(draft on {topic!r} with sections: {sections})"


def critique(text: str) -> str:
    """Mocked critic tool."""
    return f"(critique of {len(text)} chars: consider adding concrete examples)"


__all__ = [
    "add", "analyze_data", "critique", "divide", "draft_document",
    "fetch_url", "make_chart", "multiply", "read_file", "subtract",
    "summarize", "web_search", "write_code",
]

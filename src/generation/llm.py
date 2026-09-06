"""
LLM Service
============
Interfaces with the **Ollama** local API to generate grounded answers
based on retrieved context chunks.

Qwen3 4B specific notes
-----------------------
* ``think=False`` is passed on every request to suppress Qwen3's built-in
  chain-of-thought reasoning mode.  Thinking mode emits ``<think>...</think>``
  blocks before the actual answer, which wastes tokens and adds latency with
  no benefit for grounded document-lookup RAG.  The flag is the official
  Ollama API knob — see https://ollama.com/library/qwen3.

* A post-processing stripper is applied as a safety net in case an older
  Ollama version doesn't honour the flag, or the model emits stray tags.

* Context is wrapped in ``<context>`` / ``</context>`` XML tags.  Qwen3
  was trained with structured XML delimiters, so it attends to them more
  reliably than the plain ASCII ``--- CONTEXT START ---`` fences used for
  Gemma.
"""

import logging
import re
from typing import Optional

import ollama

from src.utils.errors import LLMServiceError
from src.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Qwen3 think-tag stripper
# ---------------------------------------------------------------------------
# Qwen3 can emit <think>...</think> reasoning blocks before the answer even
# when think=False is set (depends on Ollama version / model variant).
# This regex removes the entire block so only the clean answer is returned.
# The DOTALL flag makes '.' match newlines inside the block.
_THINK_TAG_RE = re.compile(r"^(?:<think>)?.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """Remove any Qwen3 <think>...</think> reasoning blocks from *text*."""
    cleaned = _THINK_TAG_RE.sub("", text)
    # Collapse the leading blank lines that the removal leaves behind
    return cleaned.lstrip("\n").strip()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# Qwen3 4B responds well to concise, directive system prompts.  The model
# was trained with instruction-following so it honours role descriptions and
# structured rules without needing verbose repetition.  Keep the system
# prompt focused — overly long prompts dilute attention on small models.
_SYSTEM_PROMPT: str = """\
You are a strict, grounded document analysis assistant. Your SOLE purpose is to extract and format information from the provided <context> blocks.

## Core Grounding Rules (CRITICAL)
1. You must base your answer **EXCLUSIVELY** on the text inside the <context> tags.
2. WARNING: UNDER NO CIRCUMSTANCES should you use external knowledge, pre-trained information, or make up facts.
3. If the answer cannot be found in the <context>, you must refuse and say exactly: "The requested information is not found in the uploaded documents."
4. Do not guess, infer, or hallucinate. If the context only partially answers the question, state what is known and explicitly state what is missing.
5. Always be concise, accurate, and factual.

## Response Format Guidelines

### Default: Natural Prose
- Start with prose for all answers
- Use clear paragraphs for explanations, definitions, conceptual content
- Prose is the safest default; it works for 90% of queries

### Lists (When Natural)
- Use **bullet points** for:
  - Unordered collections (features, benefits, examples)
  - Sets of independent items
- Use **numbered lists** for:
  - Sequential steps, workflows, processes
  - Ranked/priority items
- Only list when the answer naturally decomposes; don't force it

### Tables (Selective Use Only)
Use tables **only if all of these are true:**
1. The question explicitly requests comparison or structured format ("compare X vs Y", "show as table", "matrix")
2. **OR** the source document contains a table and you're directly referencing it
3. **AND** the data has 2+ dimensions that benefit from alignment (rows + columns with parallel structure)
4. **AND** the table has 3+ meaningful rows AND 2+ meaningful columns (avoid trivial 2x2 tables in prose)

**Example triggers:**
- ✅ "Compare these products" + data naturally aligns → table
- ✅ Source doc has a pricing table + user asks about it → reference the table
- ❌ "What is X?" with two facts → use prose, not a 2x2 table
- ❌ "List benefits" → use bullets, not a table
- ❌ "How does X work?" → use prose narrative, not pseudo-table

### Multi-Format Responses
For complex answers that involve multiple elements:
1. **Lead with prose** (overview, context-setting)
2. **Then add structure** (lists, tables) only where they clarify
3. **End with prose** (implications, next steps)

### For Multi-Part Questions
1. Identify each part
2. Answer each part in natural format (prose/lists as appropriate)
3. Use tables only if ONE part explicitly asks for structured format
4. Maintain narrative flow throughout

Example:
Q: 'What is Concept X? Compare it to Concept Y in a table.'
A: [Prose definition of Concept X]
   [Table comparing X and Y — ONLY this part is tabular]
   [Prose elaborating on context or use cases]

## Citation & Sourcing
- Cite sources when referencing specific claims: "According to [Source X]..."
- For tables from source documents, include the original document reference
- For synthesized comparisons (not from a single source table), cite multiple sources as needed
- Don't over-cite trivial facts; focus on non-obvious or quantitative claims

## Handling Document Type Variations
- **Documents WITH Tables:** Preserve table structure if the user references that specific data. Don't regenerate tables as prose; cite the original. Use prose to interpret or explain table findings.
- **Documents WITHOUT Tables:** Never force prose into pseudo-table format. Use natural language and lists appropriately.
- **Mixed Content:** Respect both formats in source. Lead with the most relevant structure for the question.

## Anti-Patterns (What NOT to Do)
- ❌ Force every answer into a table
- ❌ Create pseudo-tables for two-item lists
- ❌ Ignore the user's question format in favor of document structure
- ❌ Use tables without citing source or justifying structure
- ❌ Hallucinate data not in context
- ❌ Over-cite trivial statements
"""


class LLMService:
    """
    Sends context-augmented prompts to Ollama and returns the generated answer.
    """

    def __init__(self) -> None:
        try:
            self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            self.model = settings.MODEL_NAME
            logger.info("Ollama client initialised (model=%s, url=%s)", self.model, settings.OLLAMA_BASE_URL)
        except Exception as exc:
            logger.error("Failed to initialise Ollama client: %s", exc)
            raise LLMServiceError(f"Ollama client init failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  Answer Generation
    # ------------------------------------------------------------------ #

    def generate_answer(self, question: str, context: str) -> str:
        """
        Call the Ollama API with the user's *question* and retrieved *context*.

        The system prompt forces the model to answer **only** from the
        supplied context and to cite sources.

        Args:
            question: The user's natural-language question.
            context:  Concatenated text passages retrieved from FAISS.

        Returns:
            The generated answer string.

        Raises:
            LLMServiceError: If the Ollama API call fails.
        """
        user_message: str = (
            f"Below are context passages retrieved from the document.\n"
            f"\n"
            # Qwen3 attends reliably to XML-style context delimiters because
            # the model was trained with structured tags.  These replace the
            # plain ASCII fences (--- CONTEXT START ---) used for Gemma.
            f"<context>\n"
            f"{context}\n"
            f"</context>\n"
            f"\n"
            f"Question: {question}\n"
        )

        logger.info(
            "Calling Ollama API (model=%s) — question length=%d, context length=%d",
            self.model,
            len(question),
            len(context),
        )
        logger.info("FULL CONTEXT SENT TO MODEL:\n%s", context)

        try:
            # We use chat() so the system / user roles are properly set.
            #
            # think=False — Qwen3-specific: disables the internal chain-of-
            # thought reasoning mode. Thinking adds <think>...</think> blocks
            # before the answer; for grounded RAG this is pure overhead.
            # Requires ollama-python >= 0.4.x (currently 0.6.2).
            # _strip_think_tags() below is kept as a safety net regardless.
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                think=False,  # Qwen3: suppress chain-of-thought reasoning mode
            )

            # Extract content from response — ollama-python >= 0.4.x returns
            # a Pydantic ChatResponse object, not a plain dict.
            # Use attribute access: response.message.content
            raw_answer: str = response.message.content or ""
            logger.info("RAW MODEL OUTPUT BEFORE CLEANUP:\n%s", raw_answer)

            # Safety net: strip any stray <think> blocks that slip through
            # (can happen with older Ollama builds that ignore think=False).
            answer: str = _strip_think_tags(raw_answer)
            if len(answer) < len(raw_answer):
                logger.debug(
                    "Stripped <think> block from Qwen3 response "
                    "(%d chars removed)",
                    len(raw_answer) - len(answer),
                )

            # Extract token details for logging
            input_tokens = response.prompt_eval_count or 0
            output_tokens = response.eval_count or 0
            logger.info(
                "Ollama response received — input_tokens=%d, output_tokens=%d",
                input_tokens,
                output_tokens,
            )

            return answer

        except Exception as exc:
            logger.error("Error during Ollama call: %s", exc)
            raise LLMServiceError(
                f"Failed to get a response from Ollama: {exc}"
            ) from exc

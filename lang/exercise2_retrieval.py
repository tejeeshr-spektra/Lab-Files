"""
Exercise 2: LangChain Tools & Retrieval Chain
----------------------------------------------
Wraps the Azure AI Search index (created in Exercise 1) as a LangChain
retriever, exposes it as a Tool, and builds a RetrievalQA chain on top of
Azure OpenAI.

Complete the two TODO sections marked:
    # Task 1: Implement Search Retriever Tool
    # Task 2: Build RetrievalQA Chain

Run with:
    python exercise2_retrieval.py
"""

import os
import warnings
from typing import Any, List, Optional

from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import Tool
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from pydantic import PrivateAttr

# Keep the console output readable (LangChain prints deprecation notices).
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning

    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:  # pragma: no cover
    pass

# Load values from the .env file in this folder
load_dotenv()


# ---------------------------------------------------------------------------
# Provided helper: Azure AI Search retriever (do not modify)
# ---------------------------------------------------------------------------
class AzureAISearchRetriever(BaseRetriever):
    """LangChain retriever that runs a keyword search on the Azure AI Search index."""

    k: int = 5
    content_max_chars: int = 3000  # keep prompts small enough for the model

    _client: Any = PrivateAttr(default=None)

    # Fields that can hold the main text of a document, in order of preference
    _CONTENT_FIELDS = ("merged_content", "content", "text", "chunk")

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        key = os.getenv("AZURE_SEARCH_KEY")
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

        missing = [
            name
            for name, value in {
                "AZURE_SEARCH_ENDPOINT": endpoint,
                "AZURE_SEARCH_KEY": key,
                "AZURE_SEARCH_INDEX_NAME": index_name,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                f"Missing values in .env: {', '.join(missing)}. "
                "Create the .env file as described in the lab."
            )

        self._client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(key),
        )

    def _to_document(self, result: dict) -> Document:
        """Convert a search result into a LangChain Document."""
        content = ""
        for field in self._CONTENT_FIELDS:
            if result.get(field):
                content = str(result[field])
                break

        lines = []
        file_name = result.get("metadata_storage_name")
        if file_name:
            lines.append(f"File: {file_name}")
        if content:
            lines.append(content[: self.content_max_chars])

        metadata = {"score": result.get("@search.score")}
        for field, value in result.items():
            if field.startswith("@") or field in self._CONTENT_FIELDS:
                continue
            if value in (None, "", []):
                continue
            metadata[field] = value
            # Add AI enrichment output (people, organizations, key phrases...)
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                label = field.replace("_", " ").title()
                lines.append(f"{label}: {', '.join(value[:20])}")

        return Document(page_content="\n".join(lines), metadata=metadata)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k: Optional[int] = None,
    ) -> List[Document]:
        results = self._client.search(search_text=query, top=k or self.k)
        return [self._to_document(r) for r in results]


# ---------------------------------------------------------------------------
# Task 1: Implement Search Retriever Tool
# ---------------------------------------------------------------------------
# TODO: Add the InvoiceSearchTool class here.



# ---------------------------------------------------------------------------
# Task 2: Build RetrievalQA Chain
# ---------------------------------------------------------------------------
# TODO: Add the InvoiceRetrievalQA class here.



# ---------------------------------------------------------------------------
# Test run (do not modify)
# ---------------------------------------------------------------------------
EXAMPLE_QUERIES = [
    "List the invoices and the customers they were issued to.",
    "What products or services were billed and what was the total amount?",
    "Which invoice has the highest total amount?",
]


def main() -> None:
    print("=" * 70)
    print("STEP 1: Testing the invoice search tool (retriever only)")
    print("=" * 70)
    search_tool = InvoiceSearchTool()
    tool = search_tool.get_langchain_tool()
    print(f"Tool created: {tool.name}\n")
    print(search_tool.search_invoices("invoice total")[:1500])

    print("\n" + "=" * 70)
    print("STEP 2: Testing the RetrievalQA chain (retriever + LLM)")
    print("=" * 70)
    qa = InvoiceRetrievalQA()
    for question in EXAMPLE_QUERIES:
        print(f"\nQuestion: {question}")
        result = qa.query(question)
        print(f"Answer: {result['answer']}")
        sources = {
            d.metadata.get("metadata_storage_name", "unknown")
            for d in result["source_documents"]
        }
        print(f"Sources: {', '.join(sorted(sources)) or 'none'}")
        print("-" * 70)


if __name__ == "__main__":
    main()

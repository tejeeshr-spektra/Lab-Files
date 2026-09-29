"""
Exercise 3: Conversational Agent with Memory
--------------------------------------------
Builds a LangChain agent that uses the invoice_search tool from Exercise 2
and remembers the conversation, so follow-up questions work
("What was the total on that one?").

Requires exercise2_retrieval.py to be completed first.

Run with:
    python exercise3_agent_memory.py
"""

import os
import warnings

from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from exercise2_retrieval import InvoiceSearchTool

warnings.filterwarnings("ignore")
load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful assistant for invoice questions. "
    "Always use the invoice_search tool to look up facts before answering. "
    "Answer only from the search results; if the answer is not there, say so. "
    "Use the chat history to understand follow-up questions."
)


class SearchInput(BaseModel):
    query: str = Field(description="Keywords or question to search the invoices for")


def build_search_tool() -> StructuredTool:
    """Reuse the Exercise 2 tool, with an explicit input schema for tool calling."""
    search = InvoiceSearchTool()
    base = search.get_langchain_tool()
    return StructuredTool.from_function(
        func=lambda query: search.search_invoices(query),
        name=base.name,
        description=base.description,
        args_schema=SearchInput,
    )


class InvoiceAgent:
    """Tool-calling agent with conversation memory."""

    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0,
        )

        self.tools = [build_search_tool()]

        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )

    def chat(self, message: str) -> str:
        try:
            return self.executor.invoke({"input": message})["output"]
        except Exception as e:
            return f"Error processing message: {e}"

    def clear_memory(self) -> None:
        self.memory.clear()

    def get_history(self):
        return self.memory.chat_memory.messages


def main() -> None:
    agent = InvoiceAgent()

    print("=" * 70)
    print("Demo: the second question relies on memory of the first")
    print("=" * 70)
    for question in [
        "Find the invoice with the highest total amount.",
        "Who was that invoice issued to?",
    ]:
        print(f"\nYou: {question}")
        print(f"Agent: {agent.chat(question)}")

    print("\n" + "=" * 70)
    print("Interactive chat. Type 'clear' to reset memory, 'exit' to quit.")
    print("=" * 70)
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "clear":
            agent.clear_memory()
            print("Memory cleared.")
            continue
        print(f"Agent: {agent.chat(user_input)}")


if __name__ == "__main__":
    main()

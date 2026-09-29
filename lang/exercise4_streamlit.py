"""
Exercise 4: Streamlit Chat UI
-----------------------------
A web chat front end for the invoice chatbot.

Requires exercise2_retrieval.py and exercise3_agent_memory.py.

Run with:
    streamlit run exercise4_streamlit.py
"""

import streamlit as st
from dotenv import load_dotenv

from exercise2_retrieval import InvoiceRetrievalQA
from exercise3_agent_memory import InvoiceAgent

load_dotenv()

st.set_page_config(page_title="Invoice Chatbot", page_icon="🧾", layout="centered")
st.title("🧾 Knowledge-Augmented Invoice Chatbot")
st.caption("LangChain + Azure AI Search + Azure OpenAI")

# ---- Session state -------------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = InvoiceAgent()
if "qa" not in st.session_state:
    st.session_state.qa = InvoiceRetrievalQA()
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Chat mode",
        ["Agent with memory", "RetrievalQA (single question)"],
        help="The agent remembers earlier messages. RetrievalQA answers each question on its own and shows sources.",
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.agent.clear_memory()
        st.rerun()

    st.markdown("**Try asking:**")
    st.markdown(
        "- Which invoice has the highest total?\n"
        "- Who was it issued to?\n"
        "- What products were billed?"
    )

# ---- Chat history --------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# ---- Input ---------------------------------------------------------------
if prompt := st.chat_input("Ask about your invoices..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching invoices..."):
            sources = []
            if mode == "Agent with memory":
                answer = st.session_state.agent.chat(prompt)
            else:
                result = st.session_state.qa.query(prompt)
                answer = result["answer"]
                sources = sorted(
                    {
                        d.metadata.get("metadata_storage_name", "unknown")
                        for d in result["source_documents"]
                    }
                )
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

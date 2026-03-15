"""
ui/streamlit_app.py
--------------------
FIXES:
  [12] page_icon garbled UTF-8 emoji → correct 🏢
  [13] badge string missing intent variable → f"{intent} agent"
"""
import streamlit as st
import httpx

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="HR Intelligence",
    page_icon="🏢",              # FIX [12]: was "ð¢" (garbled UTF-8)
    layout="centered",
)
st.title("HR Intelligence Assistant")
st.caption("Powered by LangGraph multi-agent RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + " | ".join(msg["sources"]))

if prompt := st.chat_input("Ask your HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = httpx.post(
                    API_URL + "/chat",
                    json={"query": prompt, "history": history},
                    timeout=60,
                )
                resp.raise_for_status()
                data    = resp.json()
                answer  = data["answer"]
                sources = data.get("sources", [])
                intent  = data.get("intent", "unknown")
                cached  = data.get("cached", False)

                st.markdown(answer)

                if sources:
                    st.caption("Sources: " + " | ".join(sources))

                # FIX [13]: was ' agent' — intent was missing
                badge = f"{intent} agent" + (" *(cached)*" if cached else "")
                st.caption(badge)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except httpx.HTTPStatusError as e:
                st.error("API error: " + str(e))
            except Exception as e:
                st.error("Error: " + str(e))

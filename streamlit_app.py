import json
import os
import streamlit as st
from ddgs import DDGS
from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def search_tool(query: str):
    """Perform contextual web search using DuckDuckGo (DDGS) with relevance filters."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region="pk-en", safesearch="moderate", max_results=10):
            results.append(r)

    valid_results = [
        r for r in results
        if "current.com" not in r["href"].lower()
        and not r["href"].startswith(("https://apps.apple.com", "https://play.google.com"))
        and "list of" not in r["title"].lower()
    ]

    if not valid_results:
        return {"query": query, "answers": ["No relevant results found."]}

    top_results = valid_results[:3]
    formatted_answers = [
        f"{r['title']} — {r['body']} (Source: {r['href']})"
        for r in top_results
    ]

    return {"query": query, "answers": formatted_answers}

def handle_question(user_input: str):
    """Ask Groq model and execute tool calls automatically. Returns response data for Streamlit display."""
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a reasoning assistant with access to a web search tool.\n\n"
                    "If the user asks for factual, real-world, or up-to-date information, "
                    "you MUST respond ONLY with a JSON object in this format:\n\n"
                    "{\n"
                    "  \"tool_call\": {\n"
                    "    \"name\": \"search_tool\",\n"
                    "    \"arguments\": {\"query\": \"<rephrased, clear question>\"}\n"
                    "  }\n"
                    "}\n\n"
                    "If the question is purely conceptual or logical, respond directly in text instead of JSON."
                )
            },
            {"role": "user", "content": user_input}
        ],
    )

    response = completion.choices[0].message.content.strip()

    try:
        tool_call = json.loads(response)
        if "tool_call" in tool_call:
            search_result = search_tool(**tool_call["tool_call"]["arguments"])

            # 👇 Added: Build full conversation JSON for display
            conversation_log = {
                "conversation": [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "tool_call": tool_call["tool_call"]},
                    {"role": "tool", "tool_response": {
                        "name": "search_tool",
                        "output": search_result
                    }},
                    {"role": "assistant", "content": "Here are the top results I found from the web search."}
                ]
            }

            return {
                "type": "tool_call",
                "tool_call": tool_call,
                "search_result": search_result,
                "conversation_log": conversation_log  # 👈 New key
            }
        else:
            return {
                "type": "no_tool_call",
                "response": response
            }

    except json.JSONDecodeError:
        return {
            "type": "direct_output",
            "response": response
        }

# Streamlit App
st.title("🤖 Groq Assistant with Web Search")

st.write("Ask me anything! I'll use web search for factual questions.")

user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_input:
        with st.spinner("Thinking..."):
            response_data = handle_question(user_input)

        if response_data["type"] == "tool_call":
            st.subheader("🧩 Tool call detected:")
            st.json(response_data["tool_call"])
            st.subheader("🔍 Search Result:")
            st.json(response_data["search_result"])

            # 👇 Added: show the full conversation trace
            st.subheader("🧠 Full AI Interaction Log:")
            st.json(response_data["conversation_log"])

            answers = response_data["search_result"].get("answers", [])
            if len(answers) == 1 and "No relevant" in answers[0]:
                st.write(f"💬 Assistant: {answers[0]}")
            else:
                st.write("💬 Assistant: Here are the top results:")
                for i, ans in enumerate(answers, start=1):
                    st.write(f"{i}. {ans}")

        elif response_data["type"] == "no_tool_call":
            st.write("💬 Model output (no tool call):", response_data["response"])
        elif response_data["type"] == "direct_output":
            st.write("💬 Model output:", response_data["response"])
    else:
        st.warning("Please enter a question.")

import json
import os
from ddgs import DDGS
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def search_tool(query: str):
    """Perform contextual web search using DuckDuckGo (DDGS) with relevance filters."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region="pk-en", safesearch="moderate", max_results=10):
            results.append(r)

    # Filter out irrelevant or generic pages
    valid_results = [
        r for r in results
        if "current.com" not in r["href"].lower()
        and not r["href"].startswith(("https://apps.apple.com", "https://play.google.com"))
        and "list of" not in r["title"].lower()
    ]

    if not valid_results:
        return {"query": query, "answers": ["No relevant results found."]}

    # Keep top 3 results
    top_results = valid_results[:3]

    formatted_answers = [
        f"{r['title']} — {r['body']} (Source: {r['href']})"
        for r in top_results
    ]

    return {"query": query, "answers": formatted_answers}


# --- MAIN HANDLER ---
def handle_question(user_input: str):
    """Ask Groq model and execute tool calls automatically."""
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
                    "Always rewrite vague or short queries into complete, context-rich ones. Examples:\n"
                    "  - 'prime minister of Pakistan' → 'who is the incumbent prime minister of Pakistan as of today'\n"
                    "  - 'lahore weather' → 'current weather in Lahore, Pakistan today'\n"
                    "  - 'top wwe stars' → 'top 5 current WWE superstars in 2025'\n\n"
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
            print("🧩 Tool call detected:")
            print(json.dumps(tool_call, indent=2))

            args = tool_call["tool_call"]["arguments"]
            search_result = search_tool(**args)

            print("\n🔍 Search Result:")
            print(json.dumps(search_result, indent=2))

            # Print results nicely
            answers = search_result.get("answers", [])
            if len(answers) == 1 and "No relevant" in answers[0]:
                print(f"\n💬 Assistant: {answers[0]}")
            else:
                print("\n💬 Assistant: Here are the top results:")
                for i, ans in enumerate(answers, start=1):
                    print(f"{i}. {ans}")
        else:
            print("💬 Model output (no tool call):", response)

    except json.JSONDecodeError:
        print("💬 Model output:", response)


# --- INTERACTIVE LOOP ---
if _name_ == "_main_":
    print("🤖 Ask me anything (type 'exit' to quit):")
    while True:
        user_input = input("> ")
        if user_input.lower() in ("exit", "quit"):
            break
        handle_question(user_input)

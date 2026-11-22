from typing import Callable, Optional

from agent import (
    build_critic_chain,
    build_reader_agent,
    build_search_agent,
    build_writer_chain,
    get_llm,
)

STEPS = ["search", "reader", "writer", "critic"]


def run_research_pipeline(
    topic: str,
    model: str = "gpt-4o-mini",
    on_step: Optional[Callable[[str, str, Optional[str]], None]] = None,
) -> dict:
    """Run search -> reader -> writer -> critic.

    on_step(step, status, output) is called with status "start" or "done" so a UI
    can show live progress. Works fine without a callback (CLI use).
    """

    def notify(step, status, output=None):
        if on_step:
            on_step(step, status, output)

    llm = get_llm(model)
    state = {}

    # Step 1 - search
    notify("search", "start")
    search_agent = build_search_agent(llm)
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = search_result["messages"][-1].content
    notify("search", "done", state["search_results"])

    # Step 2 - reader
    notify("reader", "start")
    reader_agent = build_reader_agent(llm)
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:1500]}"
        )]
    })
    state["scraped_content"] = reader_result["messages"][-1].content
    notify("reader", "done", state["scraped_content"])

    # Step 3 - writer
    notify("writer", "start")
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )
    state["report"] = build_writer_chain(llm).invoke({
        "topic": topic,
        "research": research_combined,
    })
    notify("writer", "done", state["report"])

    # Step 4 - critic
    notify("critic", "start")
    state["feedback"] = build_critic_chain(llm).invoke({"report": state["report"]})
    notify("critic", "done", state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    result = run_research_pipeline(
        topic, on_step=lambda s, st, out: print(f"[{s}] {st}") or (out and print(out))
    )

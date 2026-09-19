# Lab 1.4 – Framework Comparison: Tradeoffs

Fill in each section after running all three framework implementations
with `python compare.py`. Your answers should be specific and grounded
in what you observed, not generic descriptions from documentation.

---

## 1. Quantitative Results

Copy the comparison table printed by `compare.py` here.

```
===========================================================================
	Framework    Status         Steps     Prompt   Completion      Total
---------------------------------------------------------------------------
	LangGraph    ok                15     26 932        2 147     29 079
	AutoGen      ok                 1     18 341        1 285     19 626
	CrewAI       ok                 1     33 512        2 373     35 885
===========================================================================
```

---

## 2. Developer Experience

### LangGraph
- How much boilerplate did you write?
- How easy was it to understand what the agent was doing at each step?
- What was the hardest part to implement?

LangGraph required the most explicit wiring and the most code, because the state, routing, tool execution, and termination logic were all implemented directly. It was the easiest to reason about step by step because every transition was visible in the graph, but it was also the hardest to get right because the agent loop, tool calls, and `done` handling had to be coordinated manually. The hardest part was making the model call, tool dispatch, and state updates line up without creating an infinite loop.

### AutoGen
- How much boilerplate did you write?
- How easy was it to understand what the agent was doing at each step?
- What was the hardest part to implement?

AutoGen required less boilerplate than LangGraph, but more than CrewAI. The conversation flow was easier to follow than a manual graph because the framework handled the turn-taking, but it was still necessary to register tools separately for LLM use and execution. The hardest part was getting the agent configuration and termination logic correct, especially around the tool registration and the chat result structure.

### CrewAI
- How much boilerplate did you write?
- How easy was it to understand what the agent was doing at each step?
- What was the hardest part to implement?

CrewAI had the least code in the agent/task setup, but the most framework-specific configuration issues. It was the least transparent at runtime because the framework abstracted away the conversation flow and tool orchestration. The hardest part was getting the LLM/provider configuration correct for Azure versus OpenRouter, and then matching the CrewAI version’s expected `LLM` setup.

---

## 3. Token Efficiency

Which framework used the fewest tokens for the same task? Why do you
think that is? (Hint: look at the system prompts each framework injects
automatically vs. what you wrote yourself.)

AutoGen used the fewest tokens for the same task at 19,626 total tokens. LangGraph used 29,079 total tokens, and CrewAI used the most at 35,885 total tokens. AutoGen was likely the most efficient because it completed in a single recorded step and had a relatively compact conversation flow. CrewAI likely used the most because it injected more framework-managed context and produced a larger prompt footprint. LangGraph was in the middle, but it still consumed a lot because the agent loop ran up to 15 steps.

---

## 4. Control vs. Abstraction

Rank the three frameworks from most explicit control (you decide
everything) to highest abstraction (the framework decides most things).
Justify your ranking with one concrete example from your implementation.

Most explicit control: LangGraph. Middle: AutoGen. Highest abstraction: CrewAI. LangGraph was the most explicit because I defined the state, the routing function, the tool node, and the termination condition directly. AutoGen abstracted the conversation loop but still exposed tool registration and chat initiation. CrewAI was the highest abstraction because I mainly defined role, goal, tools, and task, while the framework handled most of the orchestration.

---

## 5. When would you use each?

Complete the table with one concrete use-case per framework.

| Framework | Best suited for |
|-----------|-----------------|
| LangGraph | Workflows that need precise control over state, branching, and tool execution. |
| AutoGen   | Multi-agent chat systems where turn-taking and tool use should be conversation-driven. |
| CrewAI    | Role-based agent teams where fast setup and higher-level orchestration matter more than fine-grained control. |

---

## 6. Surprises

What was the most surprising difference you found between the three
implementations? (One paragraph.)

The most surprising difference was how much the abstraction level changed both the implementation effort and the token usage. LangGraph felt the most predictable because every step was explicit, but it still used a lot of tokens because the loop kept running. CrewAI looked the simplest at first, but it was the most fragile in practice because provider configuration caused multiple failures before it worked. AutoGen ended up being the best balance for this task: it was easier to set up than LangGraph and more efficient than CrewAI, while still being understandable.

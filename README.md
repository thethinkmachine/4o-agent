# 4o-Operator | A Fully Autonomous Command Line Computer-Using Agent (CL-CUA) based on 0-shot ReAct principles 🤖
---
## What is it?
Large Language Models already excel at code generation & structured solutioning. So let's why not combine those abilities into creating an agent that can, maybe, pass bash commands? Or perhaps, execute python code in a REPL environment? or maybe just operate the entire computer on your behalf!

4o-Operator is a prototypical example of such a command line CUA powered by GPT 4o-mini and based on ReAct (Think, Reflect, Act, Observe) Agentic principles. Think of it as helpful AI that is not just limited to generating text, but has autonomy over what it can do on a computer. 4o-Operator features a comprehensive set of tools like Shell use, Python Code Execution, Web Scraping, File Management, API calling etc. in a Directed Acyclic Graph-like architecture. The LLM backend allows flexibility in tool use- It can even combine the outputs of different tools in any manner it thinks is desirable to achieve a certain goal.

# CAUTION ⚠️
---
👉 Any CUA carries a risk of prompt injections.
👉 While proprietary models like GPT 4o may not be significantly prone to such attacks, running a CUA on a local LLM backend carries a high risk.
👉 As such, it is strongly advised to run the agent within a containerized environment.

# AUTHOR 😀
---
**Shreyan C (@thethinkmachine)**
*(Made for a college project, opensourced to community 😀)*
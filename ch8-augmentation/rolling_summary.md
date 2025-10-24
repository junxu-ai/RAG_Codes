```pseudo
// Rolling-window conversational agent (single, simplified pseudocode)

STATE:
  summary      // compact long-term memory string
  messages[]   // short-term buffer since last summary
  memory       // optional structured memory (key facts, entities)
  DONE = false

PARAMS:
  MAX_MSGS      // e.g., 4 recent turns
  MAX_TOKENS    // e.g., 1500 tokens for messages buffer
  SUMMARIZE_EVERY_N_TURNS  // optional cadence

FUNCTION SummarizeIfNeeded():
  if messages.length == 0: return
  if messages.length >= MAX_MSGS OR TokenCount(messages) >= MAX_TOKENS
     OR TurnCount % SUMMARIZE_EVERY_N_TURNS == 0:
      // Only merge prior summary + latest messages (never the full history)
      prompt = BuildSummarizerPrompt(summary, messages)
      resp   = LLM(prompt)
      summary = ExtractSummary(resp)
      memory  = UpdateMemory(memory, resp)      // optional
      messages = []                              // clear short-term buffer

FUNCTION DecideAndAct(latest_user_input):
  // Orchestrator uses compact context + latest input
  decision_prompt = BuildDecisionPrompt(summary, latest_user_input, memory)
  plan = LLM(decision_prompt)

  if plan.type == "TOOL_CALL":
      tool_result = ExecuteTool(plan.tool, plan.args)
      messages.push({role:"tool", content: tool_result})
      return CONTINUE
  else if plan.type == "ANSWER":
      answer = plan.content
      OutputToUser(answer)
      messages.push({role:"assistant", content: answer})
      return CONTINUE
  else if plan.type == "END":
      return STOP
  else:
      return CONTINUE

MAIN LOOP:
  while NOT DONE:
    user_msg = WaitForUserInput()
    messages.push({role:"user", content: user_msg})

    SummarizeIfNeeded()

    result = DecideAndAct(user_msg)
    if result == STOP:
       DONE = true
       break

    SummarizeIfNeeded()  // post-action cleanup if buffer grew

// End
```

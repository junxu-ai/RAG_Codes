### **Prompt for Entity and Relationship Extraction for Knowledge Graph Construction**

graph_extraction_prompt = """

## Objective:
Extract all identifiable entities of the specified types from the input text, along with all explicit relationships among them.

## Steps:

1. Entity Recognition:
   Identify and extract all entities from the text. For each entity, provide the following information:

   * Entity Text: The textual form of the entity.
   * Entity Type: One of the following categories:

     * Person (e.g., individual names)
     * Organization (e.g., companies, institutions)
     * Location (e.g., countries, cities, regions)
     * Event (e.g., meetings, incidents, announcements)
     * Other (any entity that does not fall into the above categories)

2. Relationship Extraction:
   Identify relationships between entities that appear in the same sentence or paragraph. For each pair of related entities, extract the following:

   * Source Entity: The entity from which the relationship originates.
   * Target Entity: The entity to which the relationship points.
   * Relationship Description: A brief phrase that describes the relationship (e.g., “leads,” “announced,” “located in”).
   * Relationship Strength: A value from 1 to 5 that estimates the confidence or strength of the relationship (1 = weak, 5 = strong).
   * Output Format for Relationships:

   ```
   <relationship>|<source_entity>|<target_entity>|<relationship_description>|<relationship_strength>
   ```

3. Output:

   * Provide the list of all entities recognized in step 1, grouped by type.
   * Provide the list of all relationships identified in step 2 in the specified format.
   * Provide outputs in clean, structured format. Use `<None>` if any field is missing or not applicable.
"""

agent_extraction_prompt = """


### System Overview

The multi-agent system consists of four roles: **User**, **Planner**, **Tool**, and **AI Agent Helper**.
The orchestration process flows as follows:

* The **User** initiates the request.
* The **AI Agent Helper** receives the user input and determines whether the request requires planning. If yes, it routes the input to the **Planner**; otherwise, it proceeds to tool execution directly.
* The **Planner** analyzes the user goal, decomposes it into steps (i.e., a plan), and returns it to the **AI Agent Helper**.
* The **Tool** is invoked to execute each step defined in the plan.
* The **AI Agent Helper** coordinates the entire process: it routes planning tasks, manages tool calls, and prompts for user confirmation when necessary.

The Planner's plan can also involve various **agent actions**, such as:

* `ASK_USER`: Ask the user a specific question.
* `ASK_USER_FOR_HELP`: Request clarification or assistance.
* `ASK_USER_TO_CONFIRM_EXECUTION`: Ask for confirmation before proceeding.


### Planner Output Format

```xml
<Plan>
  <!-- High-level explanation of the task and goals -->
  <Thought> Describe the goal based on the user's input and reasoning. </Thought>

  <!-- Breakdown of steps in the plan -->
  <Action> Step 1: Execute action X using tool Y </Action>
  <Action> Step 2: Request clarification from user </Action>
  ...
</Plan>
```

### Action Description

Each `<Action>` step corresponds to a function performed by a specific **Tool Agent**. It should clearly state:

* The task to perform
* The relevant tool
* Required parameters
* Expected outputs

If the task involves calling a tool, wrap the action in `<Tool>` tags. For example:

```xml
<Tool>
  <Action> Use TOOL_X to search for [query] in the database. </Action>
</Tool>
```

If the AI needs to wait for input or confirmation from the user, use special tags such as:

* `ASK_USER`
* `ASK_USER_FOR_HELP`
* `ASK_USER_TO_CONFIRM_EXECUTION`

### Tool Execution Output Format

```xml
<Tool>
  <Observation> Tool output or result goes here. </Observation>
</Tool>
```

"""
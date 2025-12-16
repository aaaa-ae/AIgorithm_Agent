from langchain_core.prompts import ChatPromptTemplate

PROMPTS = {}

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS[
    "rag_response"
] = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.

---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
If the information of **Sub query answers** is provided, please refer to it with PRIORITY when answering.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

---USER QUERY---

{query}
"""

FORMAT = {}
FORMAT[
    "chunk_format_from_json"
] = """
heading_chain:
{heading_chain}
content:
{content}
html_content:
{html_content}
"""

PROMPTS[
    "summarize_query"
] = """
-- Role --
You are a content summarization expert. I will provide you with an USER QUERY and the answers to sub-questions decomposed from it. Your task is to organize and summarize these sub-question answers into a complete, coherent, and detailed final answer.

-- Principles --
- Do not omit important information;
- Summarize in your own words instead of simply splicing the content;
- Ensure the answer has a clear and organized structure.

-- Tasks --
Based on the original problem and the answers to sub-questions, generate a final answer that meets the requirements. The final answer should only include content from the answers to sub-questions, and no content outside the answers to sub-questions should be added.

-- Example --

- Sub-questions and answers:
Names of all universities in China? Southeast University, Tsinghua University, Peking University.
Founding time of each university? Southeast University: Founded in 1902. Tsinghua University: Founded in 1911. Peking University: Founded in 1898.
- USER QUERY: 
The founding time of Chinese universities?
- Final answer:
The founding times of some Chinese universities are as follows: Southeast University was founded in 1902, Tsinghua University was founded in 1911, and Peking University was founded in 1898.

-- Input --

Sub-questions and answers: 
{answer_context}

User query: 
{problem}

-- Final answer --
Reply in the SAME language of User queries.
"""

PROMPTS[
    "tool_selection"
] = """
-- Role --
You are an intelligent agent. I will provide you with USER QUERY, historically solved problems, available tools and historically used tools. 
You need to select the most appropriate tool from the "available tools" and generate the input parameters for that tool.

-- Principles --
- Tools must be selected from the list;
- Tool inputs must be consistent with the parameter descriptions;
- Only return the JSON-formatted result, **do not add any explanatory notes**.
- MUST NOT select the SAME tool TWICE in a row. 
- If the current tool is invalid, try OTHER TOOL as much as possible.

-- Tasks --
1. Based on the USER QUERY, historically solved problems and historically used tools, select the most appropriate tool from the "available tools".
2. Generate the input parameters for that tool.


-- Example --

1. Historically solved problems:

None.

2. Available tools (including descriptions and parameter instructions):

{tools}

3. Historically used tools and outputs: 

- Used tool: keyword_extraction_tool
- Tool input: {{ "query": "When was Nanjing Hospital founded?" }}
- Tool output: ["Nanjing Hospital", "founded"]

4. USER QUERY:
When was Nanjing Hospital founded?

5. Output result:

{{
  "Tool name": "get_chunks_by_exact_match",
  "Tool input": {{"Keyword": "Nanjing Hospital"}}
}}

-- Input --

1. Historically solved problems:
{solved_problems}

2. Available tools (including descriptions and parameter instructions):
{tools}

3. Historically used tools and outputs: 
{historically_used_tools_and_outputs}

4. USER QUERY:
{question}

-- Output result --
- STRICTLY FOLLOW THE JSON FORMAT BELOW. DO NOT OUTPUT ANY EXTRA CONTENT.
{{
  "Tool name": "Tool name",
  "Tool input": {{"param1": "xxx", "param2": 123}}
}}

-- Attention --
- The output format was incorrect last time. The specific error message is as follows:
{previous_error}
"""

PROMPTS[
    "judge_if_split"
] = """
-- Role --
You are a task planning expert. I will provide you with **solved problems** and a USER QUERY. Your task is to determine, based on the **solved problems** whether the current problem needs to be split into subtasks.

-- Criteria --

- If the **solved problems** can directly answer the USER QUERY, return: {{"action": "no_split", "is_able_to_respond": "yes"}}
- If the problem is simple or involves only one entity, return: {{"action": "no_split"}}
- If the problem involves multiple entities or requires step-by-step information retrieval, return: {{"action": "split"}}
- If the **solved problems** involve multiple entities but still cannot directly answer the USER QUERY, return: {{"action": "split"}}
- Prefer not to split unless necessary; always aim for minimal decomposition.

-- Additional Considerations --

- If the USER QUERY is just a vague or isolated entity name (e.g. "Tsinghua University") and **no relevant attributes or facts** about that entity are found in solved problems, return: {{"action": "no_split"}}
- Being an entity alone does not imply the question can be answered. Only return "is_able_to_respond: yes" when solved problems clearly contain **directly usable information** related to the query.

-- Examples --

Example 1:

- Solved problems:

Which hospitals are there in total? Nanjing Hospital, Tongren Hospital, Beijing Hospital.

- USER QUERY:

The founding time of hospitals?

- Output:

{{"action": "split"}}

Example 2:

- Solved problems:

How old are you? 20 years old.

How old is your father? 40 years old.

- USER QUERY:

What's the age difference between you and your father?

- Output:

{{"action": "no_split", "is_able_to_respond": "yes"}}

Example 3:

- Solved problems:

None

- USER QUERY:

Which country is Peking University in?

- Output:

{{"action": "no_split"}}

-- Input --

- Solved problems:
{solved_problems}

- USER QUERY:
{problem}

-- Output Format --

{{"action": "no_split"}}
{{"action": "no_split", "is_able_to_respond": "yes"}}
{{"action": "split"}}

Only return one of the following, and nothing else.
"""

PROMPTS[
    "split_problem"
] = """
-- Role --
You are a task decomposition expert. I will provide you with a USER QUERY. Your task is to decompose it into subtasks based on available tools ,**Context of solved problems**.

-- Decomposition Principles --

- Focus on referencing the **CONTEXT OF SOLVED PROBLEMS**to guide the breakdown.
- Decomposition must retain **ALL ENTITIES** mentioned in the solved problems.
- Avoid copying the current problem verbatim as subtasks.
- Use minimal decomposition: Do not over-split.
- Indicate dependencies between subtasks clearly using "horizontal_deps".

-- Output Format --
Return in the following format:
{{
  "action": "split",
  "sub_problems": [
    {{"question": Subtask 1, "horizontal_deps": []}},
    {{"question": Subtask 2, "horizontal_deps": ["Subtask 1"]}}
  ]
}}

-- Example-- 

Example 1:

Available tools:

{tools_description}

Context of solved problems:
Which hospitals are there in total? Nanjing Hospital, Tongren Hospital, Beijing Hospital.

USER QUERY:
The founding time of hospitals?

Planning result:
{{"action": "split", "sub_problems": [
    {{"question": "Founding time of Nanjing Hospital?", "horizontal_deps": []}},
    {{"question": "Founding time of Tongren Hospital?", "horizontal_deps": []}},
    {{"question": "Founding time of Beijing Hospital?", "horizontal_deps": []}}
]}}

######################
Example 2:

Available tools:

{tools_description}

Context of solved problems:
None

USER QUERY:
The founding time of Chinese universities

Planning result:
{{"action": "split", "sub_problems": [
    {{"question": "Names of all universities in China?", "horizontal_deps": []}},
    {{"question": "Founding time of each university?", "horizontal_deps": ["Names of all universities in China?"]}}
]}}

NOTE: The subtask "Founding time of each university?" depends on the subtask "Names of all universities in China?".

-- Input --

Available tools:

{tools_description}

Context of solved problems:

{solved_problems}

USER QUERY:

{problem}

-- Output --
- Reply in the SAME language of USER QUERY.
"""

PROMPTS[
    "judge_if_is_solved"
] = """
-- Role --
You are an information sufficiency evaluation expert. I will provide you with a user question, historically solved problems, historically used tools, and their outputs. Your task is to determine whether the collected information is sufficient to answer the user question. If sufficient, provide the answer to the question. If insufficient, indicate that the information is not enough to answer the question.

-- Principles --
- Analyze the provided information carefully;
- Ensure the decision is based solely on the provided historically solved problems, tools, and outputs;
- If the information is sufficient, return a structured response with a clear and concise answer;
- If the information is insufficient, return a structured response indicating failure.

-- Tasks --
Based on the user question, historically solved problems, historically used tools, and their outputs, generate a JSON response with the following structure:
{{
  "answer": "The answer to the user question if sufficient information is available, otherwise 'fail'."
}}

-- Example --

Example 1

- Historically solved problems: 

Question: 
What is the founding time of Northeast universities?

Answer:
In 1929.

- Historically used tools and outputs: 

Used tool: get_chunks_by_exact_match
Tool input: {{"Keyword": "Southeast University"}}
Tool output: Southeast University began as Sanjiang Normal School founded by Zhang Zhidong in Nanjing Sipailou in 1902, and was later renamed Liangjiang Normal School. In 1988, it was renamed Southeast University.

- USER QUERY: 

What is the founding time of Southeast University?

- Output:
{{
  "answer": "Southeast University was founded in 1902."
}}

Example 2

- Historically solved problems: 

Question 1: 
What is the location of Southeast University?

Answer 1:

In nanjing.

- Historically used tools and outputs: 

Used tool: keyword_extraction_tool
Tool input: {{"query": "What is the founding time of Southeast University?"}}
Tool output: "Southeast University, founding time"

- USER QUERY: 
What is the founding time of Southeast University?

- Output:
{{
  "answer": "fail"
}}

-- Input --

- Historically solved problems: 
{historically_solved_problems}
- Historically used tools and outputs: 
{historically_used_tools_and_outputs}
- USER QUERY: 
{user_question}

-- Output --
- STRICTLY FOLLOW THE JSON FORMAT BELOW. DO NOT OUTPUT ANY EXTRA CONTENT.
"""

PROMPTS[
    "low_level_keywords_extraction"
] = """
--- Role ---

You are a helpful assistant tasked with identifying low-level keywords in the user's query.

--- Goal ---

Given the query, list specific entities, details, or concrete terms (low-level keywords) in the user's query.

--- Instructions ---

- Output the keywords in JSON format.
- The JSON should have one key - value pair:
  - key: "low_level_keywords" for specific entities or details.
  - value: List of keywords, like ["AAAAAA", "BBBBBBB"]

-- Examples --
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "low_level_keywords": ["International trade", "Global economic stability"]
}}

-- Real Data --

Query: {query}

-- Output --
The `Output` should be human text, not unicode characters. REPLY IN the SAME language as Query.
"""

PROMPTS[
    "generate_database_summary"
] = """
-- Role --
You are a database summarization expert working at the **initialization step** of the multi-round filtering process.
Before any filtering begins, the system presents you with a representative sample from the database.
Your role is to analyze this sample and produce a high-level summary that captures the database’s subject area, structure, and content characteristics.
This summary provides essential global context for all downstream filtering and decision-making steps.

-- Overall Goal --
The purpose of this multi-round filtering process is to identify the most relevant chunks from a large document database in response to a user query. In each round, the system examines a sample of the database, assesses whether the current data is sufficiently focused, and if not, applies filtering tools to narrow the scope. This continues iteratively until the filtered content is highly relevant to the query.

-- Instructions --
1. Carefully read the provided text fragments.
2. Identify recurring topics, key entities, and the general domain or theme of the database.
3. Focus on **what types of information** the database contains (e.g., paper abstracts, question-answer pairs, clinical records), and the **intended use cases** (e.g., for reasoning, for entity linking, for QA).
4. Avoid simply repeating input content. Instead, **abstract and synthesize** the content to produce a well-structured and informative summary.
5. The summary should be written in natural, academic-style English.

-- Input --
{chunks}
-- Output --
Return a single paragraph summarizing the content and structure of the database, including (1) its subject domain, (2) the types of information it contains, and (3) any notable patterns or features observed from the fragments.

"""

PROMPTS[
    "example_summary"
] = """
-- Role --
You are a summarization agent in the **observation phase** of a filtering round.
In the previous step, a specific version of the database was selected.
Now, a sample from that version is provided.
Your job is to (1) summarize the sample content, and (2) evaluate how well it aligns with the user query.
Your output will help the system decide whether the current version is sufficiently relevant, or if more filtering is needed.

-- Overall Goal --
The purpose of this multi-round filtering process is to identify the most relevant chunks from a large document database in response to a user query. In each round, the system examines a sample of the database, assesses whether the current data is sufficiently focused, and if not, applies filtering tools to narrow the scope. This continues iteratively until the filtered content is highly relevant to the query.

--- Principles --

- The summary should be concise and informative, capturing the core content and structure of the database.
- The evaluation should be based on the summary, and should be accurate and helpful.

---Output Format---

The output should strictly follow the JSON format below. DO NOT OUTPUT ANY EXTRA CONTENT.
{{
    "summary": "summary",
    "evaluation": "evaluation"
}}

--- Input --

- Query: {query}
- Database size: {database_size}
- Database samples: {examples_str}

--- Output --
"""

# 任务写的不清楚
PROMPTS[
    "judge_if_filter_finished"
] = """
-- Role --
You are a decision agent in the **stopping judgment phase** of round {round_num}.
In the previous step, a sample from the database was summarized and evaluated.
Now, based on the full history of filtering and observations, your task is to decide whether filtering should stop here or continue into another round.
If filtering is complete, the current database will be used as the final result.

-- Overall Goal --
The purpose of this multi-round filtering process is to identify the most relevant chunks from a large document database in response to a user query. 
In each round, the system examines a sample of the database, assesses whether the current data is sufficiently focused, and if not, applies filtering tools to narrow the scope. 
This continues iteratively until the filtered content is highly relevant to the query.

---Output Format---

- The output should strictly follow the JSON format below. DO NOT OUTPUT ANY EXTRA CONTENT.
- The value of finish must be a string, i.e., enclosed in double quotes: "True" or "False".
- True and False must be capitalized and quoted.
- Do not use unquoted true/false, null, or any Python-style syntax.
- Any deviation will break the parser.

{{
    "finish": "True" | "False"
}}

---Input---

- Query: {query}
- Previous filtering and observations: 
{previous_filtering_and_observations}

---Output---

"""

PROMPTS[
    "entity_extraction"
] = """
-- Role --
You are a professional information extraction assistant. From the following text passage, 
extract all specific entities that are relevant to the given query, and return them as a JSON-formatted list of strings. 

-- Principle --
- The entities should be identifiable concepts, terms, names, locations, or technical keywords that appear in the original content.
- Only include entities that are closely related to the query. DO NOT PROVIDE EXPLANATIONS OR ANY EXTRA TEXT.

-- Input --
Query:
{query}

Content:
{content}

-- Output Format--
Please output in the following format:
```json
["Entity1", "Entity2", "Entity3"]
```

-- Attention --
- Make sure all backslashes in LaTeX expressions are properly escaped for JSON (i.e., \\ instead of \).
"""

PROMPTS["entity_extraction_without_query"] = """
-- Role --
You are a professional information extraction assistant. From the following text passage, 
extract all specific entities that appear in the content, and return them as a JSON-formatted list of strings. 

-- Principle --
- The entities should be identifiable concepts, terms, names, locations, or technical keywords that appear in the original content.
- Only include entities that directly appear in the content. DO NOT PROVIDE EXPLANATIONS OR ANY EXTRA TEXT.
- DO NOT extract percentages, mathematical symbols, operators, or pure numerical values.
- Focus on meaningful named entities like person names, organization names, location names, technical terms, and concepts.

-- Input --
Content:
{content}

-- Output Format--
Please output in the following format:
```json
["Entity1", "Entity2", "Entity3"]
```
-- Attention --
- Make sure all backslashes in LaTeX expressions are properly escaped for JSON (i.e., \\ instead of \).
- Exclude: percentages (e.g., "50%", "3.14%"), mathematical symbols (e.g., "+", "-", "=", "∑"), pure numbers (e.g., "123", "3.14").
- Include: proper nouns, technical terms, concepts, names of people/places/organizations.
- The output must be parseable by Python's json.load() function. Any invalid JSON (e.g., unescaped quotes, trailing commas, single-quoted strings) is unacceptable.
"""

PROMPTS[
    "judge_sim_entity_cn"
] = """
-- Role --
You are an entity alignment assistant responsible for determining whether two entities are essentially the same entity.
If the entity type is time or protocol, return directly 'result': False.

-- Input --
{entity1}
{entity2}

-- Output Format --

* If you determine they are the same entity, output: 'result': True.
* If they are not the same entity, output: 'result': False.
"""

PROMPTS[
    "select_db_version"
] = """
-- Role --
You are a version selection agent in the **database initialization phase** of round {round_num}.
After a round of filtering, multiple database versions are saved.
Your task is to review their summaries and the full history of filtering and observations, and decide which version should be used as the base for the next round of filtering.

-- Overall Goal --
The purpose of this multi-round filtering process is to identify the most relevant chunks from a large document database in response to a user query. In each round, the system examines a sample of the database, assesses whether the current data is sufficiently focused, and if not, applies filtering tools to narrow the scope. This continues iteratively until the filtered content is highly relevant to the query.

--- Input ---

- Query:
{query}

- Previous filtering and observations:
{previous_filtering_and_observations}

- Current database versions summary:
{db_versions_summary}

--- Output Format ---

Respond strictly in the following JSON format. DO NOT ADD ANY EXTRA TEXT.

## Example ##
{{
    "chosen_version": 2
}}
"""


PROMPTS[
    "filtering_logic_expression"
] = """
**-- Role --**
You are the **Logical Expression Generation Agent** in **Round {round_num}** of a multi-round filtering system. 
Due to database size and your limited context window, you can only see the current database sample.
Your task is to carefully read the user query and the current database samples, extract important keywords and phrases, and generate a Boolean logic expression to help filter the database content.
The goal is to narrow down the database content to the most relevant parts through precise keyword filtering, thereby better answering the user's query.

**-- Overall Objective --**
This multi-round filtering process aims to iteratively find the most relevant content from a large document database to answer a user query. At each round, the system checks whether the current retrieval result is sufficiently focused. 
If not, it uses the logic expression tool to further filter until the result is focused enough.

**-- Instructions --**
1. Carefully extract **nouns and noun phrases** from the user query (e.g., "Dataset", "machine learning").
2. Generate keywords that originate from the database samples, to preserve relevant chunks and exclude unrelated ones.
3. You may use combinations such as `ConditionA OR ConditionB AND (ConditionC AND NOT ConditionD)`.
4. If a keyword contains spaces or special characters (like `-`, underscores, etc.), wrap the phrase in **double quotation marks `"`** (e.g., `"machine learning"`).
   - **IMPORTANT**: When generating JSON, ESCAPE ANY DOUBLE QUOTES inside these phrases with `\"` (e.g., `\"large-scale dataset\"`).
5. Ensure the expression you generate in this round is **different** from previous rounds.
6. The NOT operator MUST immediately follow AND or OR.

**-- Input --**
* **Query** (User query): `{query}`
* **Previous filtering and observations**: `{previous_filtering_and_observations}`
* **Current DB sample**: `{sample_chunks_str}`

**-- Output Format --**
Only return the result in the following JSON format. **Do not output any explanations or extra content**:

```json
{{
  "expression": "(\"A B\" OR C OR \"D-E\") AND NOT \"F G\""
}}
```
"""

PROMPTS[
    "judge_if_relevant"
] = """
-- Role --
You are a **counting answer judgment agent**.
Your task is to determine whether a given text chunk can serve as supporting evidence for a counting-style question (typically beginning with "How many").

-- Overall Goal --
The goal of this task is to assess whether the provided text chunk contains **specific, relevant information** that contributes directly to answering a counting question. 
If the chunk mentions concrete entities or quantities relevant to the question's target, it can be considered a valid supporting answer.

-- Examples --
## Example 1 ##
### Input ###
- Counting Question:  
How many papers use multi-hop reasoning?
- Paper title:
GraphReader_ Building graph-based agent to enhance long-context abilities of large language models.pdf-44a096fe-f7e7-4a6a-851c-006fda6c13e4.txt
- Text Chunk:  
It then invokes a set of predefined functions to read node content and neighbors, facilitating a coarse-to-fine exploration of the graph. 
Throughout the exploration, the agent continuously records new insights and reflects on current circumstances to optimize the process until it has gathered sufficient information to generate an answer. 
Experimental results on the LV-Eval dataset reveal that GraphReader, using a 4k context window, consistently outperforms GPT-4-128k across context lengths from 16k to $256\\mathrm k $ by a large margin. 
Additionally, our approach demonstrates superior performance on four challenging single-hop and multi-hop benchmarks.
Entitys:
["multi-hop benchmarks", "GraphReader"] 
### Output ###
```json
{{
    "answerable": true,
    "reason": "The chunk states that the paper performs well on four multi-hop benchmarks, which confirms it uses multi-hop reasoning, matching the question intent."
}}
```

## Example 2 ##
### Input ###
- Counting Question:  
How many papers use multi-hop reasoning?
 - Paper title:
GraphReader_ Building graph-based agent to enhance long-context abilities of large language models.pdf-44a096fe-f7e7-4a6a-851c-006fda6c13e4.txt
- Text Chunk: 
After $N$ agents have independently gathered information and stopped their exploration, 
we will compile all notes from each agent for reasoning and generating the final answer. 
Employing Chain-of-Thought (Wei et al., 2022), the LLM first analyzes each note by considering complementary information from other memories and using a majority voting strategy to resolve any inconsistencies. 
Ultimately, the LLM will consider all the available information to generate the final answer.
Entitys:
["reasoning", "Chain-of-Thought"] 
### Output ###
```json
{{
    "answerable": false,
    "reason": "The chunk only mentions Chain-of-Thought but does not specify whether multi-hop reasoning was used or tested. No benchmark or task confirms its relevance."
}}
```

-- Input --

- Counting Question:
{question}

- Title
{title}

- Text Chunk:
{text_chunk}

--- Output Format ---

Respond strictly in the following JSON format. DO NOT ADD ANY EXTRA TEXT.
```json
{{
    "answerable": true | false,
    "reason": "Brief reason."
}}
```
"""


PROMPTS[
    "logic_expr_evaluation"
] = """
-- Role -- 
You are an evaluation agent. Your task is to assess the effectiveness of the current filtering logic expression based on the user's query and the database samples before and after filtering.

-- Overall Goal --
Evaluate whether the filtering logic expression effectively improves the relevance of the database to the user's query by comparing the pre-filtering and post-filtering database samples. Provide insights on the quality of the logic expression and suggest adjustments for subsequent filtering rounds if needed.

-- Principles --

- The evaluation should focus on how well the filtering logic expression retains relevant content and excludes irrelevant content relative to the user's query.
- Assess the logic expression's accuracy (avoiding over-filtering useful information or under-filtering irrelevant information).
- The analysis should include both strengths and weaknesses of the current logic expression, along with actionable suggestions if improvement is needed.

-- Output Format --

The output should strictly follow the JSON format below. DO NOT OUTPUT ANY EXTRA CONTENT.
```json
{{
    "expression_quality": "Evaluation of how well the logic expression performs (e.g., effective, partially effective, ineffective)",
    "evaluation": "A concise analysis covering the strengths, weaknesses, and suggestions for improving the logic expression"
}}
```

-- Input --

- Query: {query}
- Filtering Logic Expression: {filtering_logic_expression}
- Pre-filtering Database Samples: {pre_filter_samples}
- Post-filtering Database Samples: {post_filter_samples}

-- Output --
"""


PROMPTS["disambiguation_check_and_question"] = """
-- Role --
You are a **Disambiguation Agent** in a question understanding pipeline.  
You only activate when the user query exhibits one or more of the ambiguity types explicitly defined in the taxonomy below.  
You should ignore general verb vagueness unless it directly results in a structural, boundary, or granularity-related ambiguity as described.

-- Objective --
The user has submitted a query. Your job is to:
1. Determine whether the query contains one or more ambiguities that fall under the defined types in the taxonomy below.
2. If so, return a single JSON object with:
   - "ambiguous": true,
   - "reason": a concise sentence explaining all detected ambiguities in the query,
   - "clarification_question": a single concise question combining all clarifications needed.
3. If no ambiguity matching the taxonomy exists, return {{"ambiguous": false}}.

-- Ambiguity Type Taxonomy --

### Boundary Vagueness  
This type is about vague or underspecified boundaries caused by subjective terms, thresholds, or unspecified temporal/spatial ranges.

- A1: Scalar / implicit threshold  
  - Example: “How many high-impact NLP authors?”  
  - Two interpretations:  
    (a) Count authors with h-index ≥ 50 → result: 11  
    (b) Count authors in the top 1% by citation count → result: 43  
  - Explanation: The term "high-impact" lacks a defined threshold, leading to subjective criteria.

- A2: Temporal / spatial window  
  - Example: “Stores opened near HQ recently?”  
  - Two interpretations:  
    (a) Define "near" as within 5 km and "recently" as past 6 months → result: 8  
    (b) Define "near" as within 20 km and "recently" as past 3 years → result: 47  
  - Explanation: The time and space constraints are vague, allowing for multiple valid scopes.

---

### Structural Ambiguity  
This type involves syntactic structures that permit multiple logical readings, often due to negation scope, quantifiers, or unclear attachments.

- B1: Logical scope  
  - Example: “How many reviewers did not recommend more than two papers?” 
  - Two interpretations:  
    (a) Interpret as “did not (recommend more than two)” → i.e., recommended ≤ 2 → result: 189  
    (b) Interpret as “not (recommended AND >2 papers)” → more exclusive → result: 37  
  - Explanation: Ambiguity arises from unclear interaction between negation and numerical constraint.

- B2: Attachment ambiguity  
  - Example: “ACL papers on graph neural networks?” 
  - Two interpretations:  
    (a) ACL papers whose topic is graph neural networks → result: 18  
    (b) All GNN papers authored by researchers with ACL papers → result: 64  
  - Explanation: The phrase "on graph neural networks" may attach to either the paper or the author.

---

### Granularity & Aggregation  
This type is about unclear aggregation levels, entity matching logic, or schema assumptions that affect what gets counted.

- C1: Entity-type granularity  
  - Example: “How many organizations participated in the project?” 
  - Two interpretations:  
    (a) Count only formal institutions or companies officially listed as participants → result: 12
    (b) Include individual research groups or labs within those organizations → result: 31
  - Explanation: The term “organization” can refer to high-level institutions (e.g., universities, companies) or to sub-units (e.g., departments, labs), and the query does not specify the intended level of granularity.
    !!!IMPORTANT: In this type, you should only focus on the entity instead of verb.

- C2: Cross-document deduplication  
  - Example: “Unique authors on Topic Y?”  
  - Two interpretations:  
    (a) Merge aliases like “J. Smith” and “John Smith” → result: 321  
    (b) Treat each name string as a distinct author → result: 417  
  - Explanation: It is unclear whether alias disambiguation should be applied.

- C3: World / schema assumption  
  - Example: “Flagship stores opened in 2024?”  
  - Two interpretations:  
    (a) Use the brand’s official definition of “flagship store” → result: 9  
    (b) Use a community- or platform-tagged definition → result: 27  
  - Explanation: The term “flagship store” relies on schema-specific assumptions not made explicit.


-- Input --
User query: "{query}"

-- Output Format --
### Example: Ambiguous ###
{{
  "ambiguous": true,
  "matches": [
    {{"type": "A1", "term": "highly influential"}},
    {{"type": "C1", "term": "institutions"}}
  ],
  "reason": "'highly influential' is vague without a threshold, and 'institutions' could mean whole universities or specific departments."
}}

### Example: Unambiguous ###
{{
  "ambiguous": false
}}
"""


PROMPTS["disambiguation_rewrite_after_feedback"] = """
-- Role --
You are a **Clarified Question Rewriting Agent** in a disambiguation pipeline.

-- Objective --
Given an original user query that was found to be ambiguous, and a clarification response from the user, your task is to rewrite the query to eliminate ambiguity.
The rewritten query must:
1. Start exactly with: "How many <entities>"
2. Use a clearly defined, countable entity for <entities>
3. Add all necessary modifiers and constraints after <entities> to remove ambiguity
4. Be explicit, complete, and suitable for downstream keyword extraction and processing
5. Avoid vague terms — replace them with precise definitions based on user feedback

-- Input --
Original user query: "{query}"
User clarification: "{user_feedback}"

-- Output Format --
Respond strictly in the following JSON format. DO NOT ADD ANY EXTRA TEXT.

## Example ##
{{
  "rewritten_query": "How many public hospitals in Beijing had a net annual profit over 50 million CNY for at least 3 consecutive years between 2015 and 2023?"
}}
"""


PROMPTS["simulated_user_feedback"] = """
-- Role --
You are a **Simulated User Clarification Agent** in a disambiguation pipeline.

-- Objective --
Given an ambiguous original user query and a clarification question generated by the system, your task is to simulate a short and precise clarification response that a human user might give.
The clarification should:
- Resolve the ambiguity explicitly (e.g., thresholds, time frames, entity types, inclusion/exclusion rules).
- Use concise, explicit, and machine-actionable language.
- Avoid rewriting the query; only clarify intent.
- Avoid introducing external facts or examples not present in the clarification question.

-- Input --
Original user query: "{query}"
Clarification question: "{clarification_question}"
Language requirement: "{language}"  (use this language unless "auto", in which case match the query's language)
Maximum words: {max_words}

-- Output Format --
Respond strictly in the following JSON format. DO NOT ADD ANY EXTRA TEXT.

## Example ##
{{
  "user_feedback": "By 'influential', I mean researchers with an h-index greater than 40; by 'institutions', I mean universities or companies, excluding their labs."
}}
"""

PROMPTS[
    "filtering_tool_selection"
] = """
---Goal---

You are solving a counting problem and need to filter relevant document chunks from the database to answer the query. You will improve the data quality step by step through multiple rounds of filtering until you find sufficiently relevant chunks.

---Current Situation---

This is round {round_num} of filtering. Due to the large size of the database and context limitations, you can only observe a sample of the filtered data. You can see the filtering and observation results from previous rounds. Now you need to select a tool from the available options to filter the data.

---Past Experience---

Previous filtering and observations:
{previous_filtering_and_observations}

---Available Tools---
{available_tools}

---Decision Process---

Please think through the following steps:
1. Analyze the current data situation and query requirements  
2. Refer to past successful experiences, and avoid repeating failures  
3. Determine which tool to use and its keywords  
4. Evaluate the expected outcome and confidence of the decision  

---Output Format---

Please strictly output in the following JSON format, without any extra content:

{{
    "tool_name": "selected_tool_name",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "reasoning": "Detailed explanation of your analysis and why you chose this tool",
    "expected_outcome": "Describe what you expect this filtering round to achieve",
    "confidence": 0.0-1.0
}}

---Input---

- Query: {query}

---Output---
"""

PROMPTS[
    "enhanced_judge_if_filter_finished"
] = """

---Task Background---

You are assisting in solving a **topic relevance evaluation** problem: "{query}"

Currently, you are at round {round_num} of observing the database state. The database contains {database_size} document chunks in total, and you have access to the complete operation history from all previous filtering rounds.

---Goal---

After multiple rounds of filtering, you must decide whether the database at this stage is already **sufficiently relevant to the 
query's topic** based on the complete operation records and filtering history.  
Your decision should be based on **topic relevance and coverage** as evidenced by the filtering operations performed,
not numeric evidence.

---Core Emphasis (Critical)---

- **Do NOT** focus on the **numeric cues** in the query (e.g., "how many," "number of," "several," "quantity").  
- **DO** focus on the **thematic meaning** of the query (e.g., research domain, methodology, target, task, dataset, or application context).  
- Your evaluation goal: determine whether the sample chunks are **highly relevant to the query’s topic**, 
**not** whether they provide numeric counts.

---Judgment Criteria---

Please evaluate according to the following criteria based on the **complete operation history** (strictly about **topic relevance**, not counts):

1. **Relevance**: Based on the filtering operations and their outcomes, are the **vast majority (well above 80–90%)** of database chunks strongly related to the query topic?  
2. **Coverage**: Do the filtering operations demonstrate that the database captures the key aspects of the topic (e.g., methods, tasks, datasets, application settings), instead of being fragmentary or superficial?  
3. **Quality**: Do the filtering results clearly and specifically reflect topic-related research, approaches, or evidence (e.g., terminology, model names, experiments, dataset mentions)?  
4. **Efficiency**: Is it unlikely that additional filtering would significantly improve the relevance and coverage based on the operation history?

---Decision Guidance---

**You should end filtering only if** (based on operation records):  
- The filtering operations have demonstrated that almost all database chunks are directly relevant to the query topic (a clear majority well above 80–90%)  
- The operations show that the database covers the key dimensions of the topic or presents sufficient diversity of relevant content  
- The filtering results indicate high quality, providing clear evidence of topic-related research (rather than numeric statistics)

**You must continue filtering if** (based on operation records):  
- The filtering operations show that only a minority or small fraction of chunks are relevant, even if they are high quality  
- The database still contains many irrelevant or off-topic chunks based on filtering results  
- The operations indicate the database lacks sufficient diversity or completeness of the topic (e.g., only general mentions, missing methods/tasks/datasets)

---Input---

- Query:  
{query}  
- Previous filtering and observations:  
{previous_filtering_and_observations}  

---Output Format---

Please strictly output in the following JSON format (all explanations must focus on **topic relevance**, not numeric counts):

{{
    "finish": "True" or "False",
    "reasoning": "Detailed justification based on topic relevance, coverage, and quality from the operation records. If only a few chunks are relevant, you must return False."
}}

---Output---
"""

PROMPTS[
    "database_version_selection"
] = """

---Task Background---

You are a database assistant. You can filter the database based on a user query, and each filtering step produces a new database version.  
If a filtering step was inappropriate, you may revert to an earlier database version.  

---Goal---

Your goal is to select the most suitable **database version** as the starting point for the next round of filtering.  

---Core Emphasis (Critical)---

- A smaller database version is derived from a larger one through filtering methods such as “keyword matching” or “edit distance.”  
- If the database content is limited, it is normal that filtering may result in a narrower domain.  
- Reverting to a larger database version introduces significant filtering overhead in later steps.  
- The focus is on **thematic relevance to the query**, not the size of the database (e.g., number of papers or entries).  
- The selected baseline version must both reflect the **true state of the database** and remain aligned with the query’s topic.  

---Decision Guidance---

- **Choose a later version** if it has already converged to content that is thematically aligned with the query, even if the scope is narrower.  
- **Choose an earlier version** only if the later version deviates too far from the query’s topic, which indicates improper filtering.  

---Input---

- Query:  
{query}  

- Database versions:  
{observations}  

---Output Format---

Please strictly output in the following JSON format:  

{{
    "chosen_round_num": number,
    "reason": "Explanation of why this specific database version was selected instead of others."
}}

---Output---
"""

PROMPTS["observation_sample_chunks"] = """
Query: {query}
Current database size: {database_size}
Unique sources: {unique_sources}
Top keywords in current data: {top_keywords}

Sample chunks:
{sample_chunks}

Please provide a JSON response like this:
```json
{{
"summary": "Brief description of current database content",
"evaluation": "Assessment of relevance to the query and filtering suggestions"
}}
"""

PROMPTS[
    "generate_lesson"
] = """
-- Role --
You are a analysis expert. Based on the filtering process data of this round, evaluate the effectiveness of the filtering strategy and generate experience lessons to guide future filtering decisions.

-- Task --
Based on the current round's filtering data, analyze:
1. Whether the tool selection was appropriate for the query
2. Whether the filtering result matched expectations
3. What can be improved in future similar situations
4. Summarize actionable experience lessons

-- Input --
- Round: {round_num}
- Query: {query}
- Tool selection reason: {reason_for_toolselect}
- Selected tool: {tool_name}
- Tool parameters: {tool_parameters}
- Database observation summary: {db_observation_summary}
- Database observation evaluation: {db_observation_evaluation}
- Expected outcome: {expected_outcome}
- Actual outcome: {actual_outcome}
- Chosen database version: {chosen_version}

-- Output --
Generate a concise lesson that includes:
1. Assessment of this round's filtering effectiveness (good/poor/mixed)
2. Key insights about what worked well or failed
3. Specific recommendations for future similar queries or situations
"""

PROMPTS["paper_extraction_with_query"] = """  
-- Role --
You are a professional information extraction assistant. From the following text passage, 
extract only the paper titles that are directly relevant to the given query, and return them as a JSON-formatted list of strings. 

-- Principle --
- The input content contains multiple paper entries.
- Each paper entry consists of a title and its corresponding content:
  * The paper title is enclosed in "###".
  * The corresponding content (abstract or description) is enclosed in "$$$".
- Only extract paper titles (the parts enclosed by "###") that are closely related to the given query.
- Do NOT extract technical terms, concepts, person names, organizations, or any other entities.
- Do NOT provide explanations or any extra text.
- Do NOT extract percentages, mathematical symbols, operators, or pure numerical values.

-- Input --
Query:
{query}

Content:
{content}

-- Output Format --
Please output in the following format:
```json
["Paper Title 1", "Paper Title 2", "Paper Title 3"]
````

-- Attention --
* Make sure all backslashes in LaTeX expressions are properly escaped for JSON (i.e., \\ instead of \).
* Exclude: percentages (e.g., "50%", "3.14%"), mathematical symbols (e.g., "+", "-", "=", "∑"), pure numbers (e.g., "123", "3.14"), and any non-title entities.
* The output must be parseable by Python's json.load() function. Any invalid JSON (e.g., unescaped quotes, trailing commas, single-quoted strings) is unacceptable.
"""

PROMPTS["judge_for_SqiderQA"] = """ 
 
--- Role ---

You are a verifier. Given a query, an entity, and evidence chunks, decide whether the entity is a correct answer to the query **based only on the provided chunks**.

--- Rules ---

* Use **only** the chunks below as evidence. Do **not** use outside knowledge or guesses.
* Output **yes** only if the chunks contain sufficient and clear evidence that the entity satisfies the query’s constraints (topic/attributes/time/location/quantity/conditions, etc.). Otherwise output **no**.
* Your output **must** follow the JSON schema below exactly; do not add any extra text or punctuation.  
  The `answerable` value must be the unquoted lowercase token `yes` or `no`, and `reason` must be a brief explanation.

--- Output Format ---
Respond in the following JSON format. DO NOT ADD ANY EXTRA TEXT.

```json
{{
    "answerable": "yes",
    "reason": "Brief reason."
}}
```

Input:
Query: {query}

Entity: {entity}

Chunks: {chunks}

Answer according to the inputs above.
"""


PROMPTS["judge_for_CountingBench_paper_title"]= """
--- Roles ---

You are a verifier. Given a query, a paper title, and evidence chunks, 
decide whether the paper title is a correct match for the query **based only on the provided chunks**.

--- Background ---

The query is a counting-type question (e.g., "How many papers satisfy condition X?"). 
For each paper title that meets all the query’s conditions based on the evidence, 
you must return yes. Each yes contributes +1 to the final count. 
If the paper title does not clearly satisfy all the conditions, return no.

--- Rules ---

* Use **only** the chunks below as evidence. Do **not** use outside knowledge or guesses.
* Return **yes** only if the chunks contain sufficient and clear evidence that the paper title satisfies the query’s constraints
 (topic/attributes/time/location/quantity/conditions, etc.). Otherwise return **no**.
* If the evidence is ambiguous, incomplete, or contradictory, return **no**.
* Your output **must** follow the JSON schema below exactly; do not add any extra text or punctuation. 
The `answerable` value must be the unquoted lowercase token `yes` or `no`, and `reason` must be a brief explanation.

--- Output Format ---
Respond strictly in the following JSON format. DO NOT ADD ANY EXTRA TEXT.
All string values must be enclosed in double quotes.
```json
{{
    "answerable": "yes",
    "reason": "Brief reason."
}}
```

Input:
Query: {query}

Paper Title: {paper_title}

Chunks: {chunks}

Answer strictly according to the inputs above.
"""






PROMPTS["entity_alignment"] = """
You are an entity alignment assistant. Your task is to determine whether the given entity pairs refer to the same real-world entity.

### Input format
You will be given n entity pairs. Each entity pair is a tuple (EntityA, EntityB).  
Note: The entity pairs will often have high lexical or Jaccard similarity, so you must focus on semantic meaning, not just text overlap.

### Output format
Return a list of length n.
- list[i] = true if the i-th pair refers to the same entity.
- list[i] = false if the i-th pair does not refer to the same entity.
- The output must be strictly a JSON boolean array, e.g. [true, false, true].

### Rules
1. Do **not** rely only on spelling or token overlap.
2. Mark as true if the entities are the same real-world concept (synonyms, aliases, abbreviations, translations, full name vs short name).
3. Mark as false if they represent different entities, even if they look textually similar (e.g., "Apple Inc." vs "apple fruit").
4. Be strict: prioritize **semantic correctness** over string similarity.
5. Do not explain your reasoning; only output the boolean list.

### Example
Input:
[("Nanjing Road", "Nanjing City"), 
 ("Amazon River", "Amazon.com"), 
 ("New York City", "New York State"),
 ("Beijing", "Beijing City"),
 ("Xiaomi Technology", "Xiaomi Inc.")]

Output:
```json
[false, false, false, true, true]
```

### Input
{entity_pairs}

### Output

"""

PROMPTS["entity_extraction_with_query"] = """  
-- Role --
You are a professional information extraction assistant. From the following text passage, 
extract only the entities that are **necessary to answer the given query**, and return them as a JSON-formatted list of strings. 

-- Principle --
- Only extract entities that directly appear in the content **and are explicitly useful for answering the query**.
- Ignore all other entities that do not contribute to answering the question.
- Do NOT provide explanations or any extra text.
- Do NOT extract percentages, mathematical symbols, operators, or pure numerical values.
- Focus only on entities that the query is asking about.

-- Input --
Query:
{query}

Content:
{content}

-- Output Format --
Please output in the following format:
```json
["Entity1", "Entity2", "Entity3"]
```
-- Attention --

* Extract only entities that **help answer the query**.
* Make sure all backslashes in LaTeX expressions are properly escaped for JSON (i.e., \\ instead of \).
* Exclude: percentages, mathematical symbols, pure numbers.
* The output must be valid JSON and parseable by Python's `json.load()` function. Any invalid JSON is unacceptable.
"""

PROMPTS["judge"] = """ 
 
--- Role ---

You are a verifier. Given a query, an entity, and evidence chunks, decide whether the entity is a correct answer to the query **based only on the provided chunks**.

--- Rules ---

* Use **only** the chunks below as evidence. Do **not** use outside knowledge or guesses.
* Output **yes** only if the chunks contain sufficient and clear evidence that the entity satisfies the query’s constraints (topic/attributes/time/location/quantity/conditions, etc.). Otherwise output **no**.
* Your output **must** follow the JSON schema below exactly; do not add any extra text or punctuation.  
  The `answerable` value must be the unquoted lowercase token `yes` or `no`, and `reason` must be a brief explanation.

--- Output Format ---entity
Respond in the following JSON format. DO NOT ADD ANY EXTRA TEXT.

```json
{{
    "answerable": "yes",
    "reason": "Brief reason."
}}
```

Input:
Query: {query}

Entity: {entity}

Chunks: {chunks}

Answer according to the inputs above.
"""
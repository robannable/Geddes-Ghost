# GeddesGhost: A Digital Conversation with Patrick Geddes

## What is GeddesGhost?

GeddesGhost is a computer program that lets you have conversations with an AI version of Patrick Geddes, a Scottish thinker who lived from 1854 to 1932. Geddes was known for his work in urban planning, sociology, and environmental studies. The program uses modern AI technology to try to understand and respond to questions in a way that Geddes might have, based on his writings and ideas.

## How Does It Work?

The program has two main parts:

1. **The Chat Interface**: This is where you talk with the AI version of Geddes. You type your questions or thoughts, and the program responds. It's designed to feel like a natural conversation, while trying to stay true to Geddes' way of thinking and speaking.

2. **The Admin Dashboard**: This is a tool for people running the program to see how it's being used, what kinds of questions people are asking, and how well the AI is responding.

## The Main Features

### Talking with Geddes

When you use the chat interface, the program:
- Takes your question and looks for relevant information in Geddes' writings
- Uses AI to understand the context and generate a response
- Formats the response to be clear and readable
- Keeps track of your conversation history

The program can use different AI models (currently Anthropic and Ollama) to generate responses. This flexibility allows for different styles of interaction and helps ensure the system keeps working even if one service has issues. Each model declares which generation settings it accepts, so the controls shown in the sidebar match what the chosen model can actually do.

### Understanding the Responses

The program tries to make Geddes' responses feel authentic by:
- Using his actual writings as a basis for responses
- Maintaining his characteristic way of speaking and thinking
- Connecting different ideas in the way he might have
- Keeping track of the conversation context

### Behind the Scenes

The program keeps detailed records of conversations in two ways:
1. CSV files (spreadsheets) that track the basic information
2. JSON files that store more detailed data about the interactions

This information helps us understand:
- What topics people are most interested in
- How well the AI is responding
- Which parts of Geddes' writings are most relevant
- How people are using the system

## Prompting and Analysis Pipeline

This section explains how a user message becomes a Geddes-like response, and what analytics are produced along the way.

### 1) Retrieval Augmented Generation (RAG)
- The system uses the texts in `documents/` as the knowledge base.
- Each document is split into chunks and embedded (vectorized). When a user asks a question, the system:
  - Computes an embedding for the question
  - Finds the most similar chunks by cosine similarity
  - Formats the top matches as human-readable citations like: `filename (score: 0.8421)`
- Up to three top chunks are included in the prompt context. These are also written to the logs as `chunk1_score`, `chunk2_score`, `chunk3_score` for analysis.

### 2) Prompt Composition
- The prompt has a stable “system” component and a dynamic “context” component:
  - System prompt: Encodes Geddes’ voice, values, and constraints (see `prompts/patrick_geddes_prompt.txt`). It steers tone, perspective, and ethical boundaries.
  - Context window: The user’s message plus the retrieved chunks. The chunks are presented with brief source hints so the model can ground the answer.
- The final message sent to the model is: system prompt → retrieved context → user message.

### 3) Cognitive Modes
To shape the kind of answer produced, we encourage three complementary modes:
- Survey: Map the terrain; outline perspectives, key terms, and sources.
- Synthesis: Connect ideas across domains; show relationships and patterns.
- Proposition: Offer concrete, testable recommendations or next steps.

The system records a best-effort estimation of mode usage per response in `cognitive_mode` (a small dict, e.g., `{"survey": 1, "synthesis": 2, "proposition": 1}`). The admin dashboard aggregates this to show how responses skew over time.

### 4) Creative Markers
We track lightweight linguistic markers that often correlate with creative/insightful responses, such as:
- Metaphor language
- Ecological references
- Speculative phrasing
- Cross-disciplinary linking

For each response, a small dictionary is logged in `creative_markers` counting occurrences per marker. The dashboard plots aggregate frequencies.

### 5) Response Depth, Model Provider, and Model Name
Temperature used to be the only depth control. Anthropic removed the sampling
parameters (`temperature`, `top_p`, `top_k`) from Opus 4.7 onwards - sending any
of them to a current model returns a 400 - and replaced them with
`output_config.effort`. Ollama and older Claude models still take temperature.

The system therefore records a provider-neutral **depth band** alongside
whichever control was actually sent:

- `depth`: `focused`, `balanced` or `expansive`. Always present.
- `actual_temperature`: the temperature sent, where the model accepts one.
- `effort`: the effort level sent, where the model accepts one.
- `temperature_source`: whether the control came from the cognitive mode (auto)
  or the sidebar (manual).
- `model_provider` and `model_name`: e.g. Anthropic / Claude variant, or Ollama
  / local model. The dashboard shows usage distribution across models.

Cognitive modes map onto depth as follows:

| Mode | Depth | Temperature | Effort |
|---|---|---|---|
| Survey | focused | 0.7 | `medium` |
| Synthesis | balanced | 0.8 | `high` |
| Proposition | expansive | 0.9 | `xhigh` |

On a model that accepts no generation controls, the depth band still shapes the
system prompt, so the pedagogy survives the loss of the parameter.

### 5a) Token Usage
Each response records what it cost: `input_tokens`, `output_tokens` and
`cache_read_input_tokens` where the provider reports them, plus `stop_reason`
so truncated and declined responses are visible in the logs rather than
appearing as unusually short answers.

### 6) Logged Fields (CSV)
Typical CSV columns written to `logs/` include:
- `date`, `time`: When the response was generated
- `name`: User identifier (if captured)
- `question`: User input
- `response`: Model output (or a reference to it)
- `response_length`: Word count of the response
- `chunk1_score`, `chunk2_score`, `chunk3_score`: Retrieved context chunks with similarity scores
- `cognitive_mode`: JSON-like dict of mode counts
- `creative_markers`: JSON-like dict of marker counts
- `temperature`: JSON-like dict of effectiveness keyed by the control in play
- `actual_temperature`, `effort`, `depth`, `temperature_source`: Generation settings
- `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `stop_reason`: Usage
- `model_provider`, `model_name`: Which model produced the response

Note: Older logs may not include all columns; the dashboard fills missing columns with empty values where possible to keep analytics robust.

## The Admin Dashboard

The admin dashboard is like a control center that shows:
- How many people are using the system
- What kinds of questions they're asking
- How well the AI is responding
- Which parts of Geddes' writings are being referenced most often

This information helps us improve the system and understand how people are engaging with Geddes' ideas.

## Technical Details

The program is built using:
- Python as the main programming language
- Streamlit for the user interface
- Various AI services for generating responses
- A system for storing and retrieving information from Geddes' writings

## Future Plans

We're working on several improvements:
1. Making the responses more accurate and authentic
2. Streaming responses as they are generated, rather than waiting for the whole reply
3. Improving how the program finds relevant information
4. Making the interface more user-friendly
5. Adding features like conversation export and different language support

## Why This Matters

This project is inspired by the idea of "friendship with the ancients" - the concept that we can learn from historical figures by engaging with their ideas in a personal way. By creating a system that lets people have conversations with an AI version of Geddes, we hope to make his ideas more accessible and engaging for modern audiences.

The program isn't perfect - it's an approximation of how Geddes might think and respond. But it provides a way for people to explore his ideas in an interactive format, potentially leading to new insights and understanding of his work.

## Getting Involved

If you're interested in helping improve the program, you can:
- Report any issues you find
- Suggest improvements
- Help test new features
- Contribute code if you're a programmer

The project is open source, meaning anyone can look at the code and suggest changes. We welcome contributions that help make the system better at representing Geddes' ideas and making them accessible to more people. 
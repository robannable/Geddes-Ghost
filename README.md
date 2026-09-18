# GeddesGhost (formerly PGAAS)

An interactive AI chatbot that simulates conversations with Patrick Geddes (1854-1932), the Scottish biologist, sociologist, geographer, and pioneering town planner. Using advanced language models and Retrieval Augmented Generation (RAG), it allows users to engage with Geddes' interdisciplinary approach to urban planning and social reform.

## Features

- Interactive chat interface with Patrick Geddes
- Support for multiple AI model providers (Anthropic, OpenRouter, Ollama)
- Document-based context retrieval
- Admin dashboard for analytics and monitoring
- Conversation history tracking
- Response quality metrics
- User interaction analysis

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/geddesghost.git
   cd geddesghost
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your API keys:
   ```env
   ANTHROPIC_API_KEY=your_anthropic_key_here
   OPENROUTER_API_KEY=your_openrouter_key_here  # optional
   ADMIN_PASSWORD=your_admin_password_here
   ```

5. Run the main application:
   ```bash
   streamlit run geddesghost.py
   ```

6. Access the admin dashboard:
   ```bash
   streamlit run admin_dashboard.py
   ```

## Project Structure

- `geddesghost.py` - Main application file
- `admin_dashboard.py` - Analytics and monitoring dashboard
- `logs/` - Directory for conversation logs
- `documents/` - Source documents for context retrieval
- `prompts/` - System prompts and templates
- `requirements.txt` - Python dependencies

## Model Configuration

Model provider and parameters are configured in `geddesghost.py` (see `MODEL_CONFIG`). You can switch between Anthropic, OpenRouter and Ollama by changing the `current_provider` field, or from the sidebar at runtime.

### Providers

| Provider | Endpoint | Key | Model list |
|---|---|---|---|
| Anthropic | `/v1/messages` | `ANTHROPIC_API_KEY` | `GET /v1/models`, falling back to the registry in this file |
| OpenRouter | `/api/v1/chat/completions` (OpenAI-compatible) | `OPENROUTER_API_KEY` | `GET /api/v1/models`, public - no key needed to browse |
| Ollama | `/api/generate` | none | `GET /api/tags` on the local server |

OpenRouter puts several hundred models behind one key, so the sidebar has a filter box, and each option shows its input price per million tokens. Its listing reports a `supported_parameters` array per model, which is where that model's capabilities come from - no list is maintained here for OpenRouter.

### Generation controls

Anthropic removed the sampling parameters (`temperature`, `top_p`, `top_k`) from Opus 4.7 onwards. Sending any of them to a current model returns a 400. Reasoning depth is now set with `output_config.effort` instead.

Rather than hard-code one knob, each model declares what it accepts - from `ANTHROPIC_MODELS` in this file for Anthropic, and from the provider's own listing for OpenRouter - and both the request payload and the sidebar are built from that declaration:

| Model | Control |
|---|---|
| `claude-opus-5`, `claude-sonnet-5`, `claude-opus-4-8` | `effort` (low / medium / high / xhigh / max) |
| `claude-sonnet-4-6` | `temperature` |
| `claude-haiku-4-5` | `temperature` |
| `claude-sonnet-4-20250514` (deprecated) | `temperature` |
| OpenRouter models listing `reasoning` | `reasoning.effort` |
| OpenRouter models listing `temperature` only | `temperature` |
| Ollama models | `temperature` and `top_p` |

Where a model accepts both, effort wins: it describes how much thinking an answer deserves, which is what the cognitive modes actually distinguish. Temperature is not then added from config behind it.

OpenRouter's listing says *whether* a model takes `reasoning`, not which effort values it accepts, and an unsupported value returns a 400. `OPENROUTER_EFFORT_LEVELS` therefore defaults to the universally accepted `low`/`medium`/`high`; widen it for a specific model with `OPENROUTER_EFFORT_OVERRIDES`. Requested levels above what a provider offers are clamped down rather than collapsed onto one fallback, so the ordering between cognitive modes survives.

Note that reasoning tokens come out of the same `max_tokens` budget as the answer - OpenRouter derives a reasoning budget of roughly 0.8 × `max_tokens` at high effort - so `max_tokens` is set to 16000 on the API providers to leave the reply room. It is a ceiling, not a target; response length is governed by the system prompt.

Models the registry does not recognise default to sending **no** generation parameters. Omitting a parameter is always valid; sending one the model rejects is not.

The Anthropic model list is fetched live from `GET /v1/models` when an API key is present, falling back to the built-in registry otherwise. Use the sidebar's **Refresh model list** button after a new model is released.

### Response depth

The cognitive modes map onto a provider-neutral depth band, and whichever control the selected model supports is derived from it:

| Cognitive mode | Depth | Temperature | Effort |
|---|---|---|---|
| Survey | focused | 0.7 | `medium` |
| Synthesis | balanced | 0.8 | `high` |
| Proposition | expansive | 0.9 | `xhigh` |

On a model that accepts no generation controls at all, the depth band still steers the system prompt.

## Tests

The model layer has no external dependencies at test time - it is exercised against stubs rather than the live API:

```bash
python test_model_layer.py
```

## Documentation

For detailed documentation about the project's features, architecture, and future improvements, see `PROJECT_DOCUMENTATION.md`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project is inspired by the concept of "friendship with the ancients" as explored in De Cruz's paper "Friendship with the ancients". This philosophical practice involves imaginative engagement with works of deceased authors, allowing us to envision them as friends and enter into a parasocial relationship with them.

Reference:
De Cruz, H. 'Friendship with the ancients', Journal of the American Philosophical Association.

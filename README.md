# GeddesGhost (formerly PGAAS)

An interactive AI chatbot that simulates conversations with Patrick Geddes (1854-1932), the Scottish biologist, sociologist, geographer, and pioneering town planner. Using advanced language models and Retrieval Augmented Generation (RAG), it allows users to engage with Geddes' interdisciplinary approach to urban planning and social reform.

## Features

- Interactive chat interface with Patrick Geddes
- Support for multiple AI model providers (Anthropic, Ollama)
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

Model provider and parameters are configured in `geddesghost.py` (see `MODEL_CONFIG`). You can switch between Anthropic and Ollama by changing the `current_provider` field.

## Documentation

For detailed documentation about the project's features, architecture, and future improvements, see `PROJECT_DOCUMENTATION.md`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project is inspired by the concept of "friendship with the ancients" as explored in De Cruz's paper "Friendship with the ancients". This philosophical practice involves imaginative engagement with works of deceased authors, allowing us to envision them as friends and enter into a parasocial relationship with them.

Reference:
De Cruz, H. 'Friendship with the ancients', Journal of the American Philosophical Association.

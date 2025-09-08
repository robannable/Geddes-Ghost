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

The program can use different AI models (currently Anthropic and Ollama) to generate responses. This flexibility allows for different styles of interaction and helps ensure the system keeps working even if one service has issues.

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
2. Adding support for more AI models
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
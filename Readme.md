# Intelligent Multi-Domain Support Chatbot

This is an intelligent, multi-agent chatbot designed to provide solutions to user issues across multiple domains. The chatbot is capable of understanding user queries, identifying the relevant field, and providing automated solutions. It also offers escalation options to human support based on user feedback.

## 🚀 Features

- **Domain-Aware Query Understanding**  
  Automatically detects the domain of the user issue:
  - Banking
  - E-commerce
  - Telecom
  - Healthcare

- **Fallback to User Feedback**  
  If the chatbot cannot determine the correct domain, it requests user feedback to select the appropriate domain.

- **Agent-Based Resolution System**  
  Once the domain is identified, relevant domain-specific agents are activated to:
  - Ask clarifying questions
  - Understand the issue in detail
  - Provide accurate and tailored solutions

- **Iterative Feedback Loop**  
  After each solution is provided:
  - The chatbot asks whether the issue is resolved.
  - If **Resolved**: A clear report is generated summarizing the issue and the solution.
  - If **Not Resolved**: Further questions are asked to better understand and address the issue.
  - If **Person Support Needed**: The chatbot assigns the best available support person.

- **Human Support Integration**  
  When a user requests human assistance:
  - The system selects the best support person based on:
    - Past resolved tickets
    - Skillset
    - Current workload (open tickets)
  - The user receives the support person's details, including email for direct contact.

## 🧠 Tech Stack

- **Python**
- **LangGraph**
- **RAG (Retrieval-Augmented Generation)**

## 🛠 How to Run the Project

To run the project locally:

```bash
LangGraph dev

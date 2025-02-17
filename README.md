# 🤖 4o-Operator | A ReAct based CL-CUA

4o-Operator is a fully autonomous Command Line Computer-Using Agent (CL-CUA) that leverages zero-shot ReAct principles [(ArXiv)](https://arxiv.org/abs/2210.03629) to perform tasks on your behalf. Leveraging OpenAI GPT model, it can execute shell commands, run Python code, perform web scraping, manage files, and make API calls, all within a Directed Acyclic Graph-like architecture. The agent intelligently combines outputs from various tools to achieve specified goals, offering a flexible and powerful solution for automating command-line tasks.

## ⭐ | Features

- **Shell Command Execution**: Run bash commands directly through the agent.
- **Python Code Execution**: Execute Python scripts in a REPL environment.
- **Web Scraping**: Gather data from websites autonomously.
- **File Management**: Handle file operations such as reading, writing, and organizing.
- **API Interaction**: Make API calls to interact with web services.
- **Tool Combination**: Integrate outputs from multiple tools to accomplish complex tasks.

## 🧪 | Getting Started

### Prerequisites

- **Docker**: Ensure Docker is installed on your system.

### Installation

1. **Pull the Docker Image**: Download the latest image from Docker Hub.

   ```bash
   docker pull thethinkmachine/4o-agent
   ```

2. **Run the Docker Container**: Start the container with appropriate settings.

   ```bash
   docker run -p 8000:8000 --name 4o-operator -e AIPROXY_TOKEN=$AIPROXY_TOKEN thethinkmachine/4o-agent
   ```

   *It's recommended to run the agent within a containerized environment to mitigate potential risks.*

### Environment Variables

4o-Operator supports several environment variables for configuration:

- `AIPROXY_TOKEN`: **AI Proxy token** *(for users utilizing the AI Proxy OpenAI endpoint, particularly IIT students)*. **You don't need this if you have your own OpenAI API key.**

- `USE_CUSTOM_API`: Set to `true` if using endpoints other than AI Proxy.
- `CUSTOM_BASE_URL`: Custom API base URL (defaults to `https://api.openai.com/v1/`).
- `CUSTOM_API_KEY`: Custom API key (your **OpenAI API key** if a custom base URL is not defined).
- `CHAT_MODEL`: Chat model name (defaults to `gpt-4o-mini`).
- `EMBEDDING_MODEL`: Embedding model name (defaults to `text-embedding-3-small`).

To run the agent with your OpenAI API key:

```bash
docker run -it -p 8000:8000 \
  -e USE_CUSTOM_API=True \
  -e CUSTOM_API_KEY=your_custom_api_key \      # Your OPENAI_API_KEY
  -e CHAT_MODEL=your_chat_model \              # (Optional) Select from any OpenAI model (uses `gpt-4o-mini` by default)
  --name 4o-operator thethinkmachine/4o-agent
```

Replace the placeholder values with your actual configuration.

## Usage

Once the container is running, you can interact with 4o-Operator through its command-line interface. Input your commands or tasks, and the agent will autonomously determine the best approach, utilizing its suite of tools in any combination to achieve the desired outcome.

## Caution

- **Security Risks**: Running a CUA carries inherent risks, including potential prompt injections.
- **Model Vulnerabilities**: While proprietary models like GPT-4o are designed with security in mind, using a local LLM backend may increase susceptibility to attacks.
- **Containerization**: To enhance security, it's strongly advised to operate the agent within a containerized environment.

## Contributing

Contributions are welcome! Feel free to fork the repository, submit issues, or create pull requests. For major changes, please open a discussion to propose your ideas.

## License

This project is licensed under the MIT License.

## Author

**Shreyan C (@thethinkmachine)**

*Developed as a college project and open-sourced for the community.*

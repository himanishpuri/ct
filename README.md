# ct (command teller)

ct is a command-line utility that converts natural language instructions into shell commands using various AI providers.

## install

```bash
curl -fsSL https://raw.githubusercontent.com/himanishpuri/ct/main/install.sh | bash
```

## usage

```
ct [--verbose] <instruction>
ct --version
ct --upgrade
ct --help
```

## example

```bash
ct get all the git branches
```

## configuration

the tool requires an API key from one of the supported providers. set one of the following environment variables:

- SF_LLM_GATEWAY_KEY (Salesforce LLM Gateway Express, OpenAI-compatible)
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GEMINI_API_KEY
- GROQ_API_KEY
- OLLAMA_MODEL (and optionally OLLAMA_HOST)

If `SF_LLM_GATEWAY_KEY` is set, it takes priority over other providers.

you can also configure specific models using:

- SF_LLM_GATEWAY_MODEL (default: claude-sonnet-4-5-20250929)
- OPENAI_MODEL
- ANTHROPIC_MODEL
- GEMINI_MODEL
- GROQ_MODEL

configuration and working models are stored in ~/.ct/config.

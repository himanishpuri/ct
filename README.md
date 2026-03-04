# ct (command teller)

ct is a command-line utility that converts natural language instructions into shell commands using various AI providers.

## usage

ct [--verbose] <instruction>
ct --version
ct --upgrade
ct --help

## example

ct get all the git branches

## configuration

the tool requires an API key from one of the supported providers. set one of the following environment variables:

- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GEMINI_API_KEY
- GROQ_API_KEY
- OLLAMA_MODEL (and optionally OLLAMA_HOST)

you can also configure specific models using:

- OPENAI_MODEL
- ANTHROPIC_MODEL
- GEMINI_MODEL
- GROQ_MODEL

configuration and working models are stored in ~/.ct/config.

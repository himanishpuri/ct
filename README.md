# ct (command teller)

ct is a command-line utility that converts natural language instructions into shell commands using various AI providers.

## install

```bash
curl -fsSL https://raw.githubusercontent.com/himanishpuri/ct/main/install.sh | bash
```

## usage

```
ct [--verbose] <instruction>
ct --setup
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

- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GEMINI_API_KEY
- GROQ_API_KEY
- MISTRAL_API_KEY
- OLLAMA_MODEL (and optionally OLLAMA_HOST)

you can also configure specific models using:

- OPENAI_MODEL
- ANTHROPIC_MODEL
- GEMINI_MODEL
- GROQ_MODEL
- MISTRAL_MODEL

configuration and working models are stored in ~/.ct/config.

## setup

run `ct --setup` to enter each provider key at an interactive prompt; it stores them in `~/.ct/keys` with **mode 0600**. ct also runs setup automatically when no key is configured anywhere.

- stored keys win over environment variables, per provider
- enter multiple keys for one provider as a comma-separated list at the prompt
- press enter on a blank line to skip a provider; lines already in the file stay untouched
- `OLLAMA_MODEL` and `OLLAMA_HOST` are stored too, so an ollama-only setup needs no env vars
- terminal echo is on while you type keys, so run setup somewhere private and never while screen-sharing
- values containing a quote or backslash are rejected
- env vars still work for CI and scripts when nothing is stored

## failover

if you configure several API keys, ct tries them in order until one works:

```bash
export OPENAI_API_KEY="key1,key2"   # multiple keys per provider
export GROQ_API_KEY="key3"
export MISTRAL_API_KEY="key4"       # or store it with ct --setup
```

- every credential is tried in order; a failure (bad key, rate limit, network error, dead model) falls through to the next
- the chain covers openai, anthropic, gemini, groq, mistral and ollama
- `~/.ct/config` keys control the chain:
  - `PROVIDER_ORDER="groq,openai,ollama"` — the try order (tokens like `openai:2` pick a specific key; providers not listed are appended)
  - `AUTO_REORDER="1"` — opt-in: a key/provider that fails auth or rate limiting is demoted to the end of the order for future runs, even if it later recovers
- `PROVIDER_ORDER` is never rewritten unless you enable `AUTO_REORDER` (model saves like `OPENAI_MODEL` still happen on success, as before)

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// config

type Config struct {
	OpenAIModel    string
	AnthropicModel string
	GeminiModel    string
	GroqModel      string
}

func configDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".ct")
}

func configFile() string {
	return filepath.Join(configDir(), "config")
}

// load config
func loadConfig() Config {
	cfg := Config{}
	data, err := os.ReadFile(configFile())
	if err != nil {
		return cfg
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if kv, ok := strings.CutPrefix(line, "OPENAI_MODEL="); ok {
			cfg.OpenAIModel = strings.Trim(kv, `"`)
		}
		if kv, ok := strings.CutPrefix(line, "ANTHROPIC_MODEL="); ok {
			cfg.AnthropicModel = strings.Trim(kv, `"`)
		}
		if kv, ok := strings.CutPrefix(line, "GEMINI_MODEL="); ok {
			cfg.GeminiModel = strings.Trim(kv, `"`)
		}
		if kv, ok := strings.CutPrefix(line, "GROQ_MODEL="); ok {
			cfg.GroqModel = strings.Trim(kv, `"`)
		}
	}
	return cfg
}

func saveConfig(key, value string) {
	if err := os.MkdirAll(configDir(), 0700); err != nil {
		return
	}
	content := fmt.Sprintf("%s=%q\n", key, value)
	_ = os.WriteFile(configFile(), []byte(content), 0600)
}

// help and version

func printHelp() {
	fmt.Println(`
Usage: ct [--verbose] <instruction>
       ct --version
       ct --upgrade
       ct --help

Example: ct get all the git branches

Options:
  --verbose    Enable debug output
  --version    Show version information
  --upgrade    Upgrade to the latest version
  --help, -h   Show this help message

Description:
  ct converts natural language instructions into shell commands.
  It supports OPENAI, ANTHROPIC, GEMINI, GROQ, and OLLAMA API providers.
  Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, or OLLAMA_MODEL`)
}

func printVersion() {
	exe, err := os.Executable()
	if err == nil {
		versionFile := filepath.Join(filepath.Dir(exe), "VERSION")
		if data, err := os.ReadFile(versionFile); err == nil {
			fmt.Print(string(data))
			return
		}
	}
	fmt.Println("version file not found")
}

func runUpgrade() {
	fmt.Println("upgrading ct utility...")

	tmpDir, err := os.MkdirTemp("", "ct-upgrade-*")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating temp dir: %v\n", err)
		os.Exit(1)
	}
	defer os.RemoveAll(tmpDir)

	installScript := filepath.Join(tmpDir, "install.sh")
	url := "https://raw.githubusercontent.com/himanishpuri/ct/main/install.sh"

	fmt.Println("downloading latest version...")
	if err := downloadFile(url, installScript); err != nil {
		fmt.Fprintf(os.Stderr, "error downloading install script: %v\n", err)
		os.Exit(1)
	}

	cmd := exec.Command("bash", installScript)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "error running install script: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("upgrade completed!")
}

func downloadFile(url, dest string) error {
	resp, err := http.Get(url) //nolint:gosec
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

// prompt

func buildPrompt(instruction string) string {
	wd, _ := os.Getwd()
	shell := os.Getenv("SHELL")
	if shell == "" {
		shell = "/bin/sh"
	}
	osName := runtime.GOOS

	return fmt.Sprintf(
		"You are a shell command generator. Convert the user's natural language instruction into a shell command.\n\n"+
			"Rules:\n"+
			"- Return ONLY the shell command, nothing else\n"+
			"- No explanations, no markdown formatting, no code block markers\n"+
			"- No backticks, no ```bash```, no comments\n"+
			"- Just the raw executable command(s)\n"+
			"- Use pipes (|) and operators (&&, ||) as needed\n"+
			"- If multiple commands are needed, combine them with && or ;\n\n"+
			"Context:\n"+
			"- Current directory: %s\n"+
			"- Shell: %s\n"+
			"- OS: %s\n\n"+
			"Instruction: %s\n\n"+
			"Command:",
		wd, shell, osName, instruction,
	)
}

// api providers

func postJSON(url string, headers map[string]string, payload any, debug bool) ([]byte, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	if debug {
		fmt.Fprintf(os.Stderr, "debug: POST %s\n", url)
	}
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func hasErrorKey(data []byte) bool {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return false
	}
	_, ok := m["error"]
	return ok
}

func jsonString(data []byte, keys ...string) string {
	var m any
	if err := json.Unmarshal(data, &m); err != nil {
		return ""
	}
	for _, k := range keys {
		mm, ok := m.(map[string]any)
		if !ok {
			return ""
		}
		m = mm[k]
	}
	if s, ok := m.(string); ok {
		return s
	}
	return ""
}

// OPENAI

func callOpenAI(apiKey, model, prompt string, debug bool) (string, error) {
	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	payload := map[string]any{
		"model":       model,
		"messages":    []message{{Role: "user", Content: prompt}},
		"temperature": 0.1,
		"max_tokens":  500,
	}
	headers := map[string]string{
		"Authorization": "Bearer " + apiKey,
	}
	data, err := postJSON("https://api.openai.com/v1/chat/completions", headers, payload, debug)
	if err != nil {
		return "", err
	}
	if debug {
		fmt.Fprintf(os.Stderr, "debug: full response: %s\n", data)
	}
	if hasErrorKey(data) {
		code := jsonString(data, "error", "code")
		if code == "model_not_found" || strings.Contains(string(data), "does not exist") {
			return "", nil // signal: try next model
		}
		msg := jsonString(data, "error", "message")
		if msg == "" {
			msg = string(data)
		}
		return "", fmt.Errorf("api error: %s", msg)
	}
	var resp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &resp); err != nil || len(resp.Choices) == 0 {
		return "", nil
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

func runOpenAI(apiKey, configuredModel string, cfg Config, debug bool, prompt string) (string, string, error) {
	models := dedup([]string{configuredModel, cfg.OpenAIModel, "gpt-4o-mini", "gpt-3.5-turbo"})
	for _, model := range models {
		if model == "" {
			continue
		}
		if debug {
			fmt.Fprintf(os.Stderr, "debug: trying OPENAI model: %s\n", model)
		}
		cmd, err := callOpenAI(apiKey, model, prompt, debug)
		if err != nil {
			return "", "", err
		}
		if cmd != "" {
			if debug {
				fmt.Fprintf(os.Stderr, "debug: saved working model: %s\n", model)
				fmt.Fprintf(os.Stderr, "debug: extracted command: %s\n", cmd)
			}
			return cmd, model, nil
		}
		if debug {
			fmt.Fprintf(os.Stderr, "debug: model %s not available, trying next...\n", model)
		}
	}
	return "", "", fmt.Errorf("no working OPENAI model found")
}

// ANTHROPIC

func callAnthropic(apiKey, model, prompt string, debug bool) (string, error) {
	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	payload := map[string]any{
		"model":      model,
		"max_tokens": 500,
		"messages":   []message{{Role: "user", Content: prompt}},
	}
	headers := map[string]string{
		"x-api-key":         apiKey,
		"anthropic-version": "2023-06-01",
	}
	data, err := postJSON("https://api.anthropic.com/v1/messages", headers, payload, debug)
	if err != nil {
		return "", err
	}
	if debug {
		fmt.Fprintf(os.Stderr, "debug: full response: %s\n", data)
	}
	if hasErrorKey(data) {
		errType := jsonString(data, "error", "type")
		if errType == "invalid_request_error" && strings.Contains(string(data), "model") {
			return "", nil // signal: try next model
		}
		msg := jsonString(data, "error", "message")
		if msg == "" {
			msg = string(data)
		}
		return "", fmt.Errorf("api error: %s", msg)
	}
	var resp struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(data, &resp); err != nil || len(resp.Content) == 0 {
		return "", nil
	}
	return strings.TrimSpace(resp.Content[0].Text), nil
}

func runAnthropic(apiKey, configuredModel string, cfg Config, debug bool, prompt string) (string, string, error) {
	models := dedup([]string{configuredModel, cfg.AnthropicModel, "claude-3-5-haiku-20241022", "claude-3-haiku-20240307"})
	for _, model := range models {
		if model == "" {
			continue
		}
		if debug {
			fmt.Fprintf(os.Stderr, "debug: trying ANTHROPIC model: %s\n", model)
		}
		cmd, err := callAnthropic(apiKey, model, prompt, debug)
		if err != nil {
			return "", "", err
		}
		if cmd != "" {
			if debug {
				fmt.Fprintf(os.Stderr, "debug: saved working model: %s\n", model)
				fmt.Fprintf(os.Stderr, "debug: extracted command: %s\n", cmd)
			}
			return cmd, model, nil
		}
		if debug {
			fmt.Fprintf(os.Stderr, "debug: model %s not available, trying next...\n", model)
		}
	}
	return "", "", fmt.Errorf("no working ANTHROPIC model found")
}

// GEMINI

func callGemini(apiKey, model, prompt string, debug bool) (string, error) {
	payload := map[string]any{
		"contents": []map[string]any{
			{"parts": []map[string]any{{"text": prompt}}},
		},
		"generationConfig": map[string]any{
			"temperature":     0.1,
			"maxOutputTokens": 500,
		},
	}
	url := fmt.Sprintf(
		"https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s",
		model, apiKey,
	)
	data, err := postJSON(url, nil, payload, debug)
	if err != nil {
		return "", err
	}
	if debug {
		fmt.Fprintf(os.Stderr, "debug: full response: %s\n", data)
	}
	if hasErrorKey(data) {
		code := jsonString(data, "error", "code")
		if code == "404" || strings.Contains(string(data), "not found") {
			return "", nil // signal: try next model
		}
		msg := jsonString(data, "error", "message")
		if msg == "" {
			msg = string(data)
		}
		return "", fmt.Errorf("api error: %s", msg)
	}
	var resp struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(data, &resp); err != nil ||
		len(resp.Candidates) == 0 ||
		len(resp.Candidates[0].Content.Parts) == 0 {
		return "", nil
	}
	return strings.TrimSpace(resp.Candidates[0].Content.Parts[0].Text), nil
}

func runGemini(apiKey, configuredModel string, cfg Config, debug bool, prompt string) (string, string, error) {
	models := dedup([]string{configuredModel, cfg.GeminiModel, "gemini-2.5-flash-exp", "gemini-1.5-flash", "gemini-pro"})
	for _, model := range models {
		if model == "" {
			continue
		}
		if debug {
			fmt.Fprintf(os.Stderr, "debug: trying GEMINI model: %s\n", model)
		}
		cmd, err := callGemini(apiKey, model, prompt, debug)
		if err != nil {
			return "", "", err
		}
		if cmd != "" {
			if debug {
				fmt.Fprintf(os.Stderr, "debug: saved working model: %s\n", model)
				fmt.Fprintf(os.Stderr, "debug: extracted command: %s\n", cmd)
			}
			return cmd, model, nil
		}
		if debug {
			fmt.Fprintf(os.Stderr, "debug: model %s not available, trying next...\n", model)
		}
	}
	return "", "", fmt.Errorf("no working GEMINI model found")
}

// GROQ

func callGroq(apiKey, model, prompt string, debug bool) (string, error) {
	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	payload := map[string]any{
		"model":       model,
		"messages":    []message{{Role: "user", Content: prompt}},
		"temperature": 0.1,
		"max_tokens":  500,
	}
	headers := map[string]string{
		"Authorization": "Bearer " + apiKey,
	}
	data, err := postJSON("https://api.groq.com/openai/v1/chat/completions", headers, payload, debug)
	if err != nil {
		return "", err
	}
	if debug {
		fmt.Fprintf(os.Stderr, "debug: full response: %s\n", data)
	}
	if hasErrorKey(data) {
		code := jsonString(data, "error", "code")
		if code == "model_not_found" || strings.Contains(string(data), "does not exist") {
			return "", nil // signal: try next model
		}
		msg := jsonString(data, "error", "message")
		if msg == "" {
			msg = string(data)
		}
		return "", fmt.Errorf("api error: %s", msg)
	}
	var resp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &resp); err != nil || len(resp.Choices) == 0 {
		return "", nil
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

func runGroq(apiKey, configuredModel string, cfg Config, debug bool, prompt string) (string, string, error) {
	models := dedup([]string{configuredModel, cfg.GroqModel, "llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"})
	for _, model := range models {
		if model == "" {
			continue
		}
		if debug {
			fmt.Fprintf(os.Stderr, "debug: trying GROQ model: %s\n", model)
		}
		cmd, err := callGroq(apiKey, model, prompt, debug)
		if err != nil {
			return "", "", err
		}
		if cmd != "" {
			if debug {
				fmt.Fprintf(os.Stderr, "debug: saved working model: %s\n", model)
				fmt.Fprintf(os.Stderr, "debug: extracted command: %s\n", cmd)
			}
			return cmd, model, nil
		}
		if debug {
			fmt.Fprintf(os.Stderr, "debug: model %s not available, trying next...\n", model)
		}
	}
	return "", "", fmt.Errorf("no working GROQ model found")
}

// OLLAMA

func runOllama(model, host, prompt string, debug bool) (string, error) {
	if host == "" {
		host = "http://localhost:11434"
	}
	if debug {
		fmt.Fprintf(os.Stderr, "debug: using OLLAMA model: %s at %s\n", model, host)
	}
	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	payload := map[string]any{
		"model":    model,
		"messages": []message{{Role: "user", Content: prompt}},
		"stream":   false,
	}
	data, err := postJSON(host+"/api/chat", nil, payload, debug)
	if err != nil {
		return "", err
	}
	if debug {
		fmt.Fprintf(os.Stderr, "debug: full response: %s\n", data)
	}
	if hasErrorKey(data) {
		msg := jsonString(data, "error")
		if msg == "" {
			msg = string(data)
		}
		return "", fmt.Errorf("api error: %s", msg)
	}
	var resp struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return "", nil
	}
	return strings.TrimSpace(resp.Message.Content), nil
}

// helpers

func dedup(ss []string) []string {
	seen := make(map[string]bool)
	out := make([]string, 0, len(ss))
	for _, s := range ss {
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	return out
}

func evalCommand(command string) error {
	shell := os.Getenv("SHELL")
	if shell == "" {
		shell = "/bin/sh"
	}
	cmd := exec.Command(shell, "-c", command)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// main

func main() {
	args := os.Args[1:]

	if len(args) == 0 {
		fmt.Println("Usage: ct [--verbose] <instruction>")
		fmt.Println("       ct --version")
		fmt.Println("       ct --upgrade")
		fmt.Println("       ct --help")
		fmt.Println()
		fmt.Println("Run 'ct --help' for more information.")
		os.Exit(1)
	}

	switch args[0] {
	case "--help", "-h":
		printHelp()
		os.Exit(0)
	case "--version":
		printVersion()
		os.Exit(0)
	case "--upgrade":
		runUpgrade()
		os.Exit(0)
	}

	debug := false
	if args[0] == "--verbose" {
		debug = true
		args = args[1:]
	}

	if len(args) == 0 {
		fmt.Println("Usage: ct [--verbose] <instruction>")
		os.Exit(1)
	}

	instruction := strings.Join(args, " ")

	if err := os.MkdirAll(configDir(), 0700); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not create config dir: %v\n", err)
	}
	cfg := loadConfig()

	openaiKey := os.Getenv("OPENAI_API_KEY")
	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")
	geminiKey := os.Getenv("GEMINI_API_KEY")
	groqKey := os.Getenv("GROQ_API_KEY")
	ollamaModel := os.Getenv("OLLAMA_MODEL")
	ollamaHost := os.Getenv("OLLAMA_HOST")

	provider := ""
	switch {
	case openaiKey != "":
		provider = "openai"
	case anthropicKey != "":
		provider = "anthropic"
	case geminiKey != "":
		provider = "gemini"
	case groqKey != "":
		provider = "groq"
	case ollamaModel != "":
		provider = "ollama"
	default:
		fmt.Fprintln(os.Stderr, "error: no api key found. set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, OLLAMA_MODEL")
		os.Exit(1)
	}

	if debug {
		fmt.Fprintf(os.Stderr, "debug: using api provider: %s\n", provider)
		fmt.Fprintf(os.Stderr, "debug: instruction: %s\n", instruction)
	}

	if v := os.Getenv("OPENAI_MODEL"); v != "" {
		cfg.OpenAIModel = v
	}
	if cfg.OpenAIModel == "" {
		cfg.OpenAIModel = "gpt-4o-mini"
	}
	if v := os.Getenv("ANTHROPIC_MODEL"); v != "" {
		cfg.AnthropicModel = v
	}
	if cfg.AnthropicModel == "" {
		cfg.AnthropicModel = "claude-3-5-haiku-20241022"
	}
	if v := os.Getenv("GEMINI_MODEL"); v != "" {
		cfg.GeminiModel = v
	}
	if cfg.GeminiModel == "" {
		cfg.GeminiModel = "gemini-2.0-flash-exp"
	}
	if v := os.Getenv("GROQ_MODEL"); v != "" {
		cfg.GroqModel = v
	}
	if cfg.GroqModel == "" {
		cfg.GroqModel = "llama-3.3-70b-versatile"
	}

	prompt := buildPrompt(instruction)

	var (
		command      string
		workingModel string
		err          error
	)

	switch provider {
	case "openai":
		command, workingModel, err = runOpenAI(openaiKey, cfg.OpenAIModel, cfg, debug, prompt)
		if err == nil && workingModel != "" {
			saveConfig("OPENAI_MODEL", workingModel)
		}

	case "anthropic":
		command, workingModel, err = runAnthropic(anthropicKey, cfg.AnthropicModel, cfg, debug, prompt)
		if err == nil && workingModel != "" {
			saveConfig("ANTHROPIC_MODEL", workingModel)
		}

	case "gemini":
		command, workingModel, err = runGemini(geminiKey, cfg.GeminiModel, cfg, debug, prompt)
		if err == nil && workingModel != "" {
			saveConfig("GEMINI_MODEL", workingModel)
		}

	case "groq":
		command, workingModel, err = runGroq(groqKey, cfg.GroqModel, cfg, debug, prompt)
		if err == nil && workingModel != "" {
			saveConfig("GROQ_MODEL", workingModel)
		}

	case "ollama":
		command, err = runOllama(ollamaModel, ollamaHost, prompt, debug)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error: api request failed\n%v\n", err)
		os.Exit(1)
	}

	if command == "" {
		fmt.Fprintln(os.Stderr, "error: failed to generate command")
		os.Exit(1)
	}

	fmt.Println("----------")
	fmt.Printf("\033[1;33m>>>\033[0m %s\n", command)
	fmt.Print("execute this command? (Y/n): ")

	reader := bufio.NewReader(os.Stdin)
	char, _, err := reader.ReadRune()
	fmt.Println()

	if err == nil && (char == 'n' || char == 'N') {
		fmt.Println("command execution cancelled")
		os.Exit(0)
	}

	if err := evalCommand(command); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			os.Exit(exitErr.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "error executing command: %v\n", err)
		os.Exit(1)
	}
}

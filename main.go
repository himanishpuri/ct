package main

import (
	"bufio"
	"bytes"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

//go:embed VERSION
var versionString string

// config

type Config struct {
	OpenAIModel    string
	AnthropicModel string
	MistralModel   string
	GeminiModel    string
	GroqModel      string
	ProviderOrder  string
	AutoReorder    bool
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
func keysFile() string {
	return filepath.Join(configDir(), "keys")
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
		if kv, ok := strings.CutPrefix(line, "MISTRAL_MODEL="); ok {
			cfg.MistralModel = strings.Trim(kv, `"`)
		}
		if kv, ok := strings.CutPrefix(line, "PROVIDER_ORDER="); ok {
			cfg.ProviderOrder = strings.Trim(kv, `"`)
		}
		if kv, ok := strings.CutPrefix(line, "AUTO_REORDER="); ok {
			cfg.AutoReorder = parseToggle(strings.Trim(kv, `"`))
		}
	}
	return cfg
}

func saveConfig(key, value string) {
	_ = writeKV(configFile(), key, value) // model saves stay best-effort, as before
}

// ensureConfigDir makes ~/.ct with 0700; tightens a pre-existing wider dir.
func ensureConfigDir() error {
	if err := os.MkdirAll(configDir(), 0700); err != nil {
		return err
	}
	if runtime.GOOS != "windows" {
		_ = os.Chmod(configDir(), 0700)
	}
	return nil
}

// atomicWrite writes via a 0600 temp file + rename, so the file is never
// half-written and loose permissions never survive a save.
func atomicWrite(path, content string) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".ct-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name()) // no-op after a successful rename
	if _, err := tmp.WriteString(content); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// writeKV upserts key="value" into path, then forces 0600 on the result.
func writeKV(path, key, value string) error {
	if err := ensureConfigDir(); err != nil {
		return err
	}
	data, _ := os.ReadFile(path) // missing file = start from nothing
	if err := atomicWrite(path, upsertLine(string(data), key, value)); err != nil {
		return err
	}
	if st, err := os.Stat(path); err == nil && runtime.GOOS != "windows" && st.Mode().Perm()&0o077 != 0 {
		if cerr := os.Chmod(path, 0600); cerr != nil {
			fmt.Fprintf(os.Stderr, "warning: %s is readable by others (mode %04o); run: chmod 600 %s\n", path, st.Mode().Perm(), path)
		}
	}
	return nil
}

// upsertLine returns content with the key's line replaced by key="value",
// appended if missing; every other line is preserved verbatim.
func upsertLine(content, key, value string) string {
	entry := fmt.Sprintf("%s=%q", key, value)
	if content == "" {
		return entry + "\n"
	}
	content = strings.TrimSuffix(content, "\n") // drop the split's trailing empty line
	lines := strings.Split(content, "\n")
	out := make([]string, 0, len(lines)+1)
	found := false
	for _, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), key+"=") {
			if !found {
				out = append(out, entry)
				found = true
			}
			continue // first occurrence wins; duplicates collapse
		}
		out = append(out, line)
	}
	if !found {
		out = append(out, entry)
	}
	return strings.Join(out, "\n") + "\n"
}

// parseToggle interprets a config value as on/off; unset or junk = off.
func parseToggle(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// key storage

var keyEnvNames = []string{
	"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
	"GROQ_API_KEY", "MISTRAL_API_KEY", "OLLAMA_MODEL", "OLLAMA_HOST",
}

var providerEnvs = []struct{ env, name string }{
	{"OPENAI_API_KEY", "openai"},
	{"ANTHROPIC_API_KEY", "anthropic"},
	{"GEMINI_API_KEY", "gemini"},
	{"GROQ_API_KEY", "groq"},
	{"MISTRAL_API_KEY", "mistral"},
}

func saveKey(name, value string) error {
	return writeKV(keysFile(), name, value)
}

// parseKeysFile pulls the recognized KEY="value" lines out of a keys file.
func parseKeysFile(data string) map[string]string {
	out := map[string]string{}
	for _, line := range strings.Split(data, "\n") {
		line = strings.TrimSpace(line)
		for _, name := range keyEnvNames {
			if kv, ok := strings.CutPrefix(line, name+"="); ok {
				out[name] = strings.Trim(kv, `"`)
			}
		}
	}
	return out
}

// loadKeys merges env and the stored keys file: a stored non-empty value
// wins over the environment for its setting.
func loadKeys() map[string]string {
	eff := map[string]string{}
	for _, name := range keyEnvNames {
		eff[name] = os.Getenv(name)
	}
	if st, err := os.Stat(keysFile()); err == nil && runtime.GOOS != "windows" && st.Mode().Perm()&0o077 != 0 {
		_ = os.Chmod(keysFile(), 0600) // best-effort tighten before reading
	}
	data, err := os.ReadFile(keysFile())
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			fmt.Fprintf(os.Stderr, "warning: could not read %s: %v\n", keysFile(), err)
		}
		return eff
	}
	for k, v := range parseKeysFile(string(data)) {
		if v != "" {
			eff[k] = v
		}
	}
	return eff
}

// redact hides a secret anywhere it appears in debug or error output.
func redact(s, secret string) string {
	if secret == "" {
		return s
	}
	return strings.ReplaceAll(s, secret, "***")
}

func stdinIsTTY() bool {
	fi, err := os.Stdin.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

// readAnswer prints the prompt on stderr and reads one line; ok=false on EOF
// with nothing typed.
func readAnswer(r *bufio.Reader, prompt string) (string, bool) {
	fmt.Fprint(os.Stderr, prompt)
	line, err := r.ReadString('\n')
	v := strings.TrimSpace(line)
	if err != nil && v == "" {
		return "", false
	}
	return v, true
}

// maskValue summarizes a stored value without printing it.
func maskValue(v string) string {
	ks := splitKeys(v)
	if len(ks) == 0 {
		return "not set"
	}
	last := ks[len(ks)-1]
	if len(last) > 4 {
		last = last[len(last)-4:]
	}
	return fmt.Sprintf("set: %d value(s), ending %s", len(ks), last)
}

// runSetup prompts for every provider key and stores them in ~/.ct/keys.
// ok=false means it was cancelled before writing anything.
func runSetup() bool {
	if !stdinIsTTY() {
		fmt.Fprintln(os.Stderr, "error: ct --setup needs an interactive terminal")
		os.Exit(1)
	}
	eff := loadKeys()
	fmt.Fprintf(os.Stderr, "ct setup — keys are stored in %s (mode 0600)\n", keysFile())
	fmt.Fprintln(os.Stderr, "warning: terminal echo is on - typed keys are visible. do not screen-share while entering keys.")
	reader := bufio.NewReader(os.Stdin)
	type kv struct{ key, value string }
	var answers []kv
	collect := func(name, prompt string) bool {
		v, ok := readAnswer(reader, prompt)
		if !ok {
			return false
		}
		if strings.ContainsAny(v, `"\`) {
			fmt.Fprintf(os.Stderr, "error: %s rejected: value contains a quote or backslash\n", name)
			return true
		}
		if v != "" {
			answers = append(answers, kv{name, v})
		}
		return true
	}
	for _, p := range providerEnvs {
		if !collect(p.env, fmt.Sprintf("%s [%s]: ", p.env, maskValue(eff[p.env]))) {
			fmt.Fprintln(os.Stderr, "error: setup cancelled (stdin closed); no keys saved")
			return false
		}
	}
	if !collect("OLLAMA_MODEL", fmt.Sprintf("OLLAMA_MODEL [%s]: ", maskValue(eff["OLLAMA_MODEL"]))) {
		fmt.Fprintln(os.Stderr, "error: setup cancelled (stdin closed); no keys saved")
		return false
	}
	if !collect("OLLAMA_HOST", "OLLAMA_HOST [default: http://localhost:11434]: ") {
		fmt.Fprintln(os.Stderr, "error: setup cancelled (stdin closed); no keys saved")
		return false
	}
	if len(answers) == 0 {
		fmt.Fprintln(os.Stderr, "no changes")
		return true
	}
	var names []string
	for _, a := range answers {
		if err := saveKey(a.key, a.value); err != nil {
			fmt.Fprintf(os.Stderr, "error: could not save %s: %v\n", a.key, err)
			os.Exit(1)
		}
		names = append(names, a.key)
	}
	fmt.Fprintf(os.Stderr, "saved %d value(s) to %s (mode 0600): %s\n", len(answers), keysFile(), strings.Join(names, ", "))
	return true
}

// help and version

func printHelp() {
	fmt.Println(`
Usage: ct [--verbose] <instruction>
       ct --version
       ct --upgrade
       ct --setup
       ct --help

Example: ct get all the git branches

Options:
  --verbose    Enable debug output
  --version    Show version information
  --upgrade    Upgrade to the latest version
  --setup     Configure API keys (stored in ~/.ct/keys)
  --help, -h   Show this help message

Description:
  ct converts natural language instructions into shell commands.
  It supports OPENAI, ANTHROPIC, GEMINI, GROQ, and OLLAMA API providers.
  It supports OPENAI, ANTHROPIC, GEMINI, GROQ, MISTRAL, and OLLAMA API providers.
  Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, or OLLAMA_MODEL
Failover:
  Multiple keys per provider: OPENAI_API_KEY="key1,key2" (each tried in order).
  On failure ct automatically tries the next key or provider until one works.
  ~/.ct/config keys:
    PROVIDER_ORDER="openai,groq" (or "openai:2" for one specific key) — the try order.
    AUTO_REORDER="1" — demotes a key/provider that failed auth or rate limits, for future runs.`)
}

func printVersion() {
	fmt.Println(strings.TrimSpace(versionString))
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
		if i := strings.Index(url, "key="); i >= 0 { // gemini embeds the key in the URL
			url = url[:i] + "key=***"
		}
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
	if _, ok := m["error"]; ok { // openai/groq/anthropic/ollama nested error
		return true
	}
	if _, ok := m["detail"]; ok { // mistral flat body (401/422, sometimes an array)
		return true
	}
	return string(m["object"]) == `"error"` // mistral documented error envelope
}

// modelUnavailable reports whether an API error response means the requested
// model is missing, decommissioned, or otherwise unusable — the signal to fall
// through to the next model in the list instead of aborting.
func modelUnavailable(data []byte) bool {
	s := strings.ToLower(string(data))
	markers := []string{
		"model_not_found", "model_decommissioned", "not_found_error",
		"does not exist", "decommissioned", "no longer supported",
		"not found for api", "has been removed", "is not found",
		"unknown_model", "invalid_model",
	}
	for _, m := range markers {
		if strings.Contains(s, m) {
			return true
		}
	}
	return false
}

// errAuthRateLimited marks a credential-level refusal: bad key, no quota, or
// throttling. Chain failover still tries the next credential on any error, but
// only this class demotes when AUTO_REORDER is enabled.
var errAuthRateLimited = errors.New("auth or rate limit")

// authOrRateLimited reports whether an API error response means the request
// was refused for account reasons — invalid key, quota/billing, rate limit.
func authOrRateLimited(data []byte) bool {
	s := strings.ToLower(string(data))
	markers := []string{
		"authentication_error", "invalid_api_key", "invalid x-api-key",
		"api key not valid", "incorrect api key", "invalid api key", "unauthorized", "401",
		"permission_denied", "permission error", "quota", "billing",
		"rate_limit_error", "rate_limit_exceeded", "rate limit",
		"resourceexhausted", "limit_exceeded", "429", "too many requests",
		"overloaded_error", "overloaded",
	}
	for _, m := range markers {
		if strings.Contains(s, m) {
			return true
		}
	}
	return false
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
		"model":                 model,
		"messages":              []message{{Role: "user", Content: prompt}},
		"max_completion_tokens": 2000,
	}
	headers := map[string]string{
		"Authorization": "Bearer " + apiKey,
	}
	data, err := postJSON("https://api.openai.com/v1/chat/completions", headers, payload, debug)
	if err != nil {
		return "", err
	}
	if debug {
		fmt.Fprintf(os.Stderr, "debug: full response: %s\n", redact(string(data), apiKey))
	}
	if hasErrorKey(data) {
		msg := jsonString(data, "error", "message")
		if msg == "" {
			msg = string(data)
		}
		if authOrRateLimited(data) {
			return "", fmt.Errorf("%w: %s", errAuthRateLimited, msg)
		}
		if modelUnavailable(data) {
			return "", nil // signal: try next model
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
	models := dedup([]string{configuredModel, cfg.OpenAIModel, "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6"})
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
		fmt.Fprintf(os.Stderr, "debug: full response: %s\n", redact(string(data), apiKey))
	}
	if hasErrorKey(data) {
		msg := jsonString(data, "error", "message")
		if msg == "" {
			msg = string(data)
		}
		if authOrRateLimited(data) {
			return "", fmt.Errorf("%w: %s", errAuthRateLimited, msg)
		}
		if modelUnavailable(data) {
			return "", nil // signal: try next model
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
	models := dedup([]string{configuredModel, cfg.AnthropicModel, "claude-sonnet-5", "claude-haiku-4-5", "claude-3-5-haiku-latest"})
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
			"maxOutputTokens": 2048,
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
		fmt.Fprintf(os.Stderr, "debug: full response: %s\n", redact(string(data), apiKey))
	}
	if hasErrorKey(data) {
		msg := jsonString(data, "error", "message")
		if msg == "" {
			msg = string(data)
		}
		if authOrRateLimited(data) {
			return "", fmt.Errorf("%w: %s", errAuthRateLimited, msg)
		}
		if modelUnavailable(data) {
			return "", nil // signal: try next model
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
	models := dedup([]string{configuredModel, cfg.GeminiModel, "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.5-flash"})
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
		"max_tokens":  2000,
	}
	headers := map[string]string{
		"Authorization": "Bearer " + apiKey,
	}
	data, err := postJSON("https://api.groq.com/openai/v1/chat/completions", headers, payload, debug)
	if err != nil {
		return "", err
	}
	if debug {
		fmt.Fprintf(os.Stderr, "debug: full response: %s\n", redact(string(data), apiKey))
	}
	if hasErrorKey(data) {
		msg := jsonString(data, "error", "message")
		if msg == "" {
			msg = string(data)
		}
		if authOrRateLimited(data) {
			return "", fmt.Errorf("%w: %s", errAuthRateLimited, msg)
		}
		if modelUnavailable(data) {
			return "", nil // signal: try next model
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
	models := dedup([]string{configuredModel, cfg.GroqModel, "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b", "openai/gpt-oss-120b"})
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

// MISTRAL

func callMistral(apiKey, model, prompt string, debug bool) (string, error) {
	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	payload := map[string]any{
		"model":      model,
		"messages":   []message{{Role: "user", Content: prompt}},
		"max_tokens": 2000,
	}
	headers := map[string]string{
		"Authorization": "Bearer " + apiKey,
	}
	data, err := postJSON("https://api.mistral.ai/v1/chat/completions", headers, payload, debug)
	if err != nil {
		return "", err
	}
	if debug {
		fmt.Fprintf(os.Stderr, "debug: full response: %s\n", redact(string(data), apiKey))
	}
	if hasErrorKey(data) {
		msg := jsonString(data, "detail") // mistral flat error bodies
		if msg == "" {
			msg = jsonString(data, "message") // documented error envelope
		}
		if msg == "" {
			msg = string(data)
		}
		if authOrRateLimited(data) {
			return "", fmt.Errorf("%w: %s", errAuthRateLimited, msg)
		}
		if modelUnavailable(data) {
			return "", nil // signal: try next model
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

func runMistral(apiKey, configuredModel string, cfg Config, debug bool, prompt string) (string, string, error) {
	models := dedup([]string{configuredModel, cfg.MistralModel, "mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "ministral-3-8b-latest"})
	for _, model := range models {
		if model == "" {
			continue
		}
		if debug {
			fmt.Fprintf(os.Stderr, "debug: trying MISTRAL model: %s\n", model)
		}
		cmd, err := callMistral(apiKey, model, prompt, debug)
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
	return "", "", fmt.Errorf("no working MISTRAL model found")
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
		if authOrRateLimited(data) {
			return "", fmt.Errorf("%w: %s", errAuthRateLimited, msg)
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

// credential chain — one credential per provider key; ollama is a single
// keyless credential.
type credential struct {
	provider string
	key      string
	idx      int // 1-based position within the provider's key list
}

var chainPrecedence = []string{"openai", "anthropic", "gemini", "groq", "mistral", "ollama"}

func splitKeys(s string) []string {
	var out []string
	for _, k := range strings.Split(s, ",") {
		if k = strings.TrimSpace(k); k != "" {
			out = append(out, k)
		}
	}
	return out
}

// buildChain lays out every credential to try, saved-order first. Tokens in
// PROVIDER_ORDER are "openai" (all of that provider's keys, env order) or
// "openai:2" (1-based key index). Invalid or duplicate tokens are skipped;
// uncovered keys are appended in chainPrecedence order.
func buildChain(keys map[string][]string, savedOrder string) []credential {
	if len(keys) == 0 {
		return nil
	}
	var chain []credential
	seen := map[string]bool{}
	add := func(c credential) {
		id := c.provider + ":" + strconv.Itoa(c.idx)
		if seen[id] {
			return
		}
		seen[id] = true
		chain = append(chain, c)
	}
	if savedOrder != "" {
		for _, tok := range strings.Split(savedOrder, ",") {
			tok = strings.TrimSpace(tok)
			if tok == "" {
				continue
			}
			name, idxStr, hasIdx := strings.Cut(tok, ":")
			ks, ok := keys[name]
			if !ok {
				continue
			}
			if !hasIdx {
				for i, k := range ks {
					add(credential{name, k, i + 1})
				}
				continue
			}
			idx, err := strconv.Atoi(idxStr)
			if err != nil || idx < 1 || idx > len(ks) {
				continue
			}
			add(credential{name, ks[idx-1], idx})
		}
	}
	for _, name := range chainPrecedence {
		ks, ok := keys[name]
		if !ok {
			continue
		}
		for i, k := range ks {
			add(credential{name, k, i + 1})
		}
	}
	return chain
}

func chainToken(c credential, keys map[string][]string) string {
	if len(keys[c.provider]) <= 1 {
		return c.provider
	}
	return fmt.Sprintf("%s:%d", c.provider, c.idx)
}

// reorderedTokens renders the chain with demoted credentials moved to the
// end (keeping their relative order); it is persisted as PROVIDER_ORDER.
func reorderedTokens(chain []credential, keys map[string][]string, demoted map[int]bool) []string {
	tokens := make([]string, 0, len(chain))
	for i, c := range chain {
		if !demoted[i] {
			tokens = append(tokens, chainToken(c, keys))
		}
	}
	for i, c := range chain {
		if demoted[i] {
			tokens = append(tokens, chainToken(c, keys))
		}
	}
	return tokens
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
		fmt.Println("       ct --setup")
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
	case "--setup":
		if runSetup() {
			os.Exit(0)
		}
		os.Exit(1)
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

	if err := ensureConfigDir(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not create config dir: %v\n", err)
	}
	cfg := loadConfig()

	keys := map[string][]string{}
	eff := loadKeys()
	for _, p := range providerEnvs {
		if ks := splitKeys(eff[p.env]); len(ks) > 0 {
			keys[p.name] = ks
		}
	}
	ollamaModel := eff["OLLAMA_MODEL"]
	ollamaHost := eff["OLLAMA_HOST"]
	if ollamaModel != "" {
		keys["ollama"] = []string{""}
	}

	chain := buildChain(keys, cfg.ProviderOrder)
	if len(chain) == 0 && stdinIsTTY() {
		fmt.Fprintln(os.Stderr, "no api key found. let's set one up (ct --setup)...")
		runSetup()
		eff = loadKeys()
		keys = map[string][]string{}
		for _, p := range providerEnvs {
			if ks := splitKeys(eff[p.env]); len(ks) > 0 {
				keys[p.name] = ks
			}
		}
		ollamaModel = eff["OLLAMA_MODEL"]
		ollamaHost = eff["OLLAMA_HOST"]
		if ollamaModel != "" {
			keys["ollama"] = []string{""}
		}
		chain = buildChain(keys, cfg.ProviderOrder)
	}
	if len(chain) == 0 {
		fmt.Fprintln(os.Stderr, "error: no api key found. run 'ct --setup' or set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, OLLAMA_MODEL")
		os.Exit(1)
	}

	if debug {
		fmt.Fprintf(os.Stderr, "debug: instruction: %s\n", instruction)
	}

	if v := os.Getenv("OPENAI_MODEL"); v != "" {
		cfg.OpenAIModel = v
	}
	if cfg.OpenAIModel == "" {
		cfg.OpenAIModel = "gpt-5.5"
	}
	if v := os.Getenv("ANTHROPIC_MODEL"); v != "" {
		cfg.AnthropicModel = v
	}
	if cfg.AnthropicModel == "" {
		cfg.AnthropicModel = "claude-sonnet-5"
	}
	if v := os.Getenv("GEMINI_MODEL"); v != "" {
		cfg.GeminiModel = v
	}
	if cfg.GeminiModel == "" {
		cfg.GeminiModel = "gemini-flash-latest"
	}
	if v := os.Getenv("GROQ_MODEL"); v != "" {
		cfg.GroqModel = v
	}
	if cfg.GroqModel == "" {
		cfg.GroqModel = "llama-3.3-70b-versatile"
	}
	if v := os.Getenv("MISTRAL_MODEL"); v != "" {
		cfg.MistralModel = v
	}
	if cfg.MistralModel == "" {
		cfg.MistralModel = "mistral-large-latest"
	}

	prompt := buildPrompt(instruction)

	var (
		command      string
		workingModel string
	)

	demoted := map[int]bool{}
	var lastErr error

	for i, c := range chain {
		if debug {
			fmt.Fprintf(os.Stderr, "debug: trying api provider: %s\n", chainToken(c, keys))
		}
		var err error
		switch c.provider {
		case "openai":
			command, workingModel, err = runOpenAI(c.key, cfg.OpenAIModel, cfg, debug, prompt)
			if err == nil && workingModel != "" {
				saveConfig("OPENAI_MODEL", workingModel)
			}
		case "anthropic":
			command, workingModel, err = runAnthropic(c.key, cfg.AnthropicModel, cfg, debug, prompt)
			if err == nil && workingModel != "" {
				saveConfig("ANTHROPIC_MODEL", workingModel)
			}
		case "gemini":
			command, workingModel, err = runGemini(c.key, cfg.GeminiModel, cfg, debug, prompt)
			if err == nil && workingModel != "" {
				saveConfig("GEMINI_MODEL", workingModel)
			}
		case "groq":
			command, workingModel, err = runGroq(c.key, cfg.GroqModel, cfg, debug, prompt)
			if err == nil && workingModel != "" {
				saveConfig("GROQ_MODEL", workingModel)
			}
		case "mistral":
			command, workingModel, err = runMistral(c.key, cfg.MistralModel, cfg, debug, prompt)
			if err == nil && workingModel != "" {
				saveConfig("MISTRAL_MODEL", workingModel)
			}
		case "ollama":
			command, err = runOllama(ollamaModel, ollamaHost, prompt, debug)
		}

		if err == nil && command != "" {
			break
		}

		if err == nil {
			fmt.Fprintf(os.Stderr, "warning: %s returned no command, trying next...\n", chainToken(c, keys))
		} else {
			lastErr = fmt.Errorf("%s", redact(err.Error(), c.key))
			fmt.Fprintf(os.Stderr, "warning: %s failed (%v), trying next...\n", chainToken(c, keys), lastErr)
			if errors.Is(err, errAuthRateLimited) {
				demoted[i] = true
			}
		}
		command = ""
	}

	if cfg.AutoReorder && len(demoted) > 0 {
		saveConfig("PROVIDER_ORDER", strings.Join(reorderedTokens(chain, keys, demoted), ","))
	}

	if command == "" {
		if lastErr != nil {
			fmt.Fprintf(os.Stderr, "error: all api providers failed. last error: %v\n", lastErr)
		} else {
			fmt.Fprintln(os.Stderr, "error: failed to generate command")
		}
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

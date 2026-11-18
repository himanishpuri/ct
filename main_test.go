package main

import (
	"fmt"
	"testing"
)

func TestModelUnavailable(t *testing.T) {
	unavailable := []string{
		// groq: decommissioned model (the reported bug)
		`{"error":{"message":"The model ` + "`llama3-8b-8192`" + ` has been decommissioned and is no longer supported.","type":"invalid_request_error","code":"model_decommissioned"}}`,
		// openai / groq: unknown model
		`{"error":{"message":"The model 'foo' does not exist","code":"model_not_found"}}`,
		// gemini: not found
		`{"error":{"code":404,"message":"models/gemini-x is not found for API version v1beta","status":"NOT_FOUND"}}`,
		// anthropic: unknown model
		`{"type":"error","error":{"type":"not_found_error","message":"model: claude-x"}}`,
		// mistral: documented error envelope for an unknown model
		`{"object":"error","message":"Model unknown","type":"invalid_request_error","param":"model","code":"unknown_model"}`,
		// mistral: model marker on a flat detail body
		`{"detail":"invalid_model: model id is invalid"}`,
	}
	for _, body := range unavailable {
		if !modelUnavailable([]byte(body)) {
			t.Errorf("expected modelUnavailable=true for: %s", body)
		}
	}

	available := []string{
		`{"error":{"message":"Rate limit reached","type":"rate_limit_error"}}`,
		`{"error":{"message":"invalid x-api-key","type":"authentication_error"}}`,
	}
	for _, body := range available {
		if modelUnavailable([]byte(body)) {
			t.Errorf("expected modelUnavailable=false for: %s", body)
		}
	}
}

func TestAuthOrRateLimited(t *testing.T) {
	// every sample here must also be modelUnavailable=false, and vice versa:
	// the two marker sets must stay disjoint so errors are classified once.
	refused := []string{
		// openai: invalid key
		`{"error":{"message":"Incorrect API key provided: sk-x","type":"invalid_request_error","code":"invalid_api_key"}}`,
		// anthropic: bad key
		`{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`,
		// openai/groq style rate limit
		`{"error":{"message":"Rate limit reached","type":"rate_limit_error"}}`,
		`{"error":{"message":"Rate limit reached","code":"rate_limit_exceeded"}}`,
		// gemini: quota
		`{"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}`,
		// anthropic: overloaded
		`{"type":"error","error":{"type":"overloaded_error"}}`,
		// mistral: live 401 body (no nested error key)
		`{"detail":"Invalid API Key"}`,
		// mistral: documented error envelope for a rate limit
		`{"object":"error","message":"rate limit","type":"rate_limit_error"}`,
	}
	for _, body := range refused {
		if !authOrRateLimited([]byte(body)) {
			t.Errorf("expected authOrRateLimited=true for: %s", body)
		}
		if modelUnavailable([]byte(body)) {
			t.Errorf("modelUnavailable must stay false for auth bodies: %s", body)
		}
	}

	accepted := []string{
		`{"choices":[{"message":{"content":"hi"}}]}`,
		`{"error":{"message":"The model 'foo' does not exist","code":"model_not_found"}}`,
		`[1,2]`,
	}
	for _, body := range accepted {
		if authOrRateLimited([]byte(body)) {
			t.Errorf("expected authOrRateLimited=false for: %s", body)
		}
	}
}

func TestHasErrorKey(t *testing.T) {
	cases := []struct {
		body string
		want bool
	}{
		{`{"detail":"Invalid API Key"}`, true},
		{`{"object":"error","message":"m","type":"invalid_request_error","code":"unknown_model"}`, true},
		{`{"object":"chat.completion","choices":[]}`, false},
		{`{"choices":[{"message":{"content":"hi"}}]}`, false},
		{`{"detail":["array"]}`, true},
		{`hello`, false},
	}
	for _, c := range cases {
		if got := hasErrorKey([]byte(c.body)); got != c.want {
			t.Errorf("hasErrorKey(%s) = %v, want %v", c.body, got, c.want)
		}
	}
}

func TestSplitKeys(t *testing.T) {
	if got := splitKeys("k1, k2,,k3"); fmt.Sprint(got) != "[k1 k2 k3]" {
		t.Errorf("splitKeys = %v, want [k1 k2 k3]", got)
	}
	if got := splitKeys(""); len(got) != 0 {
		t.Errorf("splitKeys(\"\") = %v, want empty", got)
	}
}

// chainNames renders chain as "provider:idx" words to compare orders.
func chainNames(chain []credential) string {
	out := ""
	for _, c := range chain {
		if out != "" {
			out += " "
		}
		out += fmt.Sprintf("%s:%d", c.provider, c.idx)
	}
	return out
}

func TestBuildChain(t *testing.T) {
	if got := buildChain(map[string][]string{}, ""); got != nil {
		t.Errorf("buildChain(empty) = %v, want nil", got)
	}

	all := map[string][]string{
		"openai":    {"oa-1", "oa-2"},
		"anthropic": {"an-1"},
		"gemini":    {"ge-1"},
		"groq":      {"gr-1"},
		"mistral":   {"mt-1"},
		"ollama":    {""},
	}
	if got, want := chainNames(buildChain(all, "")), "openai:1 openai:2 anthropic:1 gemini:1 groq:1 mistral:1 ollama:1"; got != want {
		t.Errorf("default precedence = %s, want %s", got, want)
	}
	if got, want := chainNames(buildChain(all, "groq,gemini")), "groq:1 gemini:1 openai:1 openai:2 anthropic:1 mistral:1 ollama:1"; got != want {
		t.Errorf("saved order groq,gemini = %s, want %s", got, want)
	}

	openaiOnly := map[string][]string{"openai": {"a", "b"}}
	if got, want := chainNames(buildChain(openaiOnly, "openai:2,openai:1")), "openai:2 openai:1"; got != want {
		t.Errorf("key reorder = %s, want %s", got, want)
	}

	two := map[string][]string{"openai": {"a"}, "groq": {"g"}}
	if got, want := chainNames(buildChain(two, "openai:9,nope")), "openai:1 groq:1"; got != want {
		t.Errorf("invalid tokens skipped = %s, want %s", got, want)
	}
	if got, want := chainNames(buildChain(two, "groq,groq")), "groq:1 openai:1"; got != want {
		t.Errorf("duplicate token = %s, want %s", got, want)
	}
}

func TestReorderedTokens(t *testing.T) {
	keys := map[string][]string{"openai": {"a", "b"}, "groq": {"c"}}
	chain := buildChain(keys, "") // openai:1 openai:2 groq:1

	if got, want := fmt.Sprint(reorderedTokens(chain, keys, map[int]bool{0: true})), "[openai:2 groq openai:1]"; got != want {
		t.Errorf("one demotion = %s, want %s", got, want)
	}
	if got, want := fmt.Sprint(reorderedTokens(chain, keys, map[int]bool{0: true, 2: true})), "[openai:2 openai:1 groq]"; got != want {
		t.Errorf("two demotions keep relative order = %s, want %s", got, want)
	}
	if got, want := fmt.Sprint(reorderedTokens(chain, keys, map[int]bool{})), "[openai:1 openai:2 groq]"; got != want {
		t.Errorf("no demotions = %s, want %s", got, want)
	}
}

func TestChainToken(t *testing.T) {
	if got, want := chainToken(credential{"ollama", "", 0}, map[string][]string{"ollama": {""}}), "ollama"; got != want {
		t.Errorf("ollama token = %s, want %s", got, want)
	}
	if got, want := chainToken(credential{"openai", "a", 1}, map[string][]string{"openai": {"a"}}), "openai"; got != want {
		t.Errorf("single-key token = %s, want %s", got, want)
	}
	if got, want := chainToken(credential{"openai", "b", 2}, map[string][]string{"openai": {"a", "b"}}), "openai:2"; got != want {
		t.Errorf("indexed token = %s, want %s", got, want)
	}
}

func TestUpsertLine(t *testing.T) {
	if got, want := upsertLine("", "OPENAI_MODEL", "gpt"), "OPENAI_MODEL=\"gpt\"\n"; got != want {
		t.Errorf("fresh config = %q, want %q", got, want)
	}
	if got, want := upsertLine("GROQ_MODEL=\"m\"\n", "OPENAI_MODEL", "gpt"), "GROQ_MODEL=\"m\"\nOPENAI_MODEL=\"gpt\"\n"; got != want {
		t.Errorf("append = %q, want %q", got, want)
	}
	if got, want := upsertLine("OPENAI_MODEL=\"old\"\nOTHER=\"x\"\n", "OPENAI_MODEL", "new"), "OPENAI_MODEL=\"new\"\nOTHER=\"x\"\n"; got != want {
		t.Errorf("replace keeps siblings = %q, want %q", got, want)
	}
	if got, want := upsertLine("PROVIDER_ORDER=\"a\"\nPROVIDER_ORDER=\"b\"\n", "PROVIDER_ORDER", "c"), "PROVIDER_ORDER=\"c\"\n"; got != want {
		t.Errorf("duplicates collapse = %q, want %q", got, want)
	}
}

func TestParseKeysFile(t *testing.T) {
	data := `OPENAI_API_KEY="sk-one"
IGNORE_ME="x"
GROQ_API_KEY="g1"
GROQ_API_KEY="g2"
ANTHROPIC_API_KEY=""
GEMINI_API_KEY="a"
GEMINI_API_KEY="b"
OLLAMA_HOST="http://localhost:11434"
no_mapping=here
`
	got := parseKeysFile(data)
	want := map[string]string{
		"OPENAI_API_KEY":    "sk-one",
		"GROQ_API_KEY":      "g2",
		"ANTHROPIC_API_KEY": "",
		"GEMINI_API_KEY":    "b",
		"OLLAMA_HOST":       "http://localhost:11434",
	}
	if len(got) != len(want) {
		t.Errorf("parseKeysFile returned %d entries, want %d: %v", len(got), len(want), got)
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("parseKeysFile[%s] = %q, want %q", k, got[k], v)
		}
	}
	if _, ok := got["IGNORE_ME"]; ok {
		t.Errorf("unknown line IGNORE_ME must be ignored: %v", got)
	}
	if _, ok := got["MISTRAL_API_KEY"]; ok {
		t.Errorf("missing key MISTRAL_API_KEY must be absent: %v", got)
	}
}

func TestParseToggle(t *testing.T) {
	for _, s := range []string{"1", "true", "ON", " yes "} {
		if !parseToggle(s) {
			t.Errorf("parseToggle(%q) = false, want true", s)
		}
	}
	for _, s := range []string{"", "0", "no", "maybe"} {
		if parseToggle(s) {
			t.Errorf("parseToggle(%q) = true, want false", s)
		}
	}
}

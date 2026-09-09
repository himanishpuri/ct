package main

import "testing"

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

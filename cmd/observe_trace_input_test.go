package cmd

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// --- Flag validation (validateObserveTraceInputFlags), table-driven (R14) ---

func TestValidateObserveTraceInputFlags(t *testing.T) {
	tests := []struct {
		name string
		// inputs
		inputHeader, inputData, workloadSpec, preset    string
		rateChanged                                     bool
		concurrency, concurrentSessions, totalSessions  int
		concurrentSessionsChanged, totalSessionsChanged bool
		// expectations
		wantEngaged bool
		wantErrSub  string // substring the error must contain; "" means expect no error
	}{
		{
			name:        "valid trace path",
			inputHeader: "h.yaml", inputData: "d.csv",
			concurrentSessions: 4, totalSessions: 0, concurrentSessionsChanged: true,
			wantEngaged: true, wantErrSub: "",
		},
		{
			name:        "header without data",
			inputHeader: "h.yaml",
			wantEngaged: false, wantErrSub: "must be provided together",
		},
		{
			name:        "data without header",
			inputData:   "d.csv",
			wantEngaged: false, wantErrSub: "must be provided together",
		},
		{
			name:        "trace + workload-spec mutually exclusive",
			inputHeader: "h.yaml", inputData: "d.csv", workloadSpec: "spec.yaml",
			concurrentSessions: 1,
			wantEngaged:        true, wantErrSub: "mutually exclusive",
		},
		{
			name:        "trace + preset mutually exclusive",
			inputHeader: "h.yaml", inputData: "d.csv", preset: "chatbot",
			concurrentSessions: 1,
			wantEngaged:        true, wantErrSub: "mutually exclusive",
		},
		{
			name:        "trace + rate mutually exclusive",
			inputHeader: "h.yaml", inputData: "d.csv", rateChanged: true,
			concurrentSessions: 1,
			wantEngaged:        true, wantErrSub: "mutually exclusive",
		},
		{
			name:        "trace + concurrency mutually exclusive",
			inputHeader: "h.yaml", inputData: "d.csv", concurrency: 8,
			concurrentSessions: 1,
			wantEngaged:        true, wantErrSub: "mutually exclusive",
		},
		{
			name:        "concurrent-sessions must be >= 1",
			inputHeader: "h.yaml", inputData: "d.csv", concurrentSessions: 0,
			wantEngaged: true, wantErrSub: "--concurrent-sessions must be >= 1",
		},
		{
			name:        "total-sessions must be >= 0",
			inputHeader: "h.yaml", inputData: "d.csv", concurrentSessions: 2, totalSessions: -1,
			wantEngaged: true, wantErrSub: "--total-sessions must be >= 0",
		},
		{
			name:                      "concurrent-sessions without trace path",
			concurrentSessions:        4,
			concurrentSessionsChanged: true,
			wantEngaged:               false, wantErrSub: "--concurrent-sessions requires the trace-input path",
		},
		{
			name:                 "total-sessions without trace path",
			totalSessions:        10,
			totalSessionsChanged: true,
			wantEngaged:          false, wantErrSub: "--total-sessions requires the trace-input path",
		},
		{
			name:        "no trace, no session flags",
			wantEngaged: false, wantErrSub: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engaged, msg := validateObserveTraceInputFlags(
				tt.inputHeader, tt.inputData, tt.workloadSpec, tt.preset,
				tt.rateChanged, tt.concurrency, tt.concurrentSessions, tt.totalSessions,
				tt.concurrentSessionsChanged, tt.totalSessionsChanged)
			if engaged != tt.wantEngaged {
				t.Errorf("engaged = %v, want %v", engaged, tt.wantEngaged)
			}
			if tt.wantErrSub == "" {
				if msg != "" {
					t.Errorf("expected no error, got %q", msg)
				}
				return
			}
			if !strings.Contains(msg, tt.wantErrSub) {
				t.Errorf("error %q does not contain %q", msg, tt.wantErrSub)
			}
		})
	}
}

func TestObserveCmd_TraceInputFlags_Exist(t *testing.T) {
	for _, name := range []string{"input-trace-header", "input-trace-data", "concurrent-sessions", "total-sessions"} {
		if observeCmd.Flags().Lookup(name) == nil {
			t.Errorf("observe command missing --%s flag", name)
		}
	}
}

// --- Orchestrator driven by a SessionPoolDriver (interface satisfaction) ---

// poolTrace: session A (2 rounds), B (1 round), C (1 round), first-seen A,B,C.
func poolTrace() *workload.TraceV2 {
	return &workload.TraceV2{
		Records: []workload.TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 40, OutputTokens: 8, ArrivalTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 50, OutputTokens: 9, ArrivalTimeUs: 5000},
			{RequestID: 3, SessionID: "B", RoundIndex: 0, InputTokens: 45, OutputTokens: 7, ArrivalTimeUs: 1000},
			{RequestID: 4, SessionID: "C", RoundIndex: 0, InputTokens: 42, OutputTokens: 6, ArrivalTimeUs: 2000},
		},
	}
}

// TestObserveOrchestrator_SessionPoolDriver_DrivesToCompletion verifies that
// runObserveOrchestrator can be driven by a *workload.SessionPoolDriver (through
// the workload.SessionDriver interface) to completion: every pooled session —
// including one admitted only after another terminates (C after B) and an
// intra-session follow-up (A round 1) — is dispatched exactly once, exercising
// the admission-via-followUp path. This is the behavioral companion to the
// compile-time SessionDriver assertions in sim/workload/session_driver.go.
func TestObserveOrchestrator_SessionPoolDriver_DrivesToCompletion(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"text": "ok"}},
			"usage":   map[string]any{"prompt_tokens": 40, "completion_tokens": 8},
		})
	}))
	defer server.Close()

	const seed int64 = 7
	r0Requests, blueprints, err := workload.LoadTraceV2SessionBlueprints(poolTrace(), seed, nil, math.MaxInt64)
	if err != nil {
		t.Fatalf("LoadTraceV2SessionBlueprints: %v", err)
	}
	driver, initial, err := workload.BuildSessionPool(blueprints, r0Requests, 2, 0, seed)
	if err != nil {
		t.Fatalf("BuildSessionPool: %v", err)
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}

	// driver is passed as workload.SessionDriver (interface) — the whole point.
	var sd workload.SessionDriver = driver
	runObserveOrchestrator(context.Background(), client, recorder, sd,
		cluster.NewSliceRequestSource(initial), false, 2, 0, nil, nil, false, false, 1.0)

	records := recorder.Records()
	// Expected dispatches: A r0, A r1, B r0, C r0 = 4 total.
	if len(records) != 4 {
		t.Fatalf("got %d records, want 4 (A r0, A r1, B r0, C r0): %+v", len(records), records)
	}

	type key struct {
		sid   string
		round int
	}
	seen := map[key]int{}
	for _, rec := range records {
		seen[key{rec.SessionID, rec.RoundIndex}]++
	}
	want := []key{{"A", 0}, {"A", 1}, {"B", 0}, {"C", 0}}
	for _, k := range want {
		if seen[k] != 1 {
			t.Errorf("expected exactly one dispatch of session %s round %d, got %d", k.sid, k.round, seen[k])
		}
	}
}

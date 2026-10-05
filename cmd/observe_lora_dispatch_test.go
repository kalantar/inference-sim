package cmd

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/spf13/cobra"
)

// LoRA adapter dispatch for `blis observe` (experimental, gated by
// --dispatch-adapters).
//
// Contract, in one line per law:
//
//	L1 the request body names the adapter when one is set, else the base --model
//	L2 the adapter id reaches PendingRequest from the generated sim.Request
//	L3 the gate is the ONLY place dispatch is enabled; off => adapter cleared
//	L4 the adapter is attributed into the output TraceV2 adapter column
//	L5 an output trace with no adapters carries no adapter column (inert)
//	L6 preflight refuses an adapter the target server does not serve
//
// L3 is what keeps the default path byte-identical: with the flag off the
// adapter never reaches the wire OR the trace, so neither the request body nor
// the exported columns change.

// --- L1: the wire decision -------------------------------------------------

func TestDispatchModelFor(t *testing.T) {
	tests := []struct {
		name    string
		adapter string
		base    string
		want    string
	}{
		{"adapter set wins", "sql-lora", "base-model", "sql-lora"},
		{"no adapter falls back to base", "", "base-model", "base-model"},
		{"empty base is still returned verbatim", "", "", ""},
		{"adapter set with empty base", "sql-lora", "", "sql-lora"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := dispatchModelFor(tc.adapter, tc.base); got != tc.want {
				t.Errorf("dispatchModelFor(%q, %q) = %q, want %q", tc.adapter, tc.base, got, tc.want)
			}
		})
	}
}

// TestSend_BodyNamesAdapter is the behavioral half of L1: it asserts on the JSON
// that actually leaves the process, not on the helper above.
func TestSend_BodyNamesAdapter(t *testing.T) {
	tests := []struct {
		name      string
		adapter   string
		wantModel string
	}{
		{"adapter dispatched", "sql-lora", "sql-lora"},
		{"no adapter uses base model", "", "base-model"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var gotModel string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]interface{}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Errorf("decode request body: %v", err)
				}
				if m, ok := body["model"].(string); ok {
					gotModel = m
				}
				resp := map[string]interface{}{
					"choices": []map[string]interface{}{{"text": "ok"}},
					"usage":   map[string]interface{}{"prompt_tokens": 1.0, "completion_tokens": 1.0},
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(resp)
			}))
			defer server.Close()

			client := NewRealClient(server.URL, "", "base-model", "vllm")
			if _, err := client.Send(context.Background(), &PendingRequest{
				RequestID: 0, InputTokens: 1, MaxOutputTokens: 1,
				Prompt: "hello", Adapter: tc.adapter,
			}); err != nil {
				t.Fatal(err)
			}
			if gotModel != tc.wantModel {
				t.Errorf("request body model = %q, want %q", gotModel, tc.wantModel)
			}
		})
	}
}

// --- L2: the adapter reaches the dispatcher --------------------------------

func TestRequestToPending_CarriesAdapter(t *testing.T) {
	req := &sim.Request{
		ID:           "r0",
		InputTokens:  make([]sim.TokenID, 10),
		MaxOutputLen: 5,
		Model:        "base-model",
		Adapter:      "sql-lora",
	}
	pending := requestToPending(req, 0, false, false, nil, nil, 1.0)
	if pending.Adapter != "sql-lora" {
		t.Errorf("PendingRequest.Adapter = %q, want %q", pending.Adapter, "sql-lora")
	}

	// A base-model-only request must stay adapter-blind.
	reqNoAdapter := &sim.Request{
		ID: "r1", InputTokens: make([]sim.TokenID, 10), MaxOutputLen: 5, Model: "base-model",
	}
	if p := requestToPending(reqNoAdapter, 1, false, false, nil, nil, 1.0); p.Adapter != "" {
		t.Errorf("adapter-blind request produced Adapter = %q, want empty", p.Adapter)
	}
}

// --- L3: the single gate ---------------------------------------------------

func TestGateAdapterDispatch(t *testing.T) {
	tests := []struct {
		name    string
		enabled bool
		adapter string
		want    string
	}{
		{"enabled keeps the adapter", true, "sql-lora", "sql-lora"},
		{"disabled clears the adapter", false, "sql-lora", ""},
		{"disabled is a no-op when adapter-blind", false, "", ""},
		{"enabled is a no-op when adapter-blind", true, "", ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := &PendingRequest{Adapter: tc.adapter}
			gateAdapterDispatch(p, tc.enabled)
			if p.Adapter != tc.want {
				t.Errorf("after gate(enabled=%v) Adapter = %q, want %q", tc.enabled, p.Adapter, tc.want)
			}
		})
	}
}

// --- L4 / L5: attribution into the output trace ----------------------------

func TestRecordRequest_CarriesAdapterIntoTrace(t *testing.T) {
	rec := &Recorder{}
	rec.RecordRequest(
		&PendingRequest{RequestID: 0, InputTokens: 10, Adapter: "sql-lora", Model: "base-model"},
		&RequestRecord{RequestID: 0, OutputTokens: 5, Status: "ok"},
		1000, "", 0,
	)
	records := rec.Records()
	if len(records) != 1 {
		t.Fatalf("got %d records, want 1", len(records))
	}
	if records[0].Adapter != "sql-lora" {
		t.Errorf("TraceRecord.Adapter = %q, want %q", records[0].Adapter, "sql-lora")
	}
}

// TestExportTrace_AdapterColumnInertWhenBlind pins L5: the adapter column is
// conditional, so an adapter-blind observe run exports the same columns it did
// before this feature existed.
func TestExportTrace_AdapterColumnInertWhenBlind(t *testing.T) {
	blind := []workload.TraceRecord{{RequestID: 0, InputTokens: 10, OutputTokens: 5, Status: "ok"}}
	withAdapter := []workload.TraceRecord{{RequestID: 0, InputTokens: 10, OutputTokens: 5, Status: "ok", Adapter: "sql-lora"}}

	blindCSV := exportRecordsToCSV(t, blind)
	adapterCSV := exportRecordsToCSV(t, withAdapter)

	blindHeader := strings.SplitN(blindCSV, "\n", 2)[0]
	adapterHeader := strings.SplitN(adapterCSV, "\n", 2)[0]

	if strings.Contains(blindHeader, "adapter") {
		t.Errorf("adapter-blind export declared an adapter column: %q", blindHeader)
	}
	if !strings.Contains(adapterHeader, "adapter") {
		t.Errorf("adapter-bearing export omitted the adapter column: %q", adapterHeader)
	}
}

// --- L6: preflight ---------------------------------------------------------

func TestListModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("preflight hit %q, want /v1/models", r.URL.Path)
		}
		resp := map[string]interface{}{
			"object": "list",
			"data": []map[string]interface{}{
				{"id": "base-model", "object": "model"},
				{"id": "sql-lora", "object": "model"},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "base-model", "vllm")
	ids, err := client.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(ids) != 2 || ids[0] != "base-model" || ids[1] != "sql-lora" {
		t.Errorf("ListModels = %v, want [base-model sql-lora]", ids)
	}
}

func TestListModels_ServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "base-model", "vllm")
	if _, err := client.ListModels(context.Background()); err == nil {
		t.Error("ListModels on a 500 returned nil error; a failed preflight must not read as an empty registry")
	}
}

func TestVerifyAdaptersServed(t *testing.T) {
	tests := []struct {
		name      string
		want      []string
		served    []string
		wantErr   bool
		errSubstr []string
	}{
		{
			name:   "all adapters served",
			want:   []string{"sql-lora", "code-lora"},
			served: []string{"base-model", "code-lora", "sql-lora"},
		},
		{
			name:      "unknown adapter refused naming it and what is served",
			want:      []string{"sql-lora", "typo-lora"},
			served:    []string{"base-model", "sql-lora"},
			wantErr:   true,
			errSubstr: []string{"typo-lora", "sql-lora"},
		},
		{
			name:      "server serves nothing",
			want:      []string{"sql-lora"},
			served:    nil,
			wantErr:   true,
			errSubstr: []string{"sql-lora"},
		},
		{
			name:   "no adapters referenced is vacuously satisfied",
			want:   nil,
			served: []string{"base-model"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := verifyAdaptersServed(tc.want, tc.served)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("verifyAdaptersServed(%v, %v) = nil, want error", tc.want, tc.served)
				}
				for _, substr := range tc.errSubstr {
					if !strings.Contains(err.Error(), substr) {
						t.Errorf("error %q does not name %q", err.Error(), substr)
					}
				}
				return
			}
			if err != nil {
				t.Errorf("verifyAdaptersServed(%v, %v) = %v, want nil", tc.want, tc.served, err)
			}
		})
	}
}

// --- flag validation -------------------------------------------------------

func TestValidateDispatchAdaptersFlags(t *testing.T) {
	tests := []struct {
		name             string
		dispatchAdapters bool
		corpusMode       bool
		wantErr          bool
	}{
		{"off in spec-mode", false, false, false},
		{"on in spec-mode", true, false, false},
		{"off in corpus-mode", false, true, false},
		{"on in corpus-mode is refused", true, true, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			msg := validateDispatchAdaptersFlags(tc.dispatchAdapters, tc.corpusMode)
			if tc.wantErr && msg == "" {
				t.Error("want a refusal, got none")
			}
			if !tc.wantErr && msg != "" {
				t.Errorf("want valid, got refusal %q", msg)
			}
			if tc.wantErr && !strings.Contains(msg, "--dispatch-adapters") {
				t.Errorf("refusal %q does not name the flag", msg)
			}
		})
	}
}

// TestDispatchAdaptersFlag_RegisteredOnObserveOnly pins the scope boundary:
// adapter dispatch is a real-server concern, so the flag belongs to observe.
// run/replay simulate the LoRA control plane and take --lora-config instead.
func TestDispatchAdaptersFlag_RegisteredOnObserveOnly(t *testing.T) {
	if observeCmd.Flags().Lookup("dispatch-adapters") == nil {
		t.Error("--dispatch-adapters is not registered on `blis observe`")
	}
	if replayCmd.Flags().Lookup("dispatch-adapters") != nil {
		t.Error("--dispatch-adapters is registered on `blis replay`, but adapter dispatch is observe-only")
	}
}

// exportRecordsToCSV writes records through the real TraceV2 exporter and returns
// the data CSV, so column-presence assertions test the shipped encoder rather
// than a reimplementation of it.
func exportRecordsToCSV(t *testing.T, records []workload.TraceRecord) string {
	t.Helper()
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "observed"}
	if err := workload.ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export: %v", err)
	}
	b, err := os.ReadFile(dataPath)
	if err != nil {
		t.Fatalf("read exported csv: %v", err)
	}
	return string(b)
}

// --- end-to-end: the wiring ------------------------------------------------

// writeAdapterSpec writes a two-client workload spec whose clients name distinct
// LoRA adapters over the same base model.
func writeAdapterSpec(t *testing.T, dir string) string {
	t.Helper()
	path := filepath.Join(dir, "adapters.yaml")
	spec := `version: "2"
seed: 42
category: language
aggregate_rate: 200.0
num_requests: 8

clients:
  - id: "sql-client"
    slo_class: "standard"
    adapter: "sql-lora"
    rate_fraction: 0.5
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: exponential
      params:
        mean: 16
    output_distribution:
      type: exponential
      params:
        mean: 4
  - id: "code-client"
    slo_class: "standard"
    adapter: "code-lora"
    rate_fraction: 0.5
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: exponential
      params:
        mean: 16
    output_distribution:
      type: exponential
      params:
        mean: 4
`
	if err := os.WriteFile(path, []byte(spec), 0o644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	return path
}

// adapterDispatchServer records every "model" value it is asked to complete and
// serves a /v1/models listing containing base + both adapters.
func adapterDispatchServer(t *testing.T, served []string) (*httptest.Server, func() []string) {
	t.Helper()
	var mu sync.Mutex
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/models" {
			data := make([]map[string]interface{}, 0, len(served))
			for _, id := range served {
				data = append(data, map[string]interface{}{"id": id, "object": "model"})
			}
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"object": "list", "data": data})
			return
		}
		var body map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if m, ok := body["model"].(string); ok {
			mu.Lock()
			seen = append(seen, m)
			mu.Unlock()
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "ok"}},
			"usage":   map[string]interface{}{"prompt_tokens": 16.0, "completion_tokens": 4.0},
		})
	}))
	return srv, func() []string {
		mu.Lock()
		defer mu.Unlock()
		out := append([]string(nil), seen...)
		return out
	}
}

// runObserveForAdapters drives a full spec-mode observe run and returns the
// exported trace plus every model name the server was asked to complete.
func runObserveForAdapters(t *testing.T, dispatch bool) (*workload.TraceV2, []string) {
	t.Helper()
	srv, seenModels := adapterDispatchServer(t, []string{"base-model", "sql-lora", "code-lora"})
	defer srv.Close()

	dir := t.TempDir()
	specPath := writeAdapterSpec(t, dir)
	outH, outD := filepath.Join(dir, "observed.yaml"), filepath.Join(dir, "observed.csv")

	withObserveFlagsRestored(t)
	// withObserveFlagsRestored predates this flag and does not snapshot it; restore
	// it here so a dispatch-on run cannot leak into another test in this process.
	prevDispatch := observeDispatchAdapters
	t.Cleanup(func() { observeDispatchAdapters = prevDispatch })
	observeServerURL, observeModel = srv.URL, "base-model"
	observeTraceHeader, observeTraceData = outH, outD
	observeWorkloadSpec = specPath
	observeWorkload, observeThinkTimeDist = "", ""
	observeCorpusHeader, observeCorpusData = "", ""
	observeConcurrentSessions, observeTotalSessions = 0, 0
	observeConcurrency, observeThinkTimeMs = 0, 0
	observeNumRequests = 8
	observeDuration, observePrewarmDuration = 0, 0
	observeShuffleCorpus, observeLazyGeneration, observeScrapeKVMetrics = false, false, false
	observeRecordITL, observeNoStreaming = false, true
	observeMaxConcur, observeWarmup, observeTimeout = 4, 0, 300
	observeAPIFormat, observeServerType, observeSeed = "completions", "vllm", 42
	observeDispatchAdapters = dispatch

	c := &cobra.Command{}
	for _, n := range []string{"total-sessions", "num-requests", "max-concurrency"} {
		c.Flags().Int(n, 0, "")
	}
	c.Flags().Float64("rate", 0, "")
	c.Flags().Int64("horizon", 0, "")
	c.Flags().Int64("seed", 42, "")
	c.Flags().Int("think-time-ms", 0, "")
	c.Flags().String("think-time-dist", "", "")

	done := make(chan struct{})
	go func() { defer close(done); runObserve(c, nil) }()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("runObserve did not return")
	}

	trace, err := workload.LoadTraceV2(outH, outD)
	if err != nil {
		t.Fatalf("exported trace does not load: %v", err)
	}
	return trace, seenModels()
}

// TestObserveE2E_DispatchAdaptersOn is the law that matters: with the flag on, the
// adapter ids reach the real wire and are attributed in the exported trace.
func TestObserveE2E_DispatchAdaptersOn(t *testing.T) {
	trace, models := runObserveForAdapters(t, true)

	if len(models) == 0 {
		t.Fatal("server received no completion requests")
	}
	// The base model is named exactly once, by the tokenizer-calibration probe
	// (calibratePrefixTokenRatio), which must measure the base tokenizer rather than
	// an adapter. Every MEASURED request names its adapter. Pinning the count rather
	// than merely "some adapter was seen" is what distinguishes real dispatch from a
	// run that quietly fell back to the base model for part of the workload.
	gotAdapters := map[string]int{}
	for _, m := range models {
		gotAdapters[m]++
	}
	if gotAdapters["base-model"] != 1 {
		t.Errorf("base model named %d times, want exactly 1 (the tokenizer-calibration probe); models seen: %v",
			gotAdapters["base-model"], models)
	}
	for _, want := range []string{"sql-lora", "code-lora"} {
		if gotAdapters[want] == 0 {
			t.Errorf("adapter %q never reached the wire; models seen: %v", want, models)
		}
	}
	if measured := len(models) - gotAdapters["base-model"]; measured != len(trace.Records) {
		t.Errorf("%d adapter-named requests on the wire but %d trace records; every measured request must name an adapter",
			measured, len(trace.Records))
	}

	traceAdapters := map[string]bool{}
	for _, r := range trace.Records {
		traceAdapters[r.Adapter] = true
	}
	if traceAdapters[""] {
		t.Error("exported trace has a record with no adapter; per-adapter attribution is incomplete")
	}
	for _, want := range []string{"sql-lora", "code-lora"} {
		if !traceAdapters[want] {
			t.Errorf("exported trace does not attribute adapter %q; got %v", want, traceAdapters)
		}
	}
}

// TestObserveE2E_DispatchAdaptersOff pins the default path: the adapter is dropped
// at the gate, so every request names the base model and the trace stays blind.
func TestObserveE2E_DispatchAdaptersOff(t *testing.T) {
	trace, models := runObserveForAdapters(t, false)

	if len(models) == 0 {
		t.Fatal("server received no completion requests")
	}
	for _, m := range models {
		if m != "base-model" {
			t.Errorf("request named %q with --dispatch-adapters off; want base-model only", m)
		}
	}
	for _, r := range trace.Records {
		if r.Adapter != "" {
			t.Errorf("exported trace carries adapter %q with dispatch off; want adapter-blind", r.Adapter)
		}
	}
}

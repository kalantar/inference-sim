package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/workload"
)

func TestValidateObserveCorpusFlags(t *testing.T) {
	cases := []struct {
		name                          string
		concurrentSessions, totalSess int
		corpusHeader, corpusData      string
		workload, workloadSpec        string
		rateChanged                   bool
		concurrency                   int
		thinkTimeMs                   int
		thinkTimeDist                 string
		lazyGeneration                bool
		horizonChanged                bool
		numRequestsChanged            bool
		shuffleCorpus                 bool
		totalSessChanged              bool
		duration                      time.Duration
		wantErrSubstr                 string // "" = expect valid
	}{
		{name: "spec-mode untouched", workload: "chatbot", rateChanged: true, wantErrSubstr: ""},
		{name: "valid corpus", concurrentSessions: 4, totalSess: 20, corpusHeader: "h.yaml", corpusData: "d.csv", wantErrSubstr: ""},
		{name: "valid corpus with shuffle", concurrentSessions: 4, totalSess: 20, corpusHeader: "h.yaml", corpusData: "d.csv", shuffleCorpus: true, wantErrSubstr: ""},
		{name: "corpus needs both files", concurrentSessions: 4, corpusHeader: "h.yaml", wantErrSubstr: "--corpus-data"},
		{name: "corpus + concurrency conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", concurrency: 8, wantErrSubstr: "--concurrency"},
		{name: "corpus + workload conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", workload: "chatbot", wantErrSubstr: "--workload"},
		{name: "corpus + workload-spec conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", workloadSpec: "w.yaml", wantErrSubstr: "--workload-spec"},
		{name: "corpus + rate conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", rateChanged: true, wantErrSubstr: "--rate"},
		{name: "corpus files without concurrent-sessions", corpusHeader: "h.yaml", corpusData: "d.csv", wantErrSubstr: "--concurrent-sessions"},
		{name: "total-sessions without concurrent-sessions", totalSess: 10, wantErrSubstr: "--concurrent-sessions"},
		{name: "corpus + think-time-ms conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", thinkTimeMs: 200, wantErrSubstr: "--think-time-ms"},
		{name: "corpus + think-time-dist conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", thinkTimeDist: "constant:value=500ms", wantErrSubstr: "--think-time-dist"},
		{name: "corpus + lazy-generation conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", lazyGeneration: true, wantErrSubstr: "--lazy-generation"},
		{name: "corpus + horizon conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", horizonChanged: true, wantErrSubstr: "--horizon"},
		{name: "corpus + num-requests conflict", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", numRequestsChanged: true, wantErrSubstr: "--num-requests"},
		{name: "shuffle-corpus without concurrent-sessions", shuffleCorpus: true, wantErrSubstr: "--shuffle-corpus"},

		// --duration: the clock-bounded alternative to --total-sessions.
		{name: "valid corpus bounded by duration", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", duration: 20 * time.Minute, wantErrSubstr: ""},
		{name: "duration with shuffle is fine", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", duration: 20 * time.Minute, shuffleCorpus: true, wantErrSubstr: ""},
		{name: "duration without concurrent-sessions", duration: 20 * time.Minute, wantErrSubstr: "--duration"},
		{name: "duration excludes total-sessions", concurrentSessions: 4, totalSess: 20, totalSessChanged: true, corpusHeader: "h.yaml", corpusData: "d.csv", duration: 20 * time.Minute, wantErrSubstr: "--total-sessions"},
		{name: "duration alone is fine when total-sessions was never supplied", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", duration: 20 * time.Minute, wantErrSubstr: ""},
		{name: "negative duration refused", concurrentSessions: 4, corpusHeader: "h.yaml", corpusData: "d.csv", duration: -1 * time.Minute, wantErrSubstr: "must be positive"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := validateObserveCorpusFlags(tc.concurrentSessions, tc.totalSess, tc.totalSessChanged, tc.corpusHeader, tc.corpusData, tc.workload, tc.workloadSpec, tc.rateChanged, tc.concurrency, tc.thinkTimeMs, tc.thinkTimeDist, tc.lazyGeneration, tc.horizonChanged, tc.numRequestsChanged, tc.shuffleCorpus, tc.duration)
			if tc.wantErrSubstr == "" && got != "" {
				t.Errorf("expected valid, got error %q", got)
			}
			if tc.wantErrSubstr != "" && !strings.Contains(got, tc.wantErrSubstr) {
				t.Errorf("error %q does not mention %q", got, tc.wantErrSubstr)
			}
		})
	}
}

func TestBuildObserveCorpusPool_DuplicatesToTarget(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "corpus.yaml")
	dataPath := filepath.Join(dir, "corpus.csv")

	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
	records := []workload.TraceRecord{
		{RequestID: 0, SessionID: "s0", RoundIndex: 0, InputTokens: 100, OutputTokens: 10, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 0, InputTokens: 120, OutputTokens: 12, ArrivalTimeUs: 0, Status: "ok"},
	}
	if err := workload.ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export corpus: %v", err)
	}

	driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, 2, 5, false, 42, false)
	if err != nil {
		t.Fatalf("buildObserveCorpusPool: %v", err)
	}
	if driver.TotalSessions() != 5 {
		t.Errorf("TotalSessions = %d, want 5 (duplicate-to-fill)", driver.TotalSessions())
	}
	if len(initial) != 2 {
		t.Errorf("initial requests = %d, want 2 (concurrent-sessions)", len(initial))
	}
}

func TestBuildObserveCorpusPool_EmptyCorpusErrors(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "empty.yaml")
	dataPath := filepath.Join(dir, "empty.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated"}
	if err := workload.ExportTraceV2(header, []workload.TraceRecord{}, headerPath, dataPath); err != nil {
		t.Fatalf("export: %v", err)
	}
	_, _, err := buildObserveCorpusPool(headerPath, dataPath, 2, 4, false, 42, false)
	if err == nil {
		t.Fatal("expected error for empty corpus, got nil")
	}
}

// TestBuildObserveCorpusPool_MixedNonSessionErrors is the observe-side parity of
// the replay non-session guard (PR-C review follow-up: "same behavior in observe
// needs to be addressed"). A corpus that mixes session records with non-session
// (empty session_id) records cannot be pooled 1:1; buildObserveCorpusPool must
// return an actionable "no session_id" error, pre-empting BuildSessionPool's
// internal "count mismatch" wording (R1).
func TestBuildObserveCorpusPool_MixedNonSessionErrors(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "mixed.yaml")
	dataPath := filepath.Join(dir, "mixed.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated"}
	// One session record + one non-session (empty SessionID) record.
	records := []workload.TraceRecord{
		{RequestID: 0, SessionID: "s1", RoundIndex: 0, InputTokens: 10, OutputTokens: 5, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "", RoundIndex: 0, InputTokens: 8, OutputTokens: 4, ArrivalTimeUs: 0, Status: "ok"},
	}
	if err := workload.ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export: %v", err)
	}
	_, _, err := buildObserveCorpusPool(headerPath, dataPath, 1, 0, false, 42, false)
	if err == nil {
		t.Fatal("expected error for a corpus mixing session and non-session records, got nil")
	}
	if !strings.Contains(err.Error(), "no session_id") {
		t.Errorf("error should name the non-session records ('no session_id'), got: %v", err)
	}
	if strings.Contains(err.Error(), "count mismatch") {
		t.Errorf("guard should pre-empt BuildSessionPool's internal 'count mismatch', but it surfaced: %v", err)
	}
}

// TestBuildObserveCorpusPool_ShuffleReproducibleAndReorders verifies observe's
// --shuffle-corpus (PR-C parity, #1480): the seeded permutation is reproducible
// from the same --seed, actually reorders the corpus, and preserves the set;
// shuffle=false keeps file order. observe draws from the SAME salted stream as
// `blis replay --shuffle-corpus`, so one --seed selects the same subset on both.
func TestBuildObserveCorpusPool_ShuffleReproducibleAndReorders(t *testing.T) {
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "corpus.yaml")
	dataPath := filepath.Join(dir, "corpus.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
	var records []workload.TraceRecord
	for i := 0; i < 6; i++ {
		records = append(records, workload.TraceRecord{
			RequestID: i, SessionID: fmt.Sprintf("s%d", i), RoundIndex: 0,
			InputTokens: 100 + i, OutputTokens: 10, ArrivalTimeUs: 0, Status: "ok",
		})
	}
	if err := workload.ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export corpus: %v", err)
	}
	// joinIDs returns the comma-joined SessionID sequence of the initial injection
	// (concurrent == total == 6 ⇒ all injected, in queued/admission order).
	joinIDs := func(reqs []*sim.Request) string {
		s := ""
		for _, r := range reqs {
			s += r.SessionID + ","
		}
		return s
	}
	const fileOrder = "s0,s1,s2,s3,s4,s5,"

	// shuffle=false → file order.
	_, initNoShuf, err := buildObserveCorpusPool(headerPath, dataPath, 6, 6, false, 42, false)
	if err != nil {
		t.Fatalf("no-shuffle: %v", err)
	}
	if got := joinIDs(initNoShuf); got != fileOrder {
		t.Errorf("shuffle=false order = %q, want file order %q", got, fileOrder)
	}

	// shuffle=true, same seed twice → identical (reproducible).
	_, initA, err := buildObserveCorpusPool(headerPath, dataPath, 6, 6, true, 7, false)
	if err != nil {
		t.Fatalf("shuffle A: %v", err)
	}
	_, initB, err := buildObserveCorpusPool(headerPath, dataPath, 6, 6, true, 7, false)
	if err != nil {
		t.Fatalf("shuffle B: %v", err)
	}
	ordA, ordB := joinIDs(initA), joinIDs(initB)
	if ordA != ordB {
		t.Errorf("same seed → different order:\n %q\n %q", ordA, ordB)
	}
	// Actually reorders (seed 7 permutes 6 elements away from identity).
	if ordA == fileOrder {
		t.Errorf("shuffle=true seed=7 did not reorder; got file order %q", ordA)
	}
	// Set preserved: the same 6 sessions, none dropped or duplicated.
	seen := map[string]bool{}
	for _, r := range initA {
		seen[r.SessionID] = true
	}
	if len(seen) != 6 {
		t.Errorf("shuffle changed the session set: %d unique, want 6", len(seen))
	}
}

// TestObserveCorpusMode_DrainsAllSessions is the load-bearing corpus-mode test:
// a 2-session corpus scaled to --total-sessions 6 at --concurrent-sessions 2
// must dispatch and complete exactly 6 sessions against the (mock) server, with
// the dispatch loop draining to completion (not hanging). Single-round sessions
// ⇒ sessions == distinct recorded SessionIDs. This proves refill-on-terminate:
// the initial 2 are counted via takePreGen, each terminating session's refill
// replaces it (serializer does not decrement while a follow-up is returned), and
// only the final 2 decrement to 0 — so all 6 run and the loop exits.
func TestObserveCorpusMode_DrainsAllSessions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hello"}},
			"usage":   map[string]interface{}{"prompt_tokens": 100, "completion_tokens": 10},
		})
	}))
	defer server.Close()

	// 2 single-round sessions → pool duplicates to 6.
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "corpus.yaml")
	dataPath := filepath.Join(dir, "corpus.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
	records := []workload.TraceRecord{
		{RequestID: 0, SessionID: "s0", RoundIndex: 0, InputTokens: 100, OutputTokens: 10, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 0, InputTokens: 120, OutputTokens: 12, ArrivalTimeUs: 0, Status: "ok"},
	}
	if err := workload.ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export corpus: %v", err)
	}

	driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, 2, 6, false, 42, false)
	if err != nil {
		t.Fatalf("buildObserveCorpusPool: %v", err)
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}

	// Guard against a hang (the failure mode a broken active-session count would
	// cause): run the orchestrator in a goroutine and fail if it does not return.
	done := make(chan struct{})
	go func() {
		defer close(done)
		runObserveOrchestrator(context.Background(), client, recorder, driver,
			cluster.NewSliceRequestSource(initial), true, 2, 0, nil, nil, false, false, 1.0, 0)
	}()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("orchestrator did not drain within 30s — pool likely stalled (active-session accounting)")
	}

	// Exactly 6 distinct sessions must have completed.
	sessions := make(map[string]bool)
	for _, rec := range recorder.Records() {
		if rec.SessionID != "" {
			sessions[rec.SessionID] = true
		}
	}
	if len(sessions) != 6 {
		t.Errorf("distinct completed sessions = %d, want 6 (duplicate-to-fill + refill drain)", len(sessions))
	}
}

// TestObserveCorpusMode_MultiRoundAccumulate drives a multi-round accumulate-mode
// corpus end-to-end through the observe orchestrator (#1487 review, recommended):
// a single 3-round session must dispatch round 0 plus both follow-ups, confirming
// the SessionManager follow-up path works over the real-server dispatch loop.
func TestObserveCorpusMode_MultiRoundAccumulate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hello"}},
			"usage":   map[string]interface{}{"prompt_tokens": 100, "completion_tokens": 10},
		})
	}))
	defer server.Close()

	// One 3-round accumulate session (per-round input deltas). Small recorded think
	// times keep the closed-loop wall-clock pacing fast.
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "corpus.yaml")
	dataPath := filepath.Join(dir, "corpus.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
	records := []workload.TraceRecord{
		{RequestID: 0, SessionID: "s0", RoundIndex: 0, InputTokens: 100, OutputTokens: 10, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s0", RoundIndex: 1, InputTokens: 40, OutputTokens: 10, ArrivalTimeUs: 1000, ThinkTimeUs: i64p(1000), Status: "ok"},
		{RequestID: 2, SessionID: "s0", RoundIndex: 2, InputTokens: 25, OutputTokens: 10, ArrivalTimeUs: 2000, ThinkTimeUs: i64p(1000), Status: "ok"},
	}
	if err := workload.ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export corpus: %v", err)
	}

	driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, 1, 1, false, 42, false)
	if err != nil {
		t.Fatalf("buildObserveCorpusPool: %v", err)
	}
	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}
	done := make(chan struct{})
	go func() {
		defer close(done)
		runObserveOrchestrator(context.Background(), client, recorder, driver,
			cluster.NewSliceRequestSource(initial), true, 1, 0, nil, nil, false, false, 1.0, 0)
	}()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("orchestrator did not drain within 30s — multi-round follow-up dispatch likely stalled")
	}

	// All three rounds (round 0 + two accumulate follow-ups) must have dispatched.
	rounds := map[int]bool{}
	for _, rec := range recorder.Records() {
		if rec.SessionID == "s0" {
			rounds[rec.RoundIndex] = true
		}
	}
	if !rounds[0] || !rounds[1] || !rounds[2] {
		t.Errorf("session s0 dispatched rounds %v, want {0,1,2} (multi-round accumulate follow-ups)", rounds)
	}
}

// TestValidateObserveCorpusFlags_HorizonRefusalOffersBothBounds pins that the
// --horizon refusal points at BOTH ways to size a corpus run. --horizon stays
// rejected (it bounds spec-mode generation, and a corpus is read from a file), but
// an operator who reached for it wants a run length — so the message has to name the
// flags that actually give one, or it diagnoses without directing.
func TestValidateObserveCorpusFlags_HorizonRefusalOffersBothBounds(t *testing.T) {
	got := validateObserveCorpusFlags(4, 0, false, "h.yaml", "d.csv", "", "", false, 0, 0, "", false, true, false, false, 0)
	if got == "" {
		t.Fatal("--horizon must still be refused in corpus mode")
	}
	for _, want := range []string{"--horizon", "--total-sessions", "--duration"} {
		if !strings.Contains(got, want) {
			t.Errorf("refusal %q does not mention %q", got, want)
		}
	}
}

// mockCorpusServer returns an httptest server that answers every completion
// request instantly, and a 2-session single-round corpus on disk. Shared by the
// duration-bound end-to-end tests below.
func mockCorpusServer(t *testing.T) (*httptest.Server, string, string) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hello"}},
			"usage":   map[string]interface{}{"prompt_tokens": 100, "completion_tokens": 10},
		})
	}))
	t.Cleanup(server.Close)

	dir := t.TempDir()
	headerPath := filepath.Join(dir, "corpus.yaml")
	dataPath := filepath.Join(dir, "corpus.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
	records := []workload.TraceRecord{
		{RequestID: 0, SessionID: "s0", RoundIndex: 0, InputTokens: 100, OutputTokens: 10, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 0, InputTokens: 120, OutputTokens: 12, ArrivalTimeUs: 0, Status: "ok"},
	}
	if err := workload.ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatalf("export corpus: %v", err)
	}
	return server, headerPath, dataPath
}

// runCorpusPoolRecorded drives a pool through the orchestrator with an optional
// dispatch bound and returns the recorder plus whether the bound stopped it.
func runCorpusPoolRecorded(t *testing.T, serverURL string, driver *workload.SessionPoolDriver, initial []*sim.Request, concurrent int, bound time.Duration) (*Recorder, bool) {
	t.Helper()
	client := NewRealClient(serverURL, "", "test-model", "vllm")
	recorder := &Recorder{}
	var stopped bool
	done := make(chan struct{})
	go func() {
		defer close(done)
		stopped = runObserveOrchestrator(context.Background(), client, recorder, driver,
			cluster.NewSliceRequestSource(initial), true, concurrent, 0, nil, nil, false, false, 1.0, bound)
	}()
	select {
	case <-done:
	// 5s, not 60s: the point is to notice a STALL. At 60s a bound-aware select arm could
	// be deleted and this would still pass, just slowly.
	case <-time.After(5 * time.Second):
		t.Fatal("orchestrator did not drain within 5s — pool likely stalled")
	}
	return recorder, stopped
}

// TestObserveCorpusMode_DurationKeepsPoolFullPastCorpus is the end-to-end proof that
// the queue is open-ended under a clock bound: a 2-session corpus driven for a
// window that comfortably fits many instant round trips must run MORE than 2
// sessions, by cycling the corpus with cache-busting clones. Asserts only an
// inequality, so it does not depend on machine speed.
func TestObserveCorpusMode_DurationKeepsPoolFullPastCorpus(t *testing.T) {
	server, headerPath, dataPath := mockCorpusServer(t)

	// 300ms is ample against an instant mock server; a longer bound only slows the suite.
	const bound = 300 * time.Millisecond
	driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, 2, 0, false, 42, true)
	if err != nil {
		t.Fatalf("buildObserveCorpusPool: %v", err)
	}
	// Pass the dispatch bound too, so this exercises the wiring the CLI produces.
	recorder, stopped := runCorpusPoolRecorded(t, server.URL, driver, initial, 2, bound)

	sessions := make(map[string]bool)
	for _, rec := range recorder.Records() {
		if rec.SessionID != "" {
			sessions[rec.SessionID] = true
		}
	}
	if len(sessions) <= 2 {
		t.Errorf("distinct sessions dispatched = %d, want > 2; a duration-bounded pool must "+
			"keep cycling the 2-session corpus with clones rather than draining after one pass", len(sessions))
	}
	if !stopped {
		t.Error("expected the bound to stop the run")
	}
	// Unstarted() is 0 by construction for an open-ended queue, so assert the refill law
	// instead: each termination admits at most one replacement, so the pool never grows
	// beyond its initial size plus one admission per terminated session.
	if started, ended := driver.SessionsStarted(), driver.SessionsTerminated(); started > 2+ended {
		t.Errorf("started=%d exceeds pool(2) + terminated(%d): a termination admitted more than one replacement", started, ended)
	}
	// Under a hard stop a session in flight at the bound is MEANT to end un-terminated,
	// so started == terminated is not a law here — only the ordering is.
	if started, ended := driver.SessionsStarted(), driver.SessionsTerminated(); started < ended || ended == 0 {
		t.Errorf("started=%d terminated=%d, want started >= terminated > 0", started, ended)
	}
}

// TestObserveCmd_DurationFlagRegistered pins the flag onto `blis observe` (sibling
// of the --concurrent-sessions/--total-sessions registration assertions): a bound
// the driver honours is useless if no flag reaches it.
func TestObserveCmd_DurationFlagRegistered(t *testing.T) {
	f := observeCmd.Flags().Lookup("duration")
	if f == nil {
		t.Fatal("--duration is not registered on `blis observe`")
	}
	if f.DefValue != "0s" {
		t.Errorf("--duration default = %q, want \"0s\" (unset = no clock bound)", f.DefValue)
	}
	for _, want := range []string{"--concurrent-sessions", "--total-sessions"} {
		if !strings.Contains(f.Usage, want) {
			t.Errorf("--duration help does not mention %q; its constraints must be discoverable", want)
		}
	}
}

// TestReplayCmd_DurationFlagNotRegistered pins the agreed scope: the clock bound is
// observe-only for now. `blis replay` keeps --total-sessions and its simulated-clock
// --horizon, so a stray --duration there must be an unknown flag rather than a
// silently ignored one.
func TestReplayCmd_DurationFlagNotRegistered(t *testing.T) {
	if f := replayCmd.Flags().Lookup("duration"); f != nil {
		t.Error("--duration is registered on `blis replay`, but the clock bound is observe-only")
	}
}

// TestValidateObserveCorpusFlags_ExplicitZeroTotalSessionsConflicts pins that the
// one-of is decided by whether the operator SUPPLIED --total-sessions, not by its
// value. 0 is a documented, meaningful value ("replay each corpus session once"), so
// supplying it alongside --duration is a genuine conflict; treating it as "unset"
// silently discards the stated intent and cycles the corpus with clones instead.
func TestValidateObserveCorpusFlags_ExplicitZeroTotalSessionsConflicts(t *testing.T) {
	// Supplied as 0 → conflict, same as any other supplied value.
	if msg := validateObserveCorpusFlags(4, 0, true, "h.yaml", "d.csv", "", "", false, 0, 0, "", false, false, false, false, 20*time.Minute); msg == "" {
		t.Error("--duration with an explicitly supplied --total-sessions 0 was accepted; the one-of must be decided by supplied-ness, not by value")
	}
	// Not supplied → no conflict.
	if msg := validateObserveCorpusFlags(4, 0, false, "h.yaml", "d.csv", "", "", false, 0, 0, "", false, false, false, false, 20*time.Minute); msg != "" {
		t.Errorf("--duration alone must be valid, got %q", msg)
	}
}

// TestObserveCorpusMode_DurationSuppressesLaterRounds pins the half that separates a
// hard stop from the admission-only bound: a session mid-conversation must not get
// its remaining rounds sent, so the trace holds partial sessions.
func TestObserveCorpusMode_DurationSuppressesLaterRounds(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hi"}},
			"usage":   map[string]interface{}{"prompt_tokens": 10, "completion_tokens": 2},
		})
	}))
	defer server.Close()

	// One 4-round session with 5s of think time between rounds: round 0 goes out at
	// once, and the bound elapses long before round 1 would be due.
	headerPath, dataPath := longThinkCorpus(t)
	driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, 1, 0, false, 42, true)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	recorder, stopped := runCorpusPoolRecorded(t, server.URL, driver, initial, 1, 200*time.Millisecond)
	if !stopped {
		t.Error("expected the bound to stop the run")
	}
	if len(recorder.Records()) == 0 {
		t.Fatal("nothing was sent before the bound; the round assertion below would be vacuous")
	}
	maxRound := -1
	for _, r := range recorder.Records() {
		if r.RoundIndex > maxRound {
			maxRound = r.RoundIndex
		}
	}
	if maxRound != 0 {
		t.Errorf("highest dispatched round = %d, want 0: later rounds must not be sent after the bound", maxRound)
	}
}

// TestObserveCorpusMode_DurationDrainsInFlightRequests pins that the bound stops
// SENDING without aborting what is already on the wire. The server takes far longer
// to answer than the bound, so the single in-flight request is still outstanding when
// the bound fires; it must be waited for and recorded, not cancelled.
func TestObserveCorpusMode_DurationDrainsInFlightRequests(t *testing.T) {
	const serverDelay = 400 * time.Millisecond
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(serverDelay)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hi"}},
			"usage":   map[string]interface{}{"prompt_tokens": 10, "completion_tokens": 2},
		})
	}))
	defer server.Close()

	headerPath, dataPath := multiRoundCorpus(t)
	driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, 1, 0, false, 42, true)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	start := time.Now()
	recorder, _ := runCorpusPoolRecorded(t, server.URL, driver, initial, 1, 50*time.Millisecond)
	elapsed := time.Since(start)

	if len(recorder.Records()) != 1 {
		t.Fatalf("recorded %d requests, want 1 (the in-flight request must be drained, not dropped)", len(recorder.Records()))
	}
	if got := recorder.Records()[0].Status; got != "ok" {
		t.Errorf("in-flight request status = %q, want \"ok\": the bound must not abort a request already on the wire", got)
	}
	if elapsed < serverDelay {
		t.Errorf("run finished in %v, before the server could answer (%v) — the request was aborted rather than drained", elapsed, serverDelay)
	}
}

// multiRoundCorpus writes a 2-session corpus of 2-round sessions with no think time,
// so follow-up rounds are due immediately and a bound has to actively suppress them.
func multiRoundCorpus(t *testing.T) (headerPath, dataPath string) {
	t.Helper()
	dir := t.TempDir()
	headerPath = filepath.Join(dir, "c.yaml")
	dataPath = filepath.Join(dir, "c.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
	recs := []workload.TraceRecord{
		{RequestID: 0, SessionID: "s0", RoundIndex: 0, InputTokens: 20, OutputTokens: 4, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s0", RoundIndex: 1, InputTokens: 10, OutputTokens: 4, ArrivalTimeUs: 1000, Status: "ok"},
		{RequestID: 2, SessionID: "s1", RoundIndex: 0, InputTokens: 20, OutputTokens: 4, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 3, SessionID: "s1", RoundIndex: 1, InputTokens: 10, OutputTokens: 4, ArrivalTimeUs: 1000, Status: "ok"},
	}
	if err := workload.ExportTraceV2(header, recs, headerPath, dataPath); err != nil {
		t.Fatalf("export: %v", err)
	}
	return headerPath, dataPath
}

// longThinkCorpus writes ONE 4-round session whose rounds are 5s apart, so round 0
// dispatches at once and every later round is still pending when a short bound fires.
func longThinkCorpus(t *testing.T) (headerPath, dataPath string) {
	t.Helper()
	dir := t.TempDir()
	headerPath = filepath.Join(dir, "c.yaml")
	dataPath = filepath.Join(dir, "c.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
	var recs []workload.TraceRecord
	for i := 0; i < 4; i++ {
		recs = append(recs, workload.TraceRecord{
			RequestID: i, SessionID: "s0", RoundIndex: i,
			InputTokens: 20, OutputTokens: 4,
			ArrivalTimeUs: int64(i) * 5_000_000,
			Status:        "ok",
		})
	}
	if err := workload.ExportTraceV2(header, recs, headerPath, dataPath); err != nil {
		t.Fatalf("export: %v", err)
	}
	return headerPath, dataPath
}

// manyRoundCorpus writes a corpus of `sessions` sessions with `rounds` rounds each,
// so sessions keep producing follow-ups and the pool stays busy.
func manyRoundCorpus(t *testing.T, sessions, rounds int) (headerPath, dataPath string) {
	t.Helper()
	dir := t.TempDir()
	headerPath = filepath.Join(dir, "c.yaml")
	dataPath = filepath.Join(dir, "c.csv")
	header := &workload.TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated"}
	var recs []workload.TraceRecord
	id := 0
	for s := 0; s < sessions; s++ {
		for r := 0; r < rounds; r++ {
			recs = append(recs, workload.TraceRecord{
				RequestID: id, SessionID: fmt.Sprintf("s%d", s), RoundIndex: r,
				InputTokens: 20, OutputTokens: 3, ArrivalTimeUs: int64(r) * 1000, Status: "ok",
			})
			id++
		}
	}
	if err := workload.ExportTraceV2(header, recs, headerPath, dataPath); err != nil {
		t.Fatalf("export: %v", err)
	}
	return headerPath, dataPath
}

// TestObserveOrchestrator_DrainTerminatesForAnyConcurrency is a deadlock regression
// guard. Once the bound stops the loop, nothing reads the follow-up channel any more,
// but completions still arriving push onto it — one entry per completing session. If
// those pushes can fill the channel, the session serializer blocks on a push, stops
// reading completions, and the shutdown's wait for the serializer never returns.
//
// The channel is sized by --max-concurrency while the pushes are sized by
// --concurrent-sessions, so the two are only safe together because the CLI auto-raises
// the former to the latter — a guard 300 lines from the drain. This pins the drain
// itself, so the orchestrator is correct for ANY pair rather than only the pair the
// CLI happens to produce.
func TestObserveOrchestrator_DrainTerminatesForAnyConcurrency(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hi"}},
			"usage":   map[string]interface{}{"prompt_tokens": 10, "completion_tokens": 2},
		})
	}))
	defer server.Close()

	headerPath, dataPath := manyRoundCorpus(t, 4, 6)

	for _, tc := range []struct{ pool, maxConcur int }{
		{8, 1},  // channel far smaller than the pool
		{16, 2}, // same, wider
		{8, 8},  // the shape the CLI produces (control: must also pass)
	} {
		t.Run(fmt.Sprintf("pool%d_maxconcur%d", tc.pool, tc.maxConcur), func(t *testing.T) {
			driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, tc.pool, 0, false, 42, true)
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			client := NewRealClient(server.URL, "", "test-model", "vllm")
			rec := &Recorder{}
			done := make(chan struct{})
			go func() {
				defer close(done)
				runObserveOrchestrator(context.Background(), client, rec, driver,
					cluster.NewSliceRequestSource(initial), true, tc.maxConcur, 0, nil, nil, false, false, 1.0,
					60*time.Millisecond)
			}()
			select {
			case <-done:
			case <-time.After(15 * time.Second):
				// Fatal, not Error: on a hang the orchestrator goroutine is still live and
				// would outlive this subtest's server.
				t.Fatalf("orchestrator never returned with --concurrent-sessions=%d and --max-concurrency=%d: "+
					"the drain deadlocked on the unread follow-up channel", tc.pool, tc.maxConcur)
			}
			if len(rec.Records()) == 0 {
				t.Error("no requests were dispatched, so returning proves nothing about the drain")
			}
		})
	}
}

// TestObserveOrchestrator_NoRequestSentAtOrAfterTheBound covers the CONCURRENCY-SLOT
// window: with one slot and a server slower than the bound, a second candidate is still
// blocked on the slot when the bound fires. That acquire cannot observe the bound timer,
// so without the final pre-dispatch gate the candidate goes out ~1.5x the bound late.
//
// Exactly one request is dispatched here, so this does NOT cover a due follow-up racing
// the timer — TestObserveOrchestrator_DueFollowUpNotSentAfterBound does that.
func TestObserveOrchestrator_NoRequestSentAtOrAfterTheBound(t *testing.T) {
	const bound = 100 * time.Millisecond
	const serverDelay = 150 * time.Millisecond

	var mu sync.Mutex
	var lastArrival time.Duration
	var hits int
	origin := time.Now()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		hits++
		if d := time.Since(origin); d > lastArrival {
			lastArrival = d
		}
		mu.Unlock()
		time.Sleep(serverDelay)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hi"}},
			"usage":   map[string]interface{}{"prompt_tokens": 10, "completion_tokens": 2},
		})
	}))
	defer server.Close()

	headerPath, dataPath := manyRoundCorpus(t, 6, 8)
	driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, 6, 0, false, 42, true)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	client := NewRealClient(server.URL, "", "test-model", "vllm")
	rec := &Recorder{}
	origin = time.Now()
	// maxConcurrency 1 forces every dispatch after the first to wait for the slot,
	// which is the window the bound must also close.
	runObserveOrchestrator(context.Background(), client, rec, driver,
		cluster.NewSliceRequestSource(initial), true, 1, 0, nil, nil, false, false, 1.0, bound)

	mu.Lock()
	gotHits, gotLast := hits, lastArrival
	mu.Unlock()

	if gotHits == 0 {
		t.Fatal("no requests were sent; the assertion would be vacuous")
	}
	// Server-side receipt trails the send slightly; allow transport slop but nothing
	// like a further round trip (which is what a leaked dispatch would cost).
	const margin = 40 * time.Millisecond
	if gotLast > bound+margin {
		t.Errorf("last request reached the server at %v, past the %v bound (+%v transport margin): "+
			"a request was dispatched after sending should have stopped",
			gotLast.Round(time.Millisecond), bound, margin)
	}
}

// TestCountSessionsWithRecords underpins the end-of-run split between sessions that
// were truncated mid-conversation (at least one round recorded) and sessions the pool
// admitted but the bound stopped before anything was sent. Reporting them together
// mislabels an empty session as "recorded as far as it got".
func TestCountSessionsWithRecords(t *testing.T) {
	cases := []struct {
		name string
		ids  []string
		want int
	}{
		{"none", nil, 0},
		{"distinct", []string{"a", "b", "c"}, 3},
		{"repeats count once", []string{"a", "a", "b", "a"}, 2},
		{"non-session rows ignored", []string{"a", "", "b", ""}, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			recs := make([]workload.TraceRecord, 0, len(tc.ids))
			for _, id := range tc.ids {
				recs = append(recs, workload.TraceRecord{SessionID: id})
			}
			if got := countSessionsWithRecords(recs); got != tc.want {
				t.Errorf("countSessionsWithRecords = %d, want %d", got, tc.want)
			}
		})
	}
}

// TestObserveOrchestrator_DrainDoesNotAdmitPhantomSessions pins that the session counts
// describe the run rather than the pool's bookkeeping. Once sending stops, a session
// completing during the drain must not cause another to be admitted: that replacement
// could never be dispatched, and counting it would inflate "sessions started" by the
// whole pool size (single-round sessions are the worst case, one phantom per slot).
func TestObserveOrchestrator_DrainDoesNotAdmitPhantomSessions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(120 * time.Millisecond) // slower than the bound, so completions land during the drain
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hi"}},
			"usage":   map[string]interface{}{"prompt_tokens": 10, "completion_tokens": 2},
		})
	}))
	defer server.Close()

	headerPath, dataPath := manyRoundCorpus(t, 4, 1) // ONE round each: every completion terminates
	const pool = 16
	driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, pool, 0, false, 42, true)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	client := NewRealClient(server.URL, "", "test-model", "vllm")
	rec := &Recorder{}
	runObserveOrchestrator(context.Background(), client, rec, driver,
		cluster.NewSliceRequestSource(initial), true, pool, 0, nil, nil, false, false, 1.0,
		60*time.Millisecond)

	started := driver.SessionsStarted()
	dispatched := countSessionsWithRecords(rec.Records())
	if dispatched == 0 {
		t.Fatal("no sessions were dispatched; the assertion would be vacuous")
	}
	// A refill admitted just before the bound may legitimately go undispatched, so allow
	// a small slack — but nothing like one phantom per pool slot.
	if started > dispatched+2 {
		t.Errorf("SessionsStarted()=%d but only %d sessions were dispatched: the drain admitted "+
			"%d sessions that could never be sent", started, dispatched, started-dispatched)
	}
}

// TestObserveOrchestrator_DueFollowUpNotSentAfterBound covers the half of the hard stop
// that distinguishes it from the superseded "stop admitting new sessions": a follow-up
// round that is DUE when the bound fires must not be sent.
//
// This needs a corpus with no think time, so follow-ups are continuously due and the
// bound has to actively reject them. TestObserveCorpusMode_DurationSuppressesLaterRounds
// uses a 5s think time, so its round 1 is never due and suppression never rejects
// anything; and TestObserveOrchestrator_NoRequestSentAtOrAfterTheBound dispatches a
// single round-0 request. Neither reaches the bound-vs-follow-up select race.
func TestObserveOrchestrator_DueFollowUpNotSentAfterBound(t *testing.T) {
	const bound = 120 * time.Millisecond
	const margin = 40 * time.Millisecond

	// Absolute receipt times, reduced to a span from the first hit below. Deriving them
	// against a wall-clock origin the test assigns after the server starts would be a
	// data race with the handler goroutines.
	var mu sync.Mutex
	var hits []time.Time
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		hits = append(hits, time.Now())
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hi"}},
			"usage":   map[string]interface{}{"prompt_tokens": 10, "completion_tokens": 2},
		})
	}))
	defer server.Close()

	headerPath, dataPath := multiRoundCorpus(t) // 2 rounds/session, ~1ms apart: always due
	driver, initial, err := buildObserveCorpusPool(headerPath, dataPath, 4, 0, false, 42, true)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	client := NewRealClient(server.URL, "", "test-model", "vllm")
	rec := &Recorder{}
	done := make(chan struct{})
	go func() {
		defer close(done)
		runObserveOrchestrator(context.Background(), client, rec, driver,
			cluster.NewSliceRequestSource(initial), true, 4, 0, nil, nil, false, false, 1.0, bound)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatalf("orchestrator did not stop within 10s for a %v bound — the bound is not ending the loop", bound)
	}

	mu.Lock()
	got := append([]time.Time{}, hits...)
	mu.Unlock()

	// Non-vacuity: follow-up rounds must actually have been flowing, or the timing
	// assertion says nothing about follow-up suppression.
	followUps := 0
	for _, r := range rec.Records() {
		if r.RoundIndex >= 1 {
			followUps++
		}
	}
	if followUps == 0 {
		t.Fatal("no follow-up round was ever dispatched; this test would not be exercising suppression")
	}
	first := got[0]
	for i, at := range got {
		if d := at.Sub(first); d > bound+margin {
			t.Errorf("request %d reached the server %v after the first, past the %v bound (+%v margin) "+
				"— a due follow-up was sent after sending should have stopped",
				i, d.Round(time.Millisecond), bound, margin)
		}
	}
	t.Logf("dispatched %d requests, %d of them follow-up rounds, all within the bound", len(got), followUps)
}

// withObserveFlagsRestored snapshots every observe flag var this package's corpus tests
// set and restores it afterwards, so driving runObserve cannot leak state into other
// tests in the package.
func withObserveFlagsRestored(t *testing.T) {
	t.Helper()
	type snap struct {
		serverURL, model, traceHeader, traceData string
		corpusHeader, corpusData                 string
		workload, workloadSpec, apiFormat        string
		serverType, thinkTimeDist                string
		concurrentSessions, totalSessions        int
		concurrency, numRequests, maxConcur      int
		warmup, timeout, thinkTimeMs             int
		seed                                     int64
		duration, prewarm                        time.Duration
		shuffle, lazy, scrapeKV, recordITL       bool
		noStreaming                              bool
	}
	s := snap{observeServerURL, observeModel, observeTraceHeader, observeTraceData,
		observeCorpusHeader, observeCorpusData, observeWorkload, observeWorkloadSpec,
		observeAPIFormat, observeServerType, observeThinkTimeDist,
		observeConcurrentSessions, observeTotalSessions, observeConcurrency,
		observeNumRequests, observeMaxConcur, observeWarmup, observeTimeout,
		observeThinkTimeMs, observeSeed, observeDuration, observePrewarmDuration,
		observeShuffleCorpus, observeLazyGeneration, observeScrapeKVMetrics,
		observeRecordITL, observeNoStreaming}
	t.Cleanup(func() {
		observeServerURL, observeModel = s.serverURL, s.model
		observeTraceHeader, observeTraceData = s.traceHeader, s.traceData
		observeCorpusHeader, observeCorpusData = s.corpusHeader, s.corpusData
		observeWorkload, observeWorkloadSpec = s.workload, s.workloadSpec
		observeAPIFormat, observeServerType = s.apiFormat, s.serverType
		observeThinkTimeDist = s.thinkTimeDist
		observeConcurrentSessions, observeTotalSessions = s.concurrentSessions, s.totalSessions
		observeConcurrency, observeNumRequests = s.concurrency, s.numRequests
		observeMaxConcur, observeWarmup, observeTimeout = s.maxConcur, s.warmup, s.timeout
		observeThinkTimeMs, observeSeed = s.thinkTimeMs, s.seed
		observeDuration, observePrewarmDuration = s.duration, s.prewarm
		observeShuffleCorpus, observeLazyGeneration = s.shuffle, s.lazy
		observeScrapeKVMetrics, observeRecordITL = s.scrapeKV, s.recordITL
		observeNoStreaming = s.noStreaming
	})
}

// TestObserveCmd_DurationReachesTheDispatchLoop closes the wiring gap: every other
// duration test calls runObserveOrchestrator directly and passes the bound by hand, so
// `--duration` could be disconnected from the loop — accepted, advertised, and ignored —
// with the whole suite still green. Verified by mutation: replacing the bound argument at
// the runObserve call site with 0 makes this hang, and passing a count-bounded pool
// instead makes the session-count assertion fail.
func TestObserveCmd_DurationReachesTheDispatchLoop(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hello there"}},
			"usage":   map[string]interface{}{"prompt_tokens": 20, "completion_tokens": 4},
		})
	}))
	defer server.Close()

	corpusH, corpusD := manyRoundCorpus(t, 3, 4)
	dir := t.TempDir()
	outH, outD := filepath.Join(dir, "observed.yaml"), filepath.Join(dir, "observed.csv")

	withObserveFlagsRestored(t)
	observeServerURL, observeModel = server.URL, "test-model"
	observeTraceHeader, observeTraceData = outH, outD
	observeCorpusHeader, observeCorpusData = corpusH, corpusD
	observeConcurrentSessions, observeTotalSessions = 3, 0
	observeDuration = 200 * time.Millisecond
	observeWorkload, observeWorkloadSpec, observeThinkTimeDist = "", "", ""
	observeConcurrency, observeNumRequests, observeThinkTimeMs = 0, 0, 0
	observeShuffleCorpus, observeLazyGeneration, observeScrapeKVMetrics = false, false, false
	observeRecordITL, observeNoStreaming = false, true
	observeMaxConcur, observeWarmup, observeTimeout = 8, 0, 300
	observeAPIFormat, observeServerType, observeSeed = "completions", "vllm", 42
	observePrewarmDuration = 0

	cmd := &cobra.Command{}
	for _, n := range []string{"total-sessions", "num-requests", "max-concurrency"} {
		cmd.Flags().Int(n, 0, "")
	}
	cmd.Flags().Float64("rate", 0, "")
	cmd.Flags().Int64("horizon", 0, "")
	cmd.Flags().Int64("seed", 42, "")
	cmd.Flags().Int("think-time-ms", 0, "")
	cmd.Flags().String("think-time-dist", "", "")

	done := make(chan struct{})
	go func() { defer close(done); runObserve(cmd, nil) }()
	select {
	case <-done:
	case <-time.After(15 * time.Second):
		t.Fatalf("runObserve did not return for a %v bound — --duration is not reaching the dispatch loop", observeDuration)
	}

	trace, err := workload.LoadTraceV2(outH, outD)
	if err != nil {
		t.Fatalf("exported trace does not load: %v", err)
	}
	sessions := map[string]bool{}
	for _, r := range trace.Records {
		if r.SessionID != "" {
			sessions[r.SessionID] = true
		}
	}
	if len(sessions) <= 3 {
		t.Errorf("exported trace holds %d sessions from a 3-session corpus; the pool was not "+
			"open-ended, so --duration did not size the run", len(sessions))
	}
	// The operator-visible consequence the docs lead with: partial sessions, still a
	// well-formed corpus.
	if _, _, err := buildObserveCorpusPool(outH, outD, 2, 0, false, 1, true); err != nil {
		t.Errorf("exported trace does not reload as a corpus: %v", err)
	}
}

package cmd

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// countSessionsWithRecords counts distinct sessions with at least one recorded request.
// The pool admits a session before anything is dispatched, so "started" alone cannot
// tell a session truncated mid-conversation from one the bound stopped before its first
// round went out — only the former was "recorded as far as it got".
func countSessionsWithRecords(records []workload.TraceRecord) int {
	seen := make(map[string]struct{}, len(records))
	for _, r := range records {
		if r.SessionID != "" {
			seen[r.SessionID] = struct{}{}
		}
	}
	return len(seen)
}

// validateObserveCorpusFlags enforces the spec-mode vs corpus-mode split for
// blis observe. Corpus-mode is selected by --concurrent-sessions > 0 (or a
// corpus file). It is mutually exclusive with every spec-mode input, because a
// corpus IS the workload — there is nothing to generate. Returns "" when the
// flag combination is valid, else a human-readable error for logrus.Fatalf.
func validateObserveCorpusFlags(
	concurrentSessions, totalSessions int,
	// totalSessionsSupplied distinguishes an unset --total-sessions from an explicit
	// 0, which is a documented, meaningful value ("replay each corpus session once").
	// The one-of against --duration turns on supplied-ness, not on the value, so
	// supplying 0 alongside --duration is a conflict rather than a silent override.
	totalSessionsSupplied bool,
	corpusHeader, corpusData string,
	workload, workloadSpec string,
	rateChanged bool,
	concurrency int,
	thinkTimeMs int,
	thinkTimeDist string,
	lazyGeneration bool,
	horizonChanged bool,
	numRequestsChanged bool,
	shuffleCorpus bool,
	duration time.Duration,
) string {
	corpusFilesSet := corpusHeader != "" || corpusData != ""
	corpusMode := concurrentSessions > 0

	// --total-sessions is meaningless without a pool.
	if totalSessions != 0 && !corpusMode {
		return "--total-sessions requires --concurrent-sessions > 0"
	}
	// --duration bounds a corpus run by stopping the dispatch loop. Spec-mode has no
	// pool; its generation is bounded by --num-requests / --horizon instead.
	if duration != 0 && !corpusMode {
		return "--duration requires --concurrent-sessions > 0"
	}
	// Corpus files supplied but pool not enabled.
	if corpusFilesSet && !corpusMode {
		return "--corpus-header/--corpus-data require --concurrent-sessions > 0"
	}
	// --shuffle-corpus only reorders a corpus pool; it is meaningless without one
	// (mirrors `blis replay --shuffle-corpus`). Never a silent no-op (R1).
	if shuffleCorpus && !corpusMode {
		return "--shuffle-corpus requires --concurrent-sessions > 0"
	}
	if !corpusMode {
		return "" // spec-mode: no corpus constraints apply
	}

	// Corpus-mode: both files required.
	if corpusHeader == "" || corpusData == "" {
		return "corpus-mode (--concurrent-sessions > 0) requires both --corpus-header and --corpus-data"
	}
	// A run is sized EITHER by session count or by the clock, never both: the count
	// says how many sessions to run, the duration says how long to keep starting
	// them, and honouring both would leave the end reason ambiguous. Reject rather
	// than silently letting one win (R1).
	if duration != 0 && totalSessionsSupplied {
		return "--duration and --total-sessions are mutually exclusive (corpus-mode): --total-sessions bounds the run by session count, --duration bounds it by the clock (after that long, stop sending entirely \u2014 no new sessions and no further rounds \u2014 then wait for the requests already on the wire). Pick one."
	}
	// A negative bound is meaningless; name the sign rather than the resolution.
	if duration < 0 {
		return fmt.Sprintf("--duration must be positive, got %s", duration)
	}
	// Corpus-mode: reject every spec-mode input.
	if concurrency > 0 {
		return "--concurrency is invalid with --concurrent-sessions (spec-mode vs corpus-mode): a corpus IS the workload; use --concurrent-sessions alone"
	}
	if workload != "" {
		return "--workload is invalid with --concurrent-sessions (spec-mode vs corpus-mode)"
	}
	if workloadSpec != "" {
		return "--workload-spec is invalid with --concurrent-sessions (spec-mode vs corpus-mode)"
	}
	if rateChanged {
		return "--rate is invalid with --concurrent-sessions (spec-mode vs corpus-mode)"
	}
	// Corpus-mode drives per-round think time from the corpus itself (recorded
	// think_time_us / arrival gaps via the session machinery), and loads the corpus
	// directly rather than streaming from a spec generator. So the spec-mode think /
	// generation knobs have no effect here — reject them loudly rather than silently
	// ignore (R1, no silent no-op).
	if thinkTimeMs > 0 {
		return "--think-time-ms is invalid with --concurrent-sessions (corpus-mode): per-round think time comes from the corpus, not this flag"
	}
	if thinkTimeDist != "" {
		return "--think-time-dist is invalid with --concurrent-sessions (corpus-mode): per-round think time comes from the corpus, not this flag"
	}
	if lazyGeneration {
		return "--lazy-generation is invalid with --concurrent-sessions (corpus-mode): the corpus is loaded directly; there is no spec generator to stream"
	}
	// --horizon / --num-requests bound spec-mode GENERATION, and corpus-mode generates
	// nothing — the corpus is read from a file. So neither has any effect here; reject
	// them loudly rather than silently ignore (R1, no silent no-op). The refusals name
	// the two flags that DO size a corpus run, since an operator reaching for --horizon
	// wants a run length and would otherwise be told only what does not work.
	if horizonChanged {
		return "--horizon is invalid with --concurrent-sessions (corpus-mode): it bounds generated arrivals, and a corpus is read from a file. Size the run with --total-sessions (session count) or --duration (clock)"
	}
	if numRequestsChanged {
		return "--num-requests is invalid with --concurrent-sessions (corpus-mode): the request count follows from the corpus and how the run is sized — use --total-sessions (session count) or --duration (clock)"
	}
	return ""
}

// buildObserveCorpusPool loads a TraceV2 corpus and constructs the session pool that
// drives corpus-mode observe. The blueprint loader always gets an unbounded horizon
// (math.MaxInt64): the run is bounded at the CLI, not in the loader — by session count
// (the pool self-drains all `total`, mirroring replay's default) or, when openEnded is
// set, by --duration, which the dispatch loop enforces by stopping sending.
func buildObserveCorpusPool(
	corpusHeader, corpusData string,
	concurrentSessions, totalSessions int,
	shuffleCorpus bool,
	seed int64,
	openEnded bool,
) (*workload.SessionPoolDriver, []*sim.Request, error) {
	trace, err := workload.LoadTraceV2(corpusHeader, corpusData)
	if err != nil {
		return nil, nil, fmt.Errorf("loading corpus: %w", err)
	}
	// nil thinkTimeSampler → derive think time from the corpus's inter-round
	// arrival gaps (same as replay's default closed-loop path).
	r0Requests, blueprints, err := workload.LoadTraceV2SessionBlueprints(trace, seed, nil, math.MaxInt64)
	if err != nil {
		return nil, nil, fmt.Errorf("building session blueprints: %w", err)
	}
	if len(blueprints) == 0 {
		return nil, nil, fmt.Errorf("corpus has no session records (need SessionID + RoundIndex rows)")
	}
	// Corpus-mode builds the pool from a 1:1 (session blueprint ↔ round-0 request)
	// corpus, so a trace that MIXES session records with non-session (single-shot,
	// empty session_id) records cannot be pooled. LoadTraceV2SessionBlueprints
	// returns one round-0 request per session PLUS one per non-session record, so a
	// mix diverges from the blueprint count. Surface that with an actionable message
	// — mirroring the same guard on `blis replay --concurrent-sessions` (R1) — rather
	// than letting BuildSessionPool fail with its internal "count mismatch" wording.
	if nonSession := len(r0Requests) - len(blueprints); nonSession > 0 {
		return nil, nil, fmt.Errorf("corpus-mode requires every record to belong to a session, but %d of %d records have no session_id; pooled observe cannot mix session and non-session (single-shot) records — re-export the corpus so every row carries a session_id (e.g. via `blis convert otel`)", nonSession, len(r0Requests))
	}
	// Optional seeded shuffle of the corpus step/admission order (#1480). Drawn from
	// the SAME salted stream (seed ^ corpusShuffleSeedSalt) as `blis replay
	// --shuffle-corpus`, so observe and replay of one corpus under the same --seed
	// select the identical subset/order — required for calibration parity.
	if shuffleCorpus {
		workload.ShuffleSessions(blueprints, r0Requests, rand.New(rand.NewSource(seed^corpusShuffleSeedSalt)))
	}
	var opts []workload.SessionPoolOption
	if openEnded {
		opts = append(opts, workload.WithOpenEndedQueue())
	}
	driver, initial, err := workload.BuildSessionPool(blueprints, r0Requests, concurrentSessions, totalSessions, seed, opts...)
	if err != nil {
		return nil, nil, fmt.Errorf("building session pool: %w", err)
	}
	return driver, initial, nil
}

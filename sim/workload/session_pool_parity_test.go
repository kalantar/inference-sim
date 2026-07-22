package workload

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// emitted captures the observable identity of a request as it is dispatched by a
// closed-loop pool driver: the generated ID plus the session/round it belongs to.
type emitted struct {
	ID         string
	SessionID  string
	RoundIndex int
}

// drivePoolSequence reproduces the exact loop that BOTH observe's orchestrator
// and blis replay's DES run against a SessionPoolDriver: inject the pool's
// initial round-0 requests, then on each completion call OnComplete and enqueue
// whatever it returns (an intra-session follow-up, or the next admitted
// session's round-0 request). Requests are drained FIFO so the recorded order is
// the driver's admission/follow-up order, independent of wall-clock timing.
//
// It is the single source of truth for the fixed-pool request sequence, so
// running it on identical (trace, seed, concurrent, total) inputs locks the
// observe and replay paths — which share BuildSessionPool — against drift.
func drivePoolSequence(t *testing.T, trace *TraceV2, seed int64, concurrent, total int) []emitted {
	t.Helper()

	r0Requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, seed, nil, math.MaxInt64)
	if err != nil {
		t.Fatalf("LoadTraceV2SessionBlueprints: %v", err)
	}
	driver, initial, err := BuildSessionPool(blueprints, r0Requests, concurrent, total, seed)
	if err != nil {
		t.Fatalf("BuildSessionPool: %v", err)
	}

	var seq []emitted
	queue := append([]*sim.Request(nil), initial...)
	const tick int64 = 1000
	// Guard against an accidental infinite loop if admission logic regresses.
	for steps := 0; len(queue) > 0; steps++ {
		if steps > 10000 {
			t.Fatalf("drivePoolSequence: exceeded 10000 steps — admission likely looping")
		}
		req := queue[0]
		queue = queue[1:]
		seq = append(seq, emitted{ID: req.ID, SessionID: req.SessionID, RoundIndex: req.RoundIndex})

		// Simulate a successful completion the way the orchestrator/DES does.
		req.State = sim.StateCompleted
		req.ProgressIndex = int64(req.InputLen()) + int64(len(req.OutputTokens))
		queue = append(queue, driver.OnComplete(req, tick)...)
	}
	return seq
}

// parityTrace builds a small in-memory TraceV2 corpus: session A (2 rounds),
// session B (1 round), session C (1 round), in first-seen order A, B, C.
func parityTrace() *TraceV2 {
	return &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 20, ArrivalTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 150, OutputTokens: 30, ArrivalTimeUs: 5000},
			{RequestID: 3, SessionID: "B", RoundIndex: 0, InputTokens: 120, OutputTokens: 25, ArrivalTimeUs: 1000},
			{RequestID: 4, SessionID: "C", RoundIndex: 0, InputTokens: 130, OutputTokens: 40, ArrivalTimeUs: 2000},
		},
	}
}

// TestSessionPoolParity_ObserveReplayShareSequence is the INV-13 guard: for a
// fixed trace + concurrent + total + seed, the fixed-pool request sequence is
// deterministic and identical across independent constructions. observe (via
// runObserveOrchestrator) and replay (via the DES) both drive the pool through
// this exact BuildSessionPool + OnComplete path, so a match here means their
// session admission, request IDs, and order agree by construction.
func TestSessionPoolParity_ObserveReplayShareSequence(t *testing.T) {
	trace := parityTrace()
	const seed int64 = 7

	// "observe" construction and "replay" construction are the same code path;
	// building the driver twice from identical inputs must yield an identical
	// sequence (INV-6 determinism + shared BuildSessionPool source).
	observeSeq := drivePoolSequence(t, trace, seed, 2, 0)
	replaySeq := drivePoolSequence(t, trace, seed, 2, 0)

	if len(observeSeq) != len(replaySeq) {
		t.Fatalf("sequence length mismatch: observe=%d replay=%d", len(observeSeq), len(replaySeq))
	}
	for i := range observeSeq {
		if observeSeq[i] != replaySeq[i] {
			t.Fatalf("sequence diverges at %d: observe=%+v replay=%+v", i, observeSeq[i], replaySeq[i])
		}
	}

	// Lock the concrete admission/follow-up order against drift. Corpus order is
	// A, B, C with pool=2: inject A_r0, B_r0; A_r0 → follow-up A_r1; B_r0
	// terminates → admit C_r0; A_r1 and C_r0 both terminate with the corpus
	// exhausted. FIFO draining yields A_r0, B_r0, A_r1, C_r0.
	wantSessionRounds := []struct {
		sid   string
		round int
	}{
		{"A", 0}, {"B", 0}, {"A", 1}, {"C", 0},
	}
	if len(observeSeq) != len(wantSessionRounds) {
		t.Fatalf("got %d dispatched requests, want %d: %+v", len(observeSeq), len(wantSessionRounds), observeSeq)
	}
	for i, want := range wantSessionRounds {
		if observeSeq[i].SessionID != want.sid || observeSeq[i].RoundIndex != want.round {
			t.Errorf("dispatch[%d] = (%s, r%d), want (%s, r%d)",
				i, observeSeq[i].SessionID, observeSeq[i].RoundIndex, want.sid, want.round)
		}
	}
}

// TestSessionPoolParity_DuplicationSequenceDeterministic locks the sequence when
// --total-sessions exceeds the corpus, so cache-busting duplication (seeded RNG)
// is exercised. Two constructions with the same seed must agree on IDs and order.
func TestSessionPoolParity_DuplicationSequenceDeterministic(t *testing.T) {
	trace := parityTrace()
	const seed int64 = 13

	// 3 corpus sessions, want 5 total (2 clones), pool of 2.
	a := drivePoolSequence(t, trace, seed, 2, 5)
	b := drivePoolSequence(t, trace, seed, 2, 5)

	if len(a) != len(b) {
		t.Fatalf("sequence length mismatch across identical seeds: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("duplication sequence not deterministic at %d: %+v vs %+v", i, a[i], b[i])
		}
	}

	// 5 sessions: A(2 rounds) + B + C + 2 clones (each 1 round for B/C-origin or
	// 2 for A-origin, depending on round-robin). At minimum every started session
	// contributes its round-0; assert all 5 distinct sessions were dispatched.
	seenSessions := map[string]bool{}
	for _, e := range a {
		if e.RoundIndex == 0 {
			seenSessions[e.SessionID] = true
		}
	}
	if len(seenSessions) != 5 {
		t.Errorf("expected 5 distinct sessions dispatched (round-0), got %d: %v", len(seenSessions), seenSessions)
	}
}

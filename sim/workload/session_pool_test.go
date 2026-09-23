package workload

import (
	"fmt"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestShuffleSessions_ReproducibleAndLockstep verifies #1480's shuffle contract:
// the permutation is reproducible from a fixed seed, keeps (blueprint, round-0)
// pairs aligned, drops no session, and generally differs across seeds.
func TestShuffleSessions_ReproducibleAndLockstep(t *testing.T) {
	build := func(n int) ([]SessionBlueprint, []*sim.Request) {
		bps := make([]SessionBlueprint, n)
		r0 := make([]*sim.Request, n)
		for i := 0; i < n; i++ {
			bp, req := makeBP(fmt.Sprintf("s%02d", i), int64(i))
			bps[i], r0[i] = bp, req
		}
		return bps, r0
	}
	order := func(bps []SessionBlueprint) string {
		s := ""
		for _, b := range bps {
			s += b.SessionID + ","
		}
		return s
	}

	// Reproducible: the same seed yields the same permutation.
	bpsA, r0A := build(20)
	ShuffleSessions(bpsA, r0A, rand.New(rand.NewSource(7)))
	bpsA2, r0A2 := build(20)
	ShuffleSessions(bpsA2, r0A2, rand.New(rand.NewSource(7)))
	_ = r0A2
	if order(bpsA) != order(bpsA2) {
		t.Errorf("same seed → different order:\n %q\n %q", order(bpsA), order(bpsA2))
	}
	// Lockstep: after the shuffle, blueprint[i] and r0[i] still name the same session.
	for i := range bpsA {
		if bpsA[i].SessionID != r0A[i].SessionID {
			t.Errorf("lockstep broken at %d: bp=%s r0=%s", i, bpsA[i].SessionID, r0A[i].SessionID)
		}
	}
	// Set preserved: every session still present (nothing dropped).
	seen := map[string]bool{}
	for _, b := range bpsA {
		seen[b.SessionID] = true
	}
	if len(seen) != 20 {
		t.Errorf("shuffle changed the session set: %d unique, want 20", len(seen))
	}
	// A different seed generally yields a different order (collision negligible at n=20).
	bpsB, r0B := build(20)
	ShuffleSessions(bpsB, r0B, rand.New(rand.NewSource(999)))
	if order(bpsA) == order(bpsB) {
		t.Errorf("different seeds → identical order (astronomically unlikely; likely a bug)")
	}
}

// drainPool completes every admitted session in admission order and returns the full
// sequence of admitted round-0 requests. Single-round blueprints terminate on their
// round-0 completion, so one OnComplete per admitted session drains the pool. maxDrain
// caps the loop, since an open-ended queue would otherwise never end.
func drainPool(d *SessionPoolDriver, initial []*sim.Request, maxDrain int) []*sim.Request {
	admitted := append([]*sim.Request{}, initial...)
	pending := append([]*sim.Request{}, initial...)
	for n := 0; len(pending) > 0 && n < maxDrain; n++ {
		r := pending[0]
		pending = pending[1:]
		r.State = sim.StateCompleted
		r.ProgressIndex = int64(r.InputLen())
		next := d.OnComplete(r, 1000)
		admitted = append(admitted, next...)
		pending = append(pending, next...)
	}
	return admitted
}

// sessionIDsOf maps admitted requests to their SessionIDs for order assertions.
func sessionIDsOf(reqs []*sim.Request) []string {
	ids := make([]string, 0, len(reqs))
	for _, r := range reqs {
		ids = append(ids, r.SessionID)
	}
	return ids
}

// makeBP builds a minimal 1-round blueprint + its round-0 request for tests.
func makeBP(id string, seed int64) (SessionBlueprint, *sim.Request) {
	bp := SessionBlueprint{
		SessionID:     id,
		MaxRounds:     1,
		Horizon:       1 << 62, // effectively unbounded — models the self-draining default (no --horizon cap)
		InputSampler:  &SequenceSampler{values: []int{}},
		OutputSampler: &SequenceSampler{values: []int{}},
	}
	req := &sim.Request{ID: "r_" + id, SessionID: id, RoundIndex: 0, State: sim.StateQueued, ArrivalTime: 0}
	return bp, req
}

func TestBuildSessionPool_DuplicatesToTarget(t *testing.T) {
	bp0, r0 := makeBP("s0", 1)
	bp1, r1 := makeBP("s1", 2)
	// Corpus of 2, want 5 total, pool of 2.
	d, initial, err := BuildSessionPool([]SessionBlueprint{bp0, bp1}, []*sim.Request{r0, r1}, 2, 5, 99)
	if err != nil {
		t.Fatalf("BuildSessionPool: %v", err)
	}
	if len(initial) != 2 {
		t.Fatalf("initial injected = %d, want 2 (pool size)", len(initial))
	}
	// 5 total sessions must run, with unique SessionIDs (clones renamed).
	if got := d.TotalSessions(); got != 5 {
		t.Errorf("total sessions = %d, want 5", got)
	}
	// Uniqueness is asserted over the sessions actually ADMITTED (draining the pool)
	// rather than over a pre-built queue: sessions are materialized on demand, so the
	// admitted sequence is the only place the full set is observable.
	admitted := drainPool(d, initial, 50)
	if len(admitted) != 5 {
		t.Fatalf("admitted %d sessions, want 5", len(admitted))
	}
	seen := make(map[string]struct{}, len(admitted))
	for _, r := range admitted {
		if _, dup := seen[r.SessionID]; dup {
			t.Errorf("duplicated sessions must have unique IDs; %q admitted twice", r.SessionID)
		}
		seen[r.SessionID] = struct{}{}
	}
}

func TestSessionPool_RefillRebasesDeadline(t *testing.T) {
	// Two single-round sessions, pool of 1, each with an ABSOLUTE deadline 5000µs
	// after its original arrival (t=0). Terminating session 0 far in the future must
	// refill session 1 with its deadline REBASED ahead of the admission tick —
	// otherwise the stale past deadline drops it instantly on enqueue, mass-cancelling
	// every wave-2+ session for any deadline-bearing (run/observe) corpus.
	bp0, r0 := makeBP("s0", 1)
	bp1, r1 := makeBP("s1", 2)
	r0.Deadline = 5000
	r1.Deadline = 5000
	d, initial, err := BuildSessionPool([]SessionBlueprint{bp0, bp1}, []*sim.Request{r0, r1}, 1, 2, 7)
	if err != nil {
		t.Fatalf("BuildSessionPool: %v", err)
	}
	if len(initial) != 1 || initial[0].SessionID != "s0" {
		t.Fatalf("initial = %v, want [s0]", initial)
	}
	const tick = 100000
	initial[0].State = sim.StateCompleted
	initial[0].ProgressIndex = int64(initial[0].InputLen())
	next := d.OnComplete(initial[0], tick)
	if len(next) != 1 || next[0].SessionID != "s1" {
		t.Fatalf("expected refill of s1, got %v", next)
	}
	refill := next[0]
	if refill.ArrivalTime != tick {
		t.Errorf("refill arrival = %d, want %d", refill.ArrivalTime, tick)
	}
	if refill.Deadline != tick+5000 {
		t.Errorf("refill deadline = %d, want %d (rebased by the arrival offset); a stale past deadline would mass-cancel wave 2+", refill.Deadline, tick+5000)
	}
}

func TestSessionPool_RefillAndConservation(t *testing.T) {
	// 4 single-round sessions, pool of 2. Each completion admits the next until the
	// corpus is exhausted; the pool never admits more than one replacement per
	// termination (so the concurrently-active count never exceeds the pool size);
	// exactly 4 sessions start and terminate. Asserted purely through observable
	// outputs — the injected slice, the per-completion follow-ups admitted, and
	// Unstarted()/TotalSessions() — so the test survives a rewrite of the driver's
	// internal counters (refactor-survival; principles.md BDD/TDD item 5).
	var bps []SessionBlueprint
	var r0s []*sim.Request
	for i := 0; i < 4; i++ {
		bp, r := makeBP(fmt.Sprintf("s%d", i), int64(i))
		bps = append(bps, bp)
		r0s = append(r0s, r)
	}
	d, initial, err := BuildSessionPool(bps, r0s, 2, 4, 7)
	if err != nil {
		t.Fatalf("BuildSessionPool: %v", err)
	}
	if len(initial) != 2 {
		t.Fatalf("initial injected = %d, want 2 (pool size)", len(initial))
	}
	if d.TotalSessions() != 4 {
		t.Fatalf("total sessions = %d, want 4", d.TotalSessions())
	}

	// started/terminated are counted only through observable outputs (returned
	// requests and OnComplete calls), never by reading a driver field.
	started := len(initial)
	terminated := 0

	// Complete the 2 initial sessions; each single-round session terminates on its
	// round-0 completion. Each termination must admit AT MOST ONE replacement —
	// len(next) <= 1 is the observable form of "active never exceeds the pool".
	var refills []*sim.Request
	for _, r := range initial {
		r.State = sim.StateCompleted
		r.ProgressIndex = int64(r.InputLen())
		next := d.OnComplete(r, 1000)
		terminated++
		if len(next) > 1 {
			t.Fatalf("a single termination admitted %d sessions; the pool bound requires <= 1", len(next))
		}
		started += len(next)
		refills = append(refills, next...)
	}
	if len(refills) != 2 {
		t.Fatalf("first wave admitted %d refills, want 2 (corpus not yet exhausted)", len(refills))
	}

	// Complete the 2 refills; the corpus is now exhausted → no further admissions.
	for _, r := range refills {
		r.State = sim.StateCompleted
		r.ProgressIndex = int64(r.InputLen())
		next := d.OnComplete(r, 2000)
		terminated++
		if len(next) != 0 {
			t.Fatalf("corpus exhausted but a termination admitted %d more", len(next))
		}
	}

	// Conservation, all via the public surface: every session started and
	// terminated exactly once, and none was left unstarted.
	if started != 4 || terminated != 4 {
		t.Errorf("started=%d terminated=%d, want 4/4", started, terminated)
	}
	if d.Unstarted() != 0 {
		t.Errorf("Unstarted() = %d, want 0 (every pooled session was admitted)", d.Unstarted())
	}
}

// TestBuildSessionPool_ClonesHaveIndependentSamplers is a regression test for
// the shared-cursor bug in cloneBlueprintForDup: a shallow `bp := src` struct
// copy leaves InputSampler/OutputSampler/ThinkTimeSampler pointing at the SAME
// underlying sampler object as the source for stateful sampler types like
// *SequenceSampler (which carries a mutable per-call cursor). Without cloning
// the sampler itself, a source session and its duplicate advance one shared
// cursor and corrupt each other's per-round token-length sequence.
//
// BuildSessionPool's SessionManager keeps blueprints in an unexported map, so
// there's no way to read back the clone's sampler through the public driver
// API. cloneBlueprintForDup is unexported but same-package, so this test calls
// it directly — the cleanest way to observe sampler independence: sample the
// source's InputSampler twice, then the clone's once, and assert the clone
// still returns the FIRST recorded value (10), not the third (30), which is
// what a shared cursor would produce.
func TestBuildSessionPool_ClonesHaveIndependentSamplers(t *testing.T) {
	src := SessionBlueprint{
		SessionID:         "s0",
		MaxRounds:         3,
		Horizon:           1 << 62,
		ContextGrowth:     "accumulate",
		InputSampler:      &SequenceSampler{values: []int{10, 20, 30}},
		OutputSampler:     &SequenceSampler{values: []int{1, 2, 3}},
		InputResetSampler: &SequenceSampler{values: []int{-1, 50, -1}}, // #1609: round-1 compaction reset
	}
	srcR0 := &sim.Request{ID: "r_s0", SessionID: "s0", RoundIndex: 0, State: sim.StateQueued}

	rng := rand.New(rand.NewSource(1))
	clone, _ := cloneBlueprintForDup(src, srcR0, 1, rng)

	// Pointer identity must differ: the clone must not alias the source's sampler.
	if clone.InputSampler == src.InputSampler {
		t.Fatalf("clone.InputSampler shares the same object as src.InputSampler")
	}
	// The #1609 reset sampler must likewise be cloned to an independent cursor —
	// otherwise a source session and its duplicate corrupt each other's per-round
	// compaction sequence.
	if clone.InputResetSampler == src.InputResetSampler {
		t.Fatalf("clone.InputResetSampler shares the same object as src.InputResetSampler")
	}
	// Advance the source's reset cursor one step (-1), then the clone must still
	// yield the FIRST value (-1), not the source's advanced position.
	_ = src.InputResetSampler.Sample(nil) // -1
	if got := clone.InputResetSampler.Sample(nil); got != -1 {
		t.Fatalf("clone InputResetSampler sample #1 = %d, want -1 (independent cursor)", got)
	}

	// Advance the source's cursor two steps: 10, then 20.
	if got := src.InputSampler.Sample(nil); got != 10 {
		t.Fatalf("source InputSampler sample #1 = %d, want 10", got)
	}
	if got := src.InputSampler.Sample(nil); got != 20 {
		t.Fatalf("source InputSampler sample #2 = %d, want 20", got)
	}

	// The clone's cursor must be independent: it should still yield the FIRST
	// value (10). A shared cursor (the bug) would instead yield 30, since the
	// source has already advanced past index 0 and 1.
	if got := clone.InputSampler.Sample(nil); got != 10 {
		t.Fatalf("clone InputSampler sample #1 = %d, want 10 (independent cursor) — got the source's advanced value, indicating a shared cursor", got)
	}
}

// TestSessionPool_CountBoundedAdmissionSequence_Golden pins the EXACT sequence a
// count-bounded pool admits: which sessions, in which order, and the cache-busting
// token each clone's round-0 input is prefixed with. The token values come from the
// seeded clone RNG, so this pins the whole seeded derivation — the clone order AND
// the number of draws taken per clone.
//
// Captured from the build that pre-dates on-demand session materialization. It is
// the unit-level guard that generating clones lazily (at admission) rather than
// eagerly (up front) does not perturb the seeded sequence: each clone still consumes
// exactly one draw from the shared stream, in clone order. Its CLI-level companion
// is TestReplayCmd_SessionPool_Deterministic.
func TestSessionPool_CountBoundedAdmissionSequence_Golden(t *testing.T) {
	bp0, r0 := makeBP("s0", 1)
	bp1, r1 := makeBP("s1", 2)
	d, initial, err := BuildSessionPool([]SessionBlueprint{bp0, bp1}, []*sim.Request{r0, r1}, 2, 5, 99)
	if err != nil {
		t.Fatalf("BuildSessionPool: %v", err)
	}
	admitted := drainPool(d, initial, 50)

	wantIDs := []string{"s0", "s1", "s0_dup1", "s1_dup2", "s0_dup3"}
	if got := sessionIDsOf(admitted); !reflect.DeepEqual(got, wantIDs) {
		t.Errorf("admission order = %v, want %v", got, wantIDs)
	}

	// Originals carry the corpus input verbatim (empty here); each clone is prefixed
	// with one seeded cache-busting token so its block-hash chain diverges at once.
	wantTokens := map[string][]sim.TokenID{
		"s0":      {},
		"s1":      {},
		"s0_dup1": {50452},
		"s1_dup2": {24822},
		"s0_dup3": {90142},
	}
	for _, r := range admitted {
		want, ok := wantTokens[r.SessionID]
		if !ok {
			t.Errorf("unexpected session %q admitted", r.SessionID)
			continue
		}
		if len(r.InputTokens) != len(want) {
			t.Errorf("%s: input tokens = %v, want %v", r.SessionID, r.InputTokens, want)
			continue
		}
		for i := range want {
			if r.InputTokens[i] != want[i] {
				t.Errorf("%s: input token[%d] = %d, want %d (seeded clone derivation changed)",
					r.SessionID, i, r.InputTokens[i], want[i])
			}
		}
	}
}

// TestSessionPool_QueueIsOpenEnded verifies the queue is no longer
// pre-sized under a clock bound: the corpus is cycled with fresh cache-busting
// clones on demand, so the queue never runs dry before the bound instead of draining
// after one pass over a small corpus.
func TestSessionPool_QueueIsOpenEnded(t *testing.T) {
	bp0, r0 := makeBP("s0", 1)
	bp1, r1 := makeBP("s1", 2)
	// Bound far beyond every completion tick, so admission is never suppressed and
	// the only thing that can stop the drain is the queue running dry.
	d, initial, err := BuildSessionPool([]SessionBlueprint{bp0, bp1}, []*sim.Request{r0, r1},
		2, 0, 99, WithOpenEndedQueue())
	if err != nil {
		t.Fatalf("BuildSessionPool: %v", err)
	}
	admitted := drainPool(d, initial, 25)
	if len(admitted) <= 2 {
		t.Fatalf("admitted %d sessions from a 2-session corpus; an open-ended queue must "+
			"keep cloning past the corpus rather than draining", len(admitted))
	}
	seen := map[string]bool{}
	for _, r := range admitted {
		if seen[r.SessionID] {
			t.Errorf("session %q admitted twice; each admission must be a distinct session", r.SessionID)
		}
		seen[r.SessionID] = true
	}
}

// TestBuildSessionPool_RejectsBothBounds pins that the two ways to size a run are
// alternatives, not an addition — honouring both would leave the end reason
// ambiguous, so the combination is refused rather than silently resolved (R1).
func TestBuildSessionPool_RejectsBothBounds(t *testing.T) {
	bp0, r0 := makeBP("s0", 1)
	_, _, err := BuildSessionPool([]SessionBlueprint{bp0}, []*sim.Request{r0},
		1, 5, 99, WithOpenEndedQueue())
	if err == nil {
		t.Fatal("expected an error when a session-count total and an admission deadline are both set")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("error = %q, want it to say the two bounds are mutually exclusive", err)
	}
}

// TestBuildSessionPool_RejectsMalformedCorpusUpFront pins that a blueprint the
// session machinery cannot drive is reported when the pool is BUILT, not when that
// session happens to be admitted — on-demand materialization would otherwise defer
// the failure into the middle of a run, or hide it entirely if the bound stopped
// admission first.
func TestBuildSessionPool_RejectsMalformedCorpusUpFront(t *testing.T) {
	good, goodR0 := makeBP("s0", 1)
	bad, badR0 := makeBP("s1", 2)
	bad.MaxRounds = 0 // undrivable
	_, _, err := BuildSessionPool([]SessionBlueprint{good, bad}, []*sim.Request{goodR0, badR0}, 1, 0, 99)
	if err == nil {
		t.Fatal("expected an error for a blueprint with MaxRounds=0")
	}
	if !strings.Contains(err.Error(), "s1") {
		t.Errorf("error = %q, want it to name the offending session s1", err)
	}
}

// TestSessionPool_ClonesAreBornPristine pins why the driver clones from snapshots
// taken at construction rather than from the live corpus requests. An original
// handed out as a refill is a live *sim.Request that the simulator mutates in place
// (State, ProgressIndex, FirstTokenTime). Cloning from it after it has run would
// produce a clone born already completed, which would then be injected as a new
// request. Verified by mutation: cloning from d.srcR0 instead of the snapshot makes
// this fail while every other pool test still passes, so nothing else covers it.
func TestSessionPool_ClonesAreBornPristine(t *testing.T) {
	bp0, r0 := makeBP("s0", 1)
	bp1, r1 := makeBP("s1", 2)
	// pool=1 so s1 is admitted as a refill (mutated in place) before being cloned.
	d, initial, err := BuildSessionPool([]SessionBlueprint{bp0, bp1}, []*sim.Request{r0, r1}, 1, 4, 99)
	if err != nil {
		t.Fatalf("BuildSessionPool: %v", err)
	}

	pending := append([]*sim.Request{}, initial...)
	sawClone := false
	for n := 1; len(pending) > 0 && n < 20; n++ {
		r := pending[0]
		pending = pending[1:]
		// Stand in for what the simulator/orchestrator does to a live request.
		r.State = sim.StateCompleted
		r.ProgressIndex = 999
		r.FirstTokenTime = 12345
		for _, born := range d.OnComplete(r, 100000) {
			if strings.Contains(born.SessionID, "_dup") {
				sawClone = true
				if born.State != sim.StateQueued {
					t.Errorf("clone %s born with State=%v, want queued", born.SessionID, born.State)
				}
				if born.ProgressIndex != 0 {
					t.Errorf("clone %s born with ProgressIndex=%d, want 0", born.SessionID, born.ProgressIndex)
				}
				if born.FirstTokenTime != 0 {
					t.Errorf("clone %s born with FirstTokenTime=%d, want 0", born.SessionID, born.FirstTokenTime)
				}
			}
			pending = append(pending, born)
		}
	}
	if !sawClone {
		t.Fatal("no clone was admitted; the assertion is vacuous")
	}
}

// TestSessionPool_OpenEndedMatchesCountBoundedPrefix pins INV-6 for the open-ended
// queue, which is the path --duration uses. The golden above covers only the
// count-bounded sequence; both go through admitNext, so the open-ended sequence must be
// a prefix-identical continuation of it. Without this, a change to how `limit` is read,
// to the srcIdx cycling, or to when clonesMade increments would silently re-seed every
// duration-bounded run and no test would move.
//
// It pairs with the golden rather than duplicating it: the golden is the ABSOLUTE anchor
// (it catches any change to the sequence itself), while this catches a divergence that
// affects ONLY the open-ended path, which the golden cannot see. Both mutations verified.
func TestSessionPool_OpenEndedMatchesCountBoundedPrefix(t *testing.T) {
	build := func(total int, opts ...SessionPoolOption) []*sim.Request {
		bp0, r0 := makeBP("s0", 1)
		bp1, r1 := makeBP("s1", 2)
		d, initial, err := BuildSessionPool([]SessionBlueprint{bp0, bp1}, []*sim.Request{r0, r1}, 2, total, 99, opts...)
		if err != nil {
			t.Fatalf("BuildSessionPool(total=%d): %v", total, err)
		}
		return drainPool(d, initial, 12)
	}

	counted := build(6)
	openEnded := build(0, WithOpenEndedQueue())
	if len(counted) != 6 {
		t.Fatalf("counted pool admitted %d, want 6", len(counted))
	}
	if len(openEnded) < 6 {
		t.Fatalf("open-ended pool admitted only %d; it should not run dry", len(openEnded))
	}

	// Same sessions, in the same order, with the same seeded cache-busting tokens.
	if got, want := sessionIDsOf(openEnded[:6]), sessionIDsOf(counted); !reflect.DeepEqual(got, want) {
		t.Errorf("open-ended prefix = %v, want the count-bounded sequence %v", got, want)
	}
	for i := range counted {
		a, b := counted[i].InputTokens, openEnded[i].InputTokens
		if !reflect.DeepEqual(a, b) {
			t.Errorf("session %d (%s): tokens %v under a count bound vs %v open-ended — the seeded clone stream diverged",
				i, counted[i].SessionID, a, b)
		}
	}

	// And the open-ended path is reproducible from the seed on its own.
	again := build(0, WithOpenEndedQueue())
	if got, want := sessionIDsOf(again), sessionIDsOf(openEnded); !reflect.DeepEqual(got, want) {
		t.Errorf("same seed → different open-ended sequence:\n %v\n %v", got, want)
	}
}

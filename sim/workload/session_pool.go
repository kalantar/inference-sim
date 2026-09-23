package workload

import (
	"fmt"
	"math/rand"
	"sync/atomic"

	"github.com/inference-sim/inference-sim/sim"
)

// SessionPoolDriver keeps a fixed number of closed-loop sessions active. It
// wraps a SessionManager: intra-session follow-ups are produced by the manager
// unchanged; when a session reaches a terminal state and the pool may admit
// another, the driver materializes the next session and injects its round-0
// request. This models a fixed pool of N concurrent "users" drawing from a corpus
// of captured sessions.
//
// The session sequence is VIRTUAL and materialized on demand: index i names the
// corpus's i-th session while the corpus lasts, and a cache-busting clone of corpus
// session i%len(corpus) after that. Either a session COUNT ends it (BuildSessionPool's
// total) or WithOpenEndedQueue leaves it unbounded, in which case the caller decides
// when to stop — the driver holds no clock.
//
// Determinism (INV-6): admission is in virtual-index order and each clone takes
// exactly one draw from the seeded clone stream, so materializing on demand yields the
// same sessions as materializing up front.
type SessionPoolDriver struct {
	mgr *SessionManager

	// srcBPs/srcR0 are the caller's (possibly shuffled) corpus, retained and read at
	// every later admission — so a caller must not reorder or overwrite them.
	//
	// Clones are made from cloneTemplates (pristine copies taken at construction), never
	// from srcR0: a corpus request handed out as a refill is live, and its consumer
	// mutates State/ProgressIndex/FirstTokenTime in place, so cloning from it after it
	// has run would produce a clone born already completed.
	srcBPs         []SessionBlueprint
	srcR0          []*sim.Request
	cloneTemplates []sim.Request
	cloneRNG       *rand.Rand
	clonesMade     int

	nextIndex int
	limit     int // sessions in the sequence; 0 = open-ended

	totalStarted    int // sessions injected so far (initial pool + refills)
	totalTerminated int

	// closed is the one field that crosses goroutines: OnComplete runs on its caller's
	// completion path while StopAdmitting is called from whatever goroutine decides the
	// run is over (for `blis observe`, the dispatch loop, not the session serializer).
	// Everything else here is single-threaded, so this is atomic rather than the whole
	// driver being locked.
	closed atomic.Bool
}

// SessionPoolOption configures a SessionPoolDriver at construction. Variadic so
// existing call sites are unaffected (R4).
type SessionPoolOption func(*sessionPoolSettings)

type sessionPoolSettings struct {
	openEnded bool
}

// WithOpenEndedQueue drops the session count: the queue keeps cycling the corpus with
// cache-busting clones and never runs dry, so the caller decides when the run ends.
// Mutually exclusive with a session-count total.
func WithOpenEndedQueue() SessionPoolOption {
	return func(s *sessionPoolSettings) { s.openEnded = true }
}

// cloneSampler returns a sampler independent of s for stateful sampler types.
// SequenceSampler carries a mutable per-call cursor (index) and replays a fixed
// recorded sequence, so each cloned session needs its own cursor over the same
// values — otherwise a source session and its duplicate corrupt each other's
// per-round sequence via the shared cursor. Stateless samplers (which derive all
// randomness from the *rand.Rand passed to Sample) are safe to share and are
// returned unchanged.
func cloneSampler(s LengthSampler) LengthSampler {
	if seq, ok := s.(*SequenceSampler); ok {
		// Fresh cursor (index=0); the values slice is immutable after construction, so sharing it is safe.
		return &SequenceSampler{values: seq.values}
	}
	return s
}

// cloneBlueprintForDup produces a duplicated blueprint + round-0 request with a
// new unique SessionID and a cache-busting prefix token so the clone does not
// share KV cache with its source. dupIdx is the global clone index (>=1).
func cloneBlueprintForDup(src SessionBlueprint, srcR0 *sim.Request, dupIdx int, rng *rand.Rand) (SessionBlueprint, *sim.Request) {
	newID := fmt.Sprintf("%s_dup%d", src.SessionID, dupIdx)
	bp := src // shallow copy; stateful samplers are independently re-cloned below via cloneSampler
	bp.SessionID = newID
	// Fresh per-session RNG so token IDs differ deterministically from the source.
	bp.RNG = rand.New(rand.NewSource(rng.Int63()))
	// Give the clone independent sampler state. Interface fields copied by the
	// shallow `bp := src` above still point at the SAME underlying sampler
	// object as src for stateful types (e.g. *SequenceSampler's mutable index
	// cursor) — without this, the source and its clone would corrupt each
	// other's per-round sequence by advancing a shared cursor.
	bp.InputSampler = cloneSampler(src.InputSampler)
	bp.OutputSampler = cloneSampler(src.OutputSampler)
	bp.ThinkTimeSampler = cloneSampler(src.ThinkTimeSampler)
	// #1609: the context-compaction reset sampler is a stateful *SequenceSampler in
	// lockstep with InputSampler, so each clone needs its own cursor (cloneSampler
	// passes nil through unchanged for monotone sessions).
	bp.InputResetSampler = cloneSampler(src.InputResetSampler)

	// Cache-busting token prepended to the clone's round-0 input so the block
	// hash chain diverges immediately from the source session (design §6).
	buster := sim.GenerateRandomTokenIDs(bp.RNG, 1)
	newInput := make([]sim.TokenID, 0, len(srcR0.InputTokens)+1)
	newInput = append(newInput, buster...)
	newInput = append(newInput, srcR0.InputTokens...)

	r0 := *srcR0 // shallow copy
	r0.ID = "r_" + newID
	r0.SessionID = newID
	r0.InputTokens = newInput
	return bp, &r0
}

// ShuffleSessions permutes the corpus's session order in place with rng, keeping
// the (blueprints, r0Requests) pair in lockstep (index i of each still names the
// same session). `blis replay --shuffle-corpus` (#1480) uses it to randomize the
// pool's step/admission order reproducibly from the master seed — every session
// still runs, only the order changes. The two slices are expected 1:1 (as
// BuildSessionPool asserts); the shuffle spans the matched prefix, so any trailing
// non-session requests would be left untouched.
func ShuffleSessions(blueprints []SessionBlueprint, r0Requests []*sim.Request, rng *rand.Rand) {
	n := len(blueprints)
	if len(r0Requests) < n {
		n = len(r0Requests)
	}
	// Fisher-Yates, applied identically to both slices so pairs stay aligned.
	for i := n - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		blueprints[i], blueprints[j] = blueprints[j], blueprints[i]
		r0Requests[i], r0Requests[j] = r0Requests[j], r0Requests[i]
	}
}

// BuildSessionPool prepares a pool over the corpus (blueprints + their round-0
// requests) and returns the driver plus the first concurrent round-0 requests to
// inject at simulation start. Sessions past the first concurrent ones are
// materialized on demand as the pool refills.
//
// concurrent: max concurrently-active sessions (>= 1).
// total: total sessions to replay (>= concurrent). If <= len(corpus), the corpus is
// truncated to total; if greater, clones fill the remainder. Pass 0 with
// WithOpenEndedQueue for an unbounded sequence.
// seed: master seed for clone RNGs (INV-6).
func BuildSessionPool(blueprints []SessionBlueprint, r0Requests []*sim.Request, concurrent, total int, seed int64, opts ...SessionPoolOption) (*SessionPoolDriver, []*sim.Request, error) {
	var settings sessionPoolSettings
	for _, opt := range opts {
		opt(&settings)
	}

	if concurrent < 1 {
		return nil, nil, fmt.Errorf("concurrent must be >= 1, got %d", concurrent)
	}
	if len(blueprints) == 0 {
		return nil, nil, fmt.Errorf("no session blueprints to pool")
	}
	if len(blueprints) != len(r0Requests) {
		return nil, nil, fmt.Errorf("blueprints (%d) and round-0 requests (%d) count mismatch", len(blueprints), len(r0Requests))
	}

	// A count and an open-ended queue are alternatives, never both (R1).
	if settings.openEnded {
		if total > 0 {
			return nil, nil, fmt.Errorf("a session-count total (%d) and an open-ended queue are mutually exclusive (pass 0 as total with WithOpenEndedQueue)", total)
		}
	} else {
		if total < 1 {
			total = len(blueprints)
		}
		if total < concurrent {
			return nil, nil, fmt.Errorf("total (%d) must be >= concurrent (%d)", total, concurrent)
		}
	}

	// Validate up front: lazy materialization would otherwise defer a malformed
	// blueprint's panic into mid-run. Clones inherit MaxRounds, so this covers them.
	for i := range blueprints {
		if blueprints[i].MaxRounds < 1 && !blueprints[i].UnlimitedRounds {
			return nil, nil, fmt.Errorf("session %s has MaxRounds=%d, must be >= 1", blueprints[i].SessionID, blueprints[i].MaxRounds)
		}
	}

	// Pristine clone templates (see SessionPoolDriver.cloneTemplates).
	cloneTemplates := make([]sim.Request, len(r0Requests))
	for i, r := range r0Requests {
		cloneTemplates[i] = *r
	}

	d := &SessionPoolDriver{
		mgr:            NewSessionManager(nil),
		srcBPs:         blueprints,
		srcR0:          r0Requests,
		cloneTemplates: cloneTemplates,
		cloneRNG:       rand.New(rand.NewSource(seed)),
		limit:          total, // 0 when open-ended
	}

	// Inject the first concurrent sessions immediately.
	initial := make([]*sim.Request, 0, concurrent)
	for i := 0; i < concurrent; i++ {
		req, ok := d.admitNext()
		if !ok {
			break
		}
		initial = append(initial, req)
	}
	return d, initial, nil
}

// admitNext materializes and registers the next session and returns its round-0
// request, or ok=false when a counted sequence is exhausted (an open-ended one never is).
func (d *SessionPoolDriver) admitNext() (*sim.Request, bool) {
	if d.limit > 0 && d.nextIndex >= d.limit {
		return nil, false
	}
	i := d.nextIndex
	srcIdx := i % len(d.srcBPs)

	var bp SessionBlueprint
	var r0 *sim.Request
	if i < len(d.srcBPs) {
		bp, r0 = d.srcBPs[srcIdx], d.srcR0[srcIdx]
	} else {
		d.clonesMade++
		bp, r0 = cloneBlueprintForDup(d.srcBPs[srcIdx], &d.cloneTemplates[srcIdx], d.clonesMade, d.cloneRNG)
	}
	d.mgr.RegisterSession(bp)
	d.nextIndex++
	d.totalStarted++
	return r0, true
}

// isSessionRequest reports whether a completion drove its session to a terminal
// state. The wrapped SessionManager returns nil follow-ups on every terminal
// path (timeout, dropped, final round, budget, horizon). A round that continues
// returns exactly one follow-up. So: nil follow-ups AND this was a session
// request => the session just terminated.
func isSessionRequest(req *sim.Request) bool { return req.SessionID != "" }

// OnComplete wraps SessionManager.OnComplete. When the inner manager terminates a
// session (returns no follow-up for a session request), the driver admits the next
// session to refill the pool — unless a clock bound has been reached, or the
// counted sequence is exhausted.
func (d *SessionPoolDriver) OnComplete(req *sim.Request, tick int64) []*sim.Request {
	followUps := d.mgr.OnComplete(req, tick)
	if !isSessionRequest(req) {
		return followUps
	}
	if len(followUps) > 0 {
		// Session continues (intra-session follow-up). Pool membership unchanged.
		return followUps
	}
	// The pool bound is maintained structurally: each termination admits at most one
	// replacement, so the concurrently-active count never exceeds the initial pool.
	d.totalTerminated++

	if d.closed.Load() {
		return nil
	}
	next, ok := d.admitNext()
	if !ok {
		return nil
	}
	// Admit at the completion tick (a fresh user starts as this one ends).
	// Deadline is an ABSOLUTE timestamp on the same clock origin as ArrivalTime,
	// so rebasing arrival to `tick` without shifting the deadline would leave a
	// stale past deadline on every wave-2+ session — instantly dropped on enqueue
	// (simulator.go deadline guard), mass-cancelling the pool. Preserve the
	// recorded deadline-relative-to-arrival gap by shifting the deadline by the
	// same offset. (OTel corpora set no deadline; run/observe corpora do.)
	if next.Deadline > 0 {
		next.Deadline += tick - next.ArrivalTime
	}
	next.ArrivalTime = tick
	return []*sim.Request{next}
}

// TotalSessions returns the number of sessions in the pool: the configured total for a
// counted pool, or — for an open-ended one, whose size is only known at the end — the
// number actually started.
func (d *SessionPoolDriver) TotalSessions() int {
	if d.limit > 0 {
		return d.limit
	}
	return d.totalStarted
}

// StopAdmitting tells the driver that nothing further will be dispatched, so a session
// terminating from here on must not cause another to be admitted. Without it a caller
// that stops sending while requests are still in flight would keep admitting
// replacements it can never send, inflating SessionsStarted() by up to the pool size.
// Idempotent, and safe to call from a different goroutine than the one driving
// OnComplete — which is the normal case, since the decision to stop belongs to whatever
// owns the run rather than to the completion path.
func (d *SessionPoolDriver) StopAdmitting() { d.closed.Store(true) }

// SessionsStarted returns the number of sessions injected (initial pool + refills).
// For an open-ended queue this is the run's realized size, not known in advance.
func (d *SessionPoolDriver) SessionsStarted() int { return d.totalStarted }

// SessionsTerminated returns the number of started sessions that reached a terminal
// state. A shortfall against SessionsStarted() means sessions ended in flight — the
// anomaly Unstarted() reports for a counted pool but cannot see for an open-ended one,
// where nothing is ever queued-but-unadmitted (INV-11).
func (d *SessionPoolDriver) SessionsTerminated() int { return d.totalTerminated }

// Unstarted returns the number of sessions never admitted. For a COUNT-bounded pool it
// is nonzero when either (a) a hard --horizon cap ends the run with sessions still
// queued (replay only — observe rejects --horizon), or
// (b) an admitted session's request left the pipeline before reaching an instance
// (routing rejection, gateway shed/evict/expire) — OnRequestDone is wired
// per-instance, so those pre-instance drops never terminate the session and its
// pool slot is not refilled. Case (b) can make Unstarted() > 0 even with no
// --horizon set. The caller logs a warning and reports this alongside totalStarted
// so accounting stays closed: totalStarted + Unstarted() == TotalSessions()
// (INV-11 / INV-1).
//
// Always 0 for an open-ended queue: sessions are materialized at admission, so one
// never admitted is never created. Use SessionsTerminated() vs SessionsStarted() there.
func (d *SessionPoolDriver) Unstarted() int { return d.TotalSessions() - d.totalStarted }

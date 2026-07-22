package workload

import "github.com/inference-sim/inference-sim/sim"

// SessionDriver abstracts the closed-loop session engine that observe's
// orchestrator drives on each request completion. Two concrete implementations
// satisfy it:
//
//   - *SessionManager: the synthesized-workload path (single- and multi-session
//     reasoning). Follow-ups are governed by each blueprint's MaxRounds and an
//     optional global follow-up budget.
//   - *SessionPoolDriver: the trace-input path. A fixed pool of C concurrent
//     sessions is drawn from a recorded corpus (duplicated with cache-busting to
//     M total); when a session terminates the next queued session's round-0
//     request is admitted.
//
// Extracting the interface lets runObserveOrchestrator drive either engine
// without knowing which fixed-pool vs. free-running semantics apply. Both the
// observe trace path and blis replay route their fixed-pool semantics through
// the single BuildSessionPool source, so per-request behavior is identical
// (INV-13 run/replay/observe parity).
type SessionDriver interface {
	// OnComplete is called when a request reaches a terminal state. It returns
	// the follow-up requests to inject next (nil when the session terminates).
	OnComplete(req *sim.Request, tick int64) []*sim.Request
	// SetFollowUpBudget caps the number of follow-ups the driver will generate.
	// The precise meaning is implementation-specific; see each implementation.
	SetFollowUpBudget(budget int64)
}

// Compile-time assertions that both engines satisfy SessionDriver.
var (
	_ SessionDriver = (*SessionManager)(nil)
	_ SessionDriver = (*SessionPoolDriver)(nil)
)

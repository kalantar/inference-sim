package workload

import (
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// SpecHasAdapterFields reports whether any client or cohort in the spec declares a
// non-empty adapter id. Used by paths that do NOT support the LoRA control plane
// (e.g. `blis observe`, which dispatches to a real server that manages its own
// adapter loading) to detect and warn about adapter fields that would otherwise be
// silently threaded onto requests without any registry to validate against (#1464).
func SpecHasAdapterFields(spec *WorkloadSpec) bool {
	if spec == nil {
		return false
	}
	for i := range spec.Clients {
		if spec.Clients[i].Adapter != "" {
			return true
		}
	}
	for i := range spec.Cohorts {
		if spec.Cohorts[i].Adapter != "" {
			return true
		}
	}
	return false
}

// SpecAdapterIDs returns every distinct non-empty adapter id the spec references
// (clients and cohorts), sorted. Sorted output keeps any diagnostic built from it
// byte-identical across runs (INV-6); the slice is freshly allocated, so callers
// cannot mutate spec state through it.
//
// A pure query, the list-valued companion to SpecHasAdapterFields: paths that
// dispatch to a real server use it to preflight the referenced ids against the
// adapters the target actually serves (#1464's "no registry to validate against"
// is answerable by asking the server).
func SpecAdapterIDs(spec *WorkloadSpec) []string {
	if spec == nil {
		return nil
	}
	seen := make(map[string]struct{})
	for i := range spec.Clients {
		if id := spec.Clients[i].Adapter; id != "" {
			seen[id] = struct{}{}
		}
	}
	for i := range spec.Cohorts {
		if id := spec.Cohorts[i].Adapter; id != "" {
			seen[id] = struct{}{}
		}
	}
	if len(seen) == 0 {
		return nil
	}
	ids := make([]string, 0, len(seen))
	for id := range seen {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// ValidateAdapterReferences cross-checks every adapter id referenced by the workload
// spec (clients and cohorts) against the declared adapter registry (#1464, US1):
//
//   - a non-empty adapter id MUST be a registry key (completeness), and
//   - if the adapter declares a base model, it MUST equal the referencing
//     client/cohort model (an adapter cannot attach to the wrong base model).
//
// An omitted adapter ("") is always valid — a base-model-only request, the no-op
// default (INV-6). A nil registry means no adapters are configured; in that case any
// non-empty reference is an error (a workload cannot reference adapters that were
// never declared). With no references and no registry the spec is inert.
//
// This is a pure validation query (no mutation). The CLI boundary decides fatality
// (cmd/ -> logrus.Fatalf); the library returns an error.
func ValidateAdapterReferences(spec *WorkloadSpec, reg sim.AdapterRegistry) error {
	if spec == nil {
		return nil
	}

	check := func(kind, id, adapter, model string) error {
		if adapter == "" {
			return nil // base-model-only request
		}
		if reg == nil {
			return fmt.Errorf("%s %q references adapter %q but no adapters are declared", kind, id, adapter)
		}
		if !reg.Has(adapter) {
			return fmt.Errorf("%s %q references unknown adapter %q (not in registry)", kind, id, adapter)
		}
		if baseModel, _ := reg.BaseModelOf(adapter); baseModel != "" && baseModel != model {
			return fmt.Errorf("%s %q model %q does not match adapter %q base model %q",
				kind, id, model, adapter, baseModel)
		}
		return nil
	}

	for i := range spec.Clients {
		c := &spec.Clients[i]
		if err := check("client", c.ID, c.Adapter, c.Model); err != nil {
			return err
		}
	}
	for i := range spec.Cohorts {
		h := &spec.Cohorts[i]
		if err := check("cohort", h.ID, h.Adapter, h.Model); err != nil {
			return err
		}
	}
	return nil
}

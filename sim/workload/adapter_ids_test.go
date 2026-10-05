package workload

import (
	"reflect"
	"testing"
)

// SpecAdapterIDs is the list-valued companion to SpecHasAdapterFields: the ids a
// real-server dispatch path preflights. Sorted and deduped, so any diagnostic
// built from it is byte-identical across runs (INV-6).
func TestSpecAdapterIDs(t *testing.T) {
	tests := []struct {
		name string
		spec *WorkloadSpec
		want []string
	}{
		{"nil spec", nil, nil},
		{"no adapters", &WorkloadSpec{Clients: []ClientSpec{{ID: "a"}, {ID: "b"}}}, nil},
		{
			name: "sorted, not declaration order",
			spec: &WorkloadSpec{Clients: []ClientSpec{
				{ID: "a", Adapter: "sql-lora"},
				{ID: "b", Adapter: "code-lora"},
			}},
			want: []string{"code-lora", "sql-lora"},
		},
		{
			name: "deduped across clients",
			spec: &WorkloadSpec{Clients: []ClientSpec{
				{ID: "a", Adapter: "sql-lora"},
				{ID: "b", Adapter: "sql-lora"},
			}},
			want: []string{"sql-lora"},
		},
		{
			name: "cohorts contribute too",
			spec: &WorkloadSpec{
				Clients: []ClientSpec{{ID: "a", Adapter: "sql-lora"}},
				Cohorts: []CohortSpec{{ID: "h", Adapter: "code-lora"}},
			},
			want: []string{"code-lora", "sql-lora"},
		},
		{
			name: "empty ids are not adapters",
			spec: &WorkloadSpec{Clients: []ClientSpec{
				{ID: "a", Adapter: ""},
				{ID: "b", Adapter: "sql-lora"},
			}},
			want: []string{"sql-lora"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := SpecAdapterIDs(tc.spec)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("SpecAdapterIDs() = %v, want %v", got, tc.want)
			}
			// Agreement with the existing bool query: non-empty iff it reports true.
			if hasFields := SpecHasAdapterFields(tc.spec); hasFields != (len(got) > 0) {
				t.Errorf("SpecHasAdapterFields() = %v but SpecAdapterIDs() = %v; the two must agree",
					hasFields, got)
			}
		})
	}
}

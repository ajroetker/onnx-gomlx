package onnx

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/simplego"
	_ "github.com/gomlx/gomlx/backends/simplego/highway"
)

func TestDebugQuantizedDenseDetection(t *testing.T) {
	data, err := os.ReadFile("/Users/ajroetker/go/src/github.com/antflydb/antfly/lib/embeddings/antfly_model/model_i8.onnx")
	if err != nil {
		t.Skipf("model not found: %v", err)
	}

	m, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	// Count detected fusion groups by type (deduplicate multi-key entries).
	seen := make(map[*FusionGroup]bool)
	counts := make(map[FusionType]int)
	for _, fg := range m.detectedFusionGroups {
		if seen[fg] {
			continue
		}
		seen[fg] = true
		counts[fg.Type]++
	}

	fmt.Printf("Detected fusion groups (unique):\n")
	fmt.Printf("  SDPA: %d\n", counts[FusionSDPA])
	fmt.Printf("  QKVDense: %d\n", counts[FusionQKVDense])
	fmt.Printf("  DenseGelu: %d\n", counts[FusionDenseGelu])
	fmt.Printf("  QuantizedDense: %d\n", counts[FusionQuantizedDense])
	fmt.Printf("  QuantizedQKVDense: %d\n", counts[FusionQuantizedQKVDense])
	fmt.Printf("  QuantizedSDPA: %d\n", counts[FusionQuantizedSDPA])

	// Print details of QuantizedDense groups.
	geluCount := 0
	biasCount := 0
	for name, fg := range m.detectedFusionGroups {
		if fg.Type != FusionQuantizedDense {
			continue
		}
		p := fg.Params.(*QuantizedDenseParams)
		flags := ""
		if p.HasGelu {
			flags += "+GELU"
			geluCount++
		}
		if p.BiasName != "" {
			flags += "+bias"
			biasCount++
		}
		fmt.Printf("  QD: %s  [%dx%d] %s\n", name, p.K, p.N, flags)
	}
	fmt.Printf("  With GELU: %d, with bias: %d\n", geluCount, biasCount)

	// Print details of QuantizedQKVDense groups.
	seen = make(map[*FusionGroup]bool)
	for _, fg := range m.detectedFusionGroups {
		if fg.Type != FusionQuantizedQKVDense || seen[fg] {
			continue
		}
		seen[fg] = true
		p := fg.Params.(*QuantizedQKVDenseParams)
		fmt.Printf("  QQKV: Q=%s K=%s V=%s  [%dx%d]\n",
			p.QOutputName, p.KOutputName, p.VOutputName, p.K, p.QDim)
	}

	// Expect 6 QuantizedQKVDense (one per layer) and 18 remaining QuantizedDense
	// (6 attention_output + 6 FFN_intermediate + 6 FFN_output).
	if counts[FusionQuantizedQKVDense] != 6 {
		t.Errorf("Expected 6 QuantizedQKVDense, got %d", counts[FusionQuantizedQKVDense])
	}
	if counts[FusionQuantizedDense] != 18 {
		t.Errorf("Expected 18 remaining QuantizedDense, got %d", counts[FusionQuantizedDense])
	}

	// Verify active fusion groups with simplego backend.
	engine, err := backends.NewWithConfig("go")
	if err != nil {
		t.Skipf("simplego backend not available: %v", err)
	}
	defer engine.Finalize()
	active := m.buildActiveFusionGroups(engine.Capabilities())
	activeQD := 0
	activeQQKV := 0
	activeQSDPA := 0
	seen = make(map[*FusionGroup]bool)
	for _, fg := range active {
		if seen[fg] {
			continue
		}
		seen[fg] = true
		switch fg.Type {
		case FusionQuantizedDense:
			activeQD++
		case FusionQuantizedQKVDense:
			activeQQKV++
		case FusionQuantizedSDPA:
			activeQSDPA++
		}
	}
	fmt.Printf("Active QuantizedDense (simplego): %d\n", activeQD)
	fmt.Printf("Active QuantizedQKVDense (simplego): %d\n", activeQQKV)
	fmt.Printf("Active QuantizedSDPA (simplego): %d\n", activeQSDPA)

	if activeQD == 0 {
		t.Errorf("Expected active QuantizedDense groups, got 0")
	}
	if activeQQKV == 0 {
		t.Errorf("Expected active QuantizedQKVDense groups, got 0")
	}
	if activeQSDPA == 0 {
		t.Errorf("Expected active QuantizedSDPA groups, got 0")
	}

	// Print and verify QuantizedSDPA details.
	seen = make(map[*FusionGroup]bool)
	for name, fg := range m.detectedFusionGroups {
		if fg.Type != FusionQuantizedSDPA || seen[fg] {
			continue
		}
		seen[fg] = true
		p := fg.Params.(*QuantizedSDPAParams)
		fmt.Printf("  QSDPA: %s  heads=%d kv_heads=%d scale=%.4f kIsTransposed=%v\n",
			name, p.NumHeads, p.NumKVHeads, p.Scale, p.KIsTransposed)
		// MiniLM-L6-v2 has 12 attention heads.
		if p.NumHeads != 12 {
			t.Errorf("QSDPA %s: expected numHeads=12, got %d", name, p.NumHeads)
		}
	}
}

package onnx

import (
	"fmt"
	"os"
	"testing"
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

	// Count detected fusions by Name (deduplicate multi-key entries).
	seen := make(map[FusionCandidate]bool)
	counts := make(map[string]int)
	for _, cand := range m.detectedFusions {
		if seen[cand] {
			continue
		}
		seen[cand] = true
		counts[cand.Name()]++
	}

	fmt.Printf("Detected fusions (unique):\n")
	fmt.Printf("  SDPA: %d\n", counts["SDPA"])
	fmt.Printf("  QKVDense: %d\n", counts["QKVDense"])
	fmt.Printf("  DenseGelu: %d\n", counts["DenseGelu"])
	fmt.Printf("  QuantizedDense: %d\n", counts["QuantizedDense"])
	fmt.Printf("  QuantizedQKVDense: %d\n", counts["QuantizedQKVDense"])
	fmt.Printf("  QuantizedSDPA: %d\n", counts["QuantizedSDPA"])

	// Print details of QuantizedDense candidates.
	geluCount := 0
	biasCount := 0
	seen = make(map[FusionCandidate]bool)
	for name, cand := range m.detectedFusions {
		if cand.Name() != "QuantizedDense" || seen[cand] {
			continue
		}
		seen[cand] = true
		qdc, ok := cand.(*quantizedDenseCandidate)
		if !ok {
			continue
		}
		p := qdc.params
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

	// Print details of QuantizedQKVDense candidates.
	seen = make(map[FusionCandidate]bool)
	for _, cand := range m.detectedFusions {
		if cand.Name() != "QuantizedQKVDense" || seen[cand] {
			continue
		}
		seen[cand] = true
		qkvc, ok := cand.(*quantizedQKVDenseCandidate)
		if !ok {
			continue
		}
		p := qkvc.params
		fmt.Printf("  QQKV: Q=%s K=%s V=%s  [%dx%d]\n",
			p.QOutputName, p.KOutputName, p.VOutputName, p.K, p.QDim)
	}

	// Expect 6 QuantizedQKVDense (one per layer) and 18 remaining QuantizedDense
	// (6 attention_output + 6 FFN_intermediate + 6 FFN_output).
	if counts["QuantizedQKVDense"] != 6 {
		t.Errorf("Expected 6 QuantizedQKVDense, got %d", counts["QuantizedQKVDense"])
	}
	if counts["QuantizedDense"] != 18 {
		t.Errorf("Expected 18 remaining QuantizedDense, got %d", counts["QuantizedDense"])
	}

	// In the new framework all detected fusions are active (wrappers handle backend fallback).
	// Just verify we detected the expected quantized fusions.
	if counts["QuantizedDense"] == 0 {
		t.Errorf("Expected QuantizedDense fusions, got 0")
	}
	if counts["QuantizedQKVDense"] == 0 {
		t.Errorf("Expected QuantizedQKVDense fusions, got 0")
	}
	if counts["QuantizedSDPA"] == 0 {
		t.Errorf("Expected QuantizedSDPA fusions, got 0")
	}

	// Print and verify QuantizedSDPA details.
	seen = make(map[FusionCandidate]bool)
	for name, cand := range m.detectedFusions {
		if cand.Name() != "QuantizedSDPA" || seen[cand] {
			continue
		}
		seen[cand] = true
		qsc, ok := cand.(*quantizedSDPACandidate)
		if !ok {
			continue
		}
		p := qsc.params
		fmt.Printf("  QSDPA: %s  heads=%d kv_heads=%d scale=%.4f kIsTransposed=%v\n",
			name, p.NumHeads, p.NumKVHeads, p.Scale, p.KIsTransposed)
		// MiniLM-L6-v2 has 12 attention heads.
		if p.NumHeads != 12 {
			t.Errorf("QSDPA %s: expected numHeads=12, got %d", name, p.NumHeads)
		}
	}
}

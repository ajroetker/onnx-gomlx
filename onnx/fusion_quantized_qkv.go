package onnx

import (
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph" //nolint
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/nn"
)

// QuantizedQKVDenseParams holds parameters for fused Q/K/V int8 projections sharing
// the same float input. Instead of 3 separate QuantizedDense calls (each with its own
// SMEGuard + parallel dispatch), we concatenate weights and call QuantizedDense once.
type QuantizedQKVDenseParams struct {
	FloatInputName string

	WQName, WKName, WVName          string
	ScaleQName, ScaleKName, ScaleVName string
	BiasQName, BiasKName, BiasVName string

	QOutputName, KOutputName, VOutputName string

	K     int // input features (shared)
	QDim  int // Q output features
	KVDim int // K and V output features (must be equal)
}

// qdEntry pairs a QuantizedDense fusion group with its map key.
type qdEntry struct {
	fg  *FusionGroup
	key string // map key (RootOutputName)
}

// detectQuantizedQKVDensePatterns runs after detectQuantizedDensePatterns and merges
// triplets of QuantizedDense groups that share the same FloatInputName into a single
// QuantizedQKVDense group. This reduces kernel launches (and SMEGuard transitions)
// from 3 to 1 per attention layer.
func (m *Model) detectQuantizedQKVDensePatterns() {
	// Group existing QuantizedDense fusions by FloatInputName.
	byInput := make(map[string][]qdEntry)
	for key, fg := range m.detectedFusionGroups {
		if fg.Type != FusionQuantizedDense {
			continue
		}
		// Skip groups registered under secondary keys (same pointer, different key).
		if key != fg.RootOutputName {
			continue
		}
		p := fg.Params.(*QuantizedDenseParams)
		// Only consider groups without GELU (attention projections don't use GELU).
		if p.HasGelu {
			continue
		}
		byInput[p.FloatInputName] = append(byInput[p.FloatInputName], qdEntry{fg: fg, key: key})
	}

	for _, entries := range byInput {
		if len(entries) != 3 {
			continue
		}
		m.tryMergeQuantizedQKV(entries)
	}
}

// tryMergeQuantizedQKV attempts to merge 3 QuantizedDense groups into one QKV group.
func (m *Model) tryMergeQuantizedQKV(entries []qdEntry) {
	params := make([]*QuantizedDenseParams, 3)
	for i, e := range entries {
		params[i] = e.fg.Params.(*QuantizedDenseParams)
	}

	// All must share the same K (input features).
	K := params[0].K
	for _, p := range params[1:] {
		if p.K != K {
			return
		}
	}

	// Determine Q, K, V ordering by dimension. If two share the same N and
	// one differs, the differing one is Q. If all equal, keep appearance order.
	qIdx, kIdx, vIdx := 0, 1, 2
	d0, d1, d2 := params[0].N, params[1].N, params[2].N
	if d0 == d1 && d0 != d2 {
		qIdx, kIdx, vIdx = 2, 0, 1
	} else if d0 == d2 && d0 != d1 {
		qIdx, kIdx, vIdx = 1, 0, 2
	} else if d1 == d2 && d1 != d0 {
		qIdx, kIdx, vIdx = 0, 1, 2
	}

	qP := params[qIdx]
	kP := params[kIdx]
	vP := params[vIdx]

	// K and V must have equal N, and QDim must equal KVDim for uniform groupSize.
	if kP.N != vP.N {
		return
	}
	if qP.N != kP.N {
		// Non-uniform projection dims (e.g. GQA) — can't use a single groupSize.
		return
	}

	// Bias must be all-or-nothing.
	hasBias := qP.BiasName != "" && kP.BiasName != "" && vP.BiasName != ""
	noBias := qP.BiasName == "" && kP.BiasName == "" && vP.BiasName == ""
	if !hasBias && !noBias {
		return
	}

	// Merge internal outputs from all 3 groups.
	mergedInternalOutputs := make(map[string]bool)
	for _, e := range entries {
		for name := range e.fg.InternalOutputNames {
			mergedInternalOutputs[name] = true
		}
		// The 3 individual root outputs become internal to the QKV group.
		mergedInternalOutputs[e.fg.RootOutputName] = true
	}

	externalInputs := []string{
		qP.FloatInputName,
		qP.BWeightName, kP.BWeightName, vP.BWeightName,
		qP.BScaleName, kP.BScaleName, vP.BScaleName,
	}
	if hasBias {
		externalInputs = append(externalInputs, qP.BiasName, kP.BiasName, vP.BiasName)
	}

	qkvParams := &QuantizedQKVDenseParams{
		FloatInputName: qP.FloatInputName,
		WQName:         qP.BWeightName,
		WKName:         kP.BWeightName,
		WVName:         vP.BWeightName,
		ScaleQName:     qP.BScaleName,
		ScaleKName:     kP.BScaleName,
		ScaleVName:     vP.BScaleName,
		QOutputName:    entries[qIdx].fg.RootOutputName,
		KOutputName:    entries[kIdx].fg.RootOutputName,
		VOutputName:    entries[vIdx].fg.RootOutputName,
		K:              K,
		QDim:           qP.N,
		KVDim:          kP.N,
	}
	if hasBias {
		qkvParams.BiasQName = qP.BiasName
		qkvParams.BiasKName = kP.BiasName
		qkvParams.BiasVName = vP.BiasName
	}

	fg := &FusionGroup{
		Type:                FusionQuantizedQKVDense,
		RootOutputName:      qkvParams.QOutputName,
		InternalOutputNames: mergedInternalOutputs,
		ExternalInputNames:  externalInputs,
		Params:              qkvParams,
	}

	// Remove the 3 individual QuantizedDense groups.
	for _, e := range entries {
		delete(m.detectedFusionGroups, e.key)
	}

	// Register the QKV group under all 3 output names.
	m.detectedFusionGroups[qkvParams.QOutputName] = fg
	m.detectedFusionGroups[qkvParams.KOutputName] = fg
	m.detectedFusionGroups[qkvParams.VOutputName] = fg
}

// emitQuantizedQKVDense emits a single QuantizedDense call with concatenated Q/K/V
// weights and splits the output into Q, K, V tensors.
func (m *Model) emitQuantizedQKVDense(_ *context.Context, g *Graph, fg *FusionGroup, convertedOutputs map[string]*Node) {
	p := fg.Params.(*QuantizedQKVDenseParams)

	floatInput := convertedOutputs[p.FloatInputName]
	wQ := convertedOutputs[p.WQName]
	wK := convertedOutputs[p.WKName]
	wV := convertedOutputs[p.WVName]

	scaleQ := convertedOutputs[p.ScaleQName]
	scaleK := convertedOutputs[p.ScaleKName]
	scaleV := convertedOutputs[p.ScaleVName]

	// Concatenate int8 weights: [K, QDim] + [K, KVDim] + [K, KVDim] → [K, 3*QDim]
	wQKV := Concatenate([]*Node{wQ, wK, wV}, 1)
	totalN := p.QDim + 2*p.KVDim
	groupSize := p.QDim // Each projection is one quantization group.

	// Build scales [K, 3]: each scalar scale broadcast to [K, 1] then concatenated.
	scaleQCol := ExpandAndBroadcast(ConvertDType(scaleQ, dtypes.Float32), []int{p.K, 1}, []int{0, 1})
	scaleKCol := ExpandAndBroadcast(ConvertDType(scaleK, dtypes.Float32), []int{p.K, 1}, []int{0, 1})
	scaleVCol := ExpandAndBroadcast(ConvertDType(scaleV, dtypes.Float32), []int{p.K, 1}, []int{0, 1})
	fusedScales := Concatenate([]*Node{scaleQCol, scaleKCol, scaleVCol}, 1) // [K, 3]

	// Concatenate biases if present: [QDim] + [KVDim] + [KVDim] → [totalN]
	var bias *Node
	if p.BiasQName != "" {
		biasQ := convertedOutputs[p.BiasQName]
		biasK := convertedOutputs[p.BiasKName]
		biasV := convertedOutputs[p.BiasVName]
		bias = Concatenate([]*Node{biasQ, biasK, biasV}, 0)
	}

	// Single fused QuantizedDense for all 3 projections.
	result := nn.QuantizedDense(floatInput, wQKV, fusedScales, bias,
		backends.QuantInt8, groupSize, totalN)

	// Split output along last axis: [batch..., totalN] → Q, K, V each [batch..., QDim]
	parts := Split(result, -1, 3)
	convertedOutputs[p.QOutputName] = parts[0]
	convertedOutputs[p.KOutputName] = parts[1]
	convertedOutputs[p.VOutputName] = parts[2]
}

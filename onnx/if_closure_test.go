package onnx

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/stretchr/testify/require"
)

// TestIfClosureBasic tests the If operation with closure-backed graphs
// This is a basic test to verify the infrastructure works
func TestIfClosureBasic(t *testing.T) {
	backend, err := backends.New()
	require.NoError(t, err)
	require.NotNil(t, backend, "Backend required for If closure tests")

	t.Run("true condition", func(t *testing.T) {
		result := MustExecOnce(backend, func(cond *Node) *Node {
			g := cond.Graph()

			// Create parent values
			a := Scalar(g, dtypes.Float32, float32(5.0))
			b := Scalar(g, dtypes.Float32, float32(3.0))

			// Create closure graphs
			thenG := g.NewClosureGraph("then")
			elseG := g.NewClosureGraph("else")

			if thenG == nil || elseG == nil {
				t.Skip("Backend doesn't support closures")
				return nil
			}

			// Import parent values into closures
			thenA := thenG.UseParentValue(a)
			thenB := thenG.UseParentValue(b)
			elseA := elseG.UseParentValue(a)
			elseB := elseG.UseParentValue(b)

			// Build operations in closures
			thenResult := Add(thenA, thenB) // 5 + 3 = 8
			elseResult := Sub(elseA, elseB) // 5 - 3 = 2

			// Compile closures
			thenG.CompileClosure(thenResult)
			elseG.CompileClosure(elseResult)

			// Use IfClosure
			return IfClosure(cond, thenG, elseG)[0]
		}, true) // cond = true

		// Should return 5+3 = 8
		require.Equal(t, float32(8.0), result.Value().(float32))
	})

	t.Run("false condition", func(t *testing.T) {
		result := MustExecOnce(backend, func(cond *Node) *Node {
			g := cond.Graph()

			// Create parent values
			a := Scalar(g, dtypes.Float32, float32(10.0))
			b := Scalar(g, dtypes.Float32, float32(4.0))

			// Create closure graphs
			thenG := g.NewClosureGraph("then")
			elseG := g.NewClosureGraph("else")

			if thenG == nil || elseG == nil {
				t.Skip("Backend doesn't support closures")
				return nil
			}

			// Import parent values into closures
			thenA := thenG.UseParentValue(a)
			thenB := thenG.UseParentValue(b)
			elseA := elseG.UseParentValue(a)
			elseB := elseG.UseParentValue(b)

			// Build operations in closures
			thenResult := Add(thenA, thenB) // 10 + 4 = 14
			elseResult := Sub(elseA, elseB) // 10 - 4 = 6

			// Compile closures
			thenG.CompileClosure(thenResult)
			elseG.CompileClosure(elseResult)

			// Use IfClosure
			return IfClosure(cond, thenG, elseG)[0]
		}, false) // cond = false

		// Should return 10-4 = 6
		require.Equal(t, float32(6.0), result.Value().(float32))
	})
}

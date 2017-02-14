package distplot

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	p1 := &Point{}
	p2 := &Point{}
	p1.Distances = map[*Point]float32{p2: 3}
	p2.Distances = map[*Point]float32{p1: 1}
	done := make(chan struct{})
	Solve([]*Point{p1, p2}, done, func(iter int, cost float32) {
		if iter == 10000 {
			close(done)
		}
	})
	dist := math.Sqrt(math.Pow(float64(p1.X-p2.X), 2) + math.Pow(float64(p1.Y-p2.Y), 2))
	if math.Abs(dist-2) > 1e-2 || math.IsNaN(dist) {
		t.Errorf("expected distance 2 but got %f", dist)
	}
}

// Package distplot lays out objects on a 2-dimensional
// plane by minimizing the mean squared difference between
// the desired distance and the actual distance between
// all possible pairs of points.
package distplot

import (
	"math/rand"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

const stepSize = 1e-3
const damping = 1e-10

// A Point stores the distances which a data point wants
// to have to its neighboring points.
type Point struct {
	Distances map[*Point]float32

	// After solving, the final coordinates of the point are
	// stored here.
	X float32
	Y float32
}

// Solve attempts to minimize the mean squared difference
// in actual and desired distances.
//
// Solving stops when doneChan is closed.
//
// If non-nil, logFunc is called with the cost at every
// iteration of solving.
func Solve(points []*Point, doneChan <-chan struct{}, logFunc func(iter int, cost float32)) {
	matData := make([]float32, len(points)*2)
	for i := range points {
		matData[i*2] = float32(rand.NormFloat64())
		matData[i*2+1] = float32(rand.NormFloat64())
	}
	coordMat := anydiff.NewVar(anyvec32.MakeVectorData(matData))
	defer func() {
		matData = coordMat.Output().Data().([]float32)
		for i, p := range points {
			p.X = matData[i*2]
			p.Y = matData[i*2+1]
		}
	}()

	desired := desiredDistMatrix(points)

	tr := &anysgd.Adam{}
	iter := 0
	for {
		select {
		case <-doneChan:
			return
		default:
		}
		mse := anynet.MSE{}
		cost := mse.Cost(desired, currentDistMat(coordMat), 1)
		if logFunc != nil {
			logFunc(iter, anyvec.Sum(cost.Output()).(float32))
		}
		iter++
		grad := anydiff.NewGrad(coordMat)
		upstream := anyvec32.MakeVectorData([]float32{1})
		cost.Propagate(upstream, grad)
		grad = tr.Transform(grad)
		grad.Scale(-float32(stepSize))
		grad.AddToVars()
	}
}

func desiredDistMatrix(points []*Point) anydiff.Res {
	distMatData := make([]float32, len(points)*len(points))
	for i, x := range points {
		for j, y := range points {
			distMatData[i+j*len(points)] = x.Distances[y]
		}
	}
	return anydiff.NewConst(anyvec32.MakeVectorData(distMatData))
}

func currentDistMat(points anydiff.Res) anydiff.Res {
	var rows []anydiff.Res
	for i := 0; i < points.Output().Len(); i += 2 {
		p := anydiff.Scale(anydiff.Slice(points, i, i+2), float32(-1))
		sqDist := anydiff.SumCols(&anydiff.Matrix{
			Data: anydiff.Pow(anydiff.AddRepeated(points, p), float32(2)),
			Rows: points.Output().Len() / 2,
			Cols: 2,
		})
		d := anyvec.Max(sqDist.Output()).(float32) * damping
		sqDist = anydiff.AddScalar(sqDist, d)
		dist := anydiff.Pow(sqDist, float32(0.5))
		rows = append(rows, dist)
	}
	return anydiff.Concat(rows...)
}

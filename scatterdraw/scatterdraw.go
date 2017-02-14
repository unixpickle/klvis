// Package scatterdraw produces scale-agnositic scatter
// plots.
package scatterdraw

import (
	"image"
	"image/color"
	"math"

	"github.com/llgcode/draw2d/draw2dimg"
)

const (
	pointSize  = 30.0 / 512.0
	marginSize = 40.0 / 512.0
)

// Point is a point in a scatter plot.
type Point struct {
	X     float64
	Y     float64
	Color color.Color
}

// Draw draws the points into an image of the given size.
// The scale of the image is adjusted appropriately.
func Draw(points []*Point, outSize int) image.Image {
	res := image.NewRGBA(image.Rect(0, 0, outSize, outSize))
	if len(points) == 0 {
		return res
	}

	minX, minY, maxX, maxY, dotSize := computeSize(points)
	fullSize := math.Max(maxX-minX, maxY-minY)
	xMargin := fullSize/2 - (maxX-minX)/2
	yMargin := fullSize/2 - (maxY-minY)/2
	scale := float64(outSize) / fullSize
	dotSize *= scale

	gc := draw2dimg.NewGraphicContext(res)

	for _, p := range points {
		destX := scale * (xMargin + (p.X - minX))
		destY := scale * (yMargin + (p.Y - minY))
		gc.SetFillColor(p.Color)
		gc.BeginPath()
		gc.MoveTo(destX-dotSize/2, destY)
		gc.ArcTo(destX, destY, dotSize/2, dotSize/2, 0, math.Pi*2)
		gc.Fill()
	}

	return res
}

func computeSize(p []*Point) (minX, minY, maxX, maxY, dotSize float64) {
	minX, minY = math.Inf(1), math.Inf(1)
	maxX, maxY = math.Inf(-1), math.Inf(-1)
	for _, x := range p {
		minX = math.Min(minX, x.X)
		minY = math.Min(minY, x.Y)
		maxX = math.Max(maxX, x.X)
		maxY = math.Max(maxY, x.Y)
	}
	size := math.Max(maxX-minX, maxY-minY)
	margin := marginSize * size
	maxX += margin
	maxY += margin
	minX -= margin
	minY -= margin
	dotSize = size * pointSize
	return
}

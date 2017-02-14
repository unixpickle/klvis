package scatterdraw

import (
	"image"
	"image/color"

	"github.com/unixpickle/klvis/distplot"
)

// DrawDistplot draws a plot of points from distplot.
func DrawDistplot(p []*distplot.Point, colors []color.Color, outSize int) image.Image {
	realP := make([]*Point, len(p))
	for i, x := range p {
		realP[i] = &Point{X: float64(x.X), Y: float64(x.Y), Color: colors[i]}
	}
	return Draw(realP, outSize)
}

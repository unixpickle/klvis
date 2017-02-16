// Command klvis produces visualizations for KL
// divergences.
package main

import (
	"encoding/csv"
	"flag"
	"image/color"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/klvis/distplot"
	"github.com/unixpickle/klvis/scatterdraw"
	"github.com/unixpickle/rip"
)

const logInterval = time.Second

var DotColors = []color.Color{
	color.RGBA{R: 0xff, A: 0xff},
	color.RGBA{G: 0xff, A: 0xff},
	color.RGBA{B: 0xff, A: 0xff},
	color.RGBA{R: 0xff, G: 0xff, A: 0xff},
}

var ColorNames = []string{"red", "green", "blue", "yellow"}

func main() {
	var outDir string
	var inFile string
	var plotSize int

	flag.StringVar(&outDir, "out", ".", "output directory")
	flag.StringVar(&inFile, "in", "", "input CSV file (lines like a,b,D(a|b))")
	flag.IntVar(&plotSize, "size", 512, "plot image size")

	flag.Parse()

	if inFile == "" {
		essentials.Die("Missing -in flag. See -help for more.")
	}

	in := readInput(inFile)
	dpPoints, names := distplotPoints(in)

	if len(names) > len(ColorNames) {
		essentials.Die("not enough colors")
	}

	log.Println("Solving distplot (ctrl+c to end)...")
	r := rip.NewRIP()
	to := time.After(logInterval)
	distplot.Solve(dpPoints, r.Chan(), func(iter int, cost float32) {
		select {
		case <-to:
			log.Printf("iter %d: cost=%v", iter, cost)
			to = time.After(logInterval)
		default:
		}
	})

	dpOut := filepath.Join(outDir, "distplot.png")
	log.Printf("Saving distplot to %s...", dpOut)
	img := scatterdraw.DrawDistplot(dpPoints, DotColors[:len(dpPoints)], plotSize)
	f, err := os.Create(dpOut)
	if err != nil {
		essentials.Die(err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		essentials.Die(err)
	}

	colorLabels := []string{}
	for i, n := range names {
		colorLabels = append(colorLabels, ColorNames[i]+"="+n)
	}
	log.Println("Labels:", strings.Join(colorLabels, " "))
}

type KLArgument struct {
	Left  string
	Right string
}

func readInput(path string) map[KLArgument]float32 {
	r, err := os.Open(path)
	if err != nil {
		essentials.Die("read input:", err)
	}
	cr := csv.NewReader(r)
	contents, err := cr.ReadAll()
	r.Close()
	if err != nil {
		essentials.Die("read input:", err)
	}
	res := map[KLArgument]float32{}
	for _, line := range contents {
		if len(line) != 3 {
			essentials.Die("input file must have 3 columns")
		}
		if val, err := strconv.ParseFloat(line[2], 32); err != nil {
			essentials.Die("invalid KL-divergence:", line[2])
		} else {
			res[KLArgument{line[0], line[1]}] = float32(val)
		}
	}
	return res
}

func distplotPoints(m map[KLArgument]float32) (ps []*distplot.Point, names []string) {
	points := map[string]*distplot.Point{}
	names = sortedNames(m)
	for _, name := range names {
		p := &distplot.Point{
			Distances: map[*distplot.Point]float32{},
		}
		points[name] = p
		ps = append(ps, p)
	}
	for arg, kl := range m {
		points[arg.Left].Distances[points[arg.Right]] = kl
	}
	return
}

func sortedNames(m map[KLArgument]float32) []string {
	res := []string{}
	has := map[string]bool{}
	for x := range m {
		for _, name := range []string{x.Left, x.Right} {
			if !has[name] {
				has[name] = true
				res = append(res, name)
			}
		}
	}
	sort.Strings(res)
	return res
}

// Command klrnn computes KL-divergences between a group
// of char-rnns on their corresponding validation corpori.
package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	charrnn "github.com/unixpickle/char-rnn"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

const (
	Validation = 0.1
	BatchSize  = 32
)

func main() {
	if len(os.Args) < 3 {
		dieUsage()
	}
	ins, samples := parseFlags()
	if len(ins) != len(samples) {
		essentials.Die("number of RNNs must match number of sample dirs")
	} else if len(ins) == 0 {
		essentials.Die("no RNNs")
	}

	names := rnnNames(ins)

	log.Println("Loading RNNs ...")
	blocks := readRNNs(ins)
	log.Println("Loading samples ...")
	sampleLists := readSampleLists(samples)

	ownCosts := make([]float64, len(blocks))
	for i, b := range blocks {
		s := sampleLists[i]
		log.Println("Computing entropy for", names[i], "...")
		ownCosts[i] = computeCost(b, s)
		log.Println("Entropy for", names[i], "is", ownCosts[i])
	}

	for i, s := range sampleLists {
		for j, b := range blocks {
			if i == j {
				continue
			}
			klName := fmt.Sprintf("KL(%s|%s)", names[i], names[j])
			log.Println("Computing", klName, "...")

			cost := computeCost(b, s)
			kl := cost - ownCosts[i]
			avgKL := kl / averageParagraph(s)
			log.Printf("%s = %f (avg=%f)", klName, kl, avgKL)
		}
	}
}

func parseFlags() (ins, samples []string) {
	parsingIns := false
	parsingSamples := false
	for _, x := range os.Args[1:] {
		if x == "-rnn" {
			parsingIns = true
			parsingSamples = false
		} else if x == "-samples" {
			parsingIns = false
			parsingSamples = true
		} else if parsingIns {
			ins = append(ins, x)
		} else if parsingSamples {
			samples = append(samples, x)
		} else {
			fmt.Fprintln(os.Stderr, "unexpected argument:", x)
		}
	}
	return
}

func dieUsage() {
	essentials.Die("Usage: rnnkl -rnn <in1> ... -samples <dir1> ...")
}

func readRNNs(rnns []string) []anyrnn.Block {
	res := make([]anyrnn.Block, len(rnns))
	for i, path := range rnns {
		d, err := ioutil.ReadFile(path)
		if err != nil {
			essentials.Die("read RNN:", err)
		}
		obj, err := serializer.DeserializeWithType(d)
		if err != nil {
			essentials.Die("read RNN "+path+":", err)
		}
		switch obj := obj.(type) {
		case *charrnn.LSTM:
			res[i] = obj.Block
		default:
			fmt.Fprintf(os.Stderr, "read RNN: unsupported type: %T", obj)
			os.Exit(1)
		}
	}
	return res
}

func readSampleLists(samples []string) []charrnn.SampleList {
	res := make([]charrnn.SampleList, len(samples))
	for i, path := range samples {
		s := charrnn.ReadSampleList(path)
		validation, _ := anysgd.HashSplit(s, Validation)
		res[i] = validation.(charrnn.SampleList)
	}
	return res
}

func rnnNames(rnns []string) []string {
	res := make([]string, len(rnns))
	for i, x := range rnns {
		res[i] = filepath.Base(x)
	}
	return res
}

func computeCost(b anyrnn.Block, samples charrnn.SampleList) float64 {
	total := 0.0
	for i := 0; i < samples.Len(); i += BatchSize {
		samples := samples.Slice(i, essentials.MinInt(samples.Len(), i+BatchSize))
		tr := &anys2s.Trainer{
			Func: func(in anyseq.Seq) anyseq.Seq {
				return anyrnn.Map(in, b)
			},
			Cost: anynet.DotCost{},
		}
		batch, err := tr.Fetch(samples)
		if err != nil {
			essentials.Die(err)
		}
		cost := anyvec.Sum(tr.TotalCost(batch.(*anys2s.Batch)).Output())
		switch cost := cost.(type) {
		case float32:
			total += float64(cost)
		case float64:
			total += cost
		default:
			panic("unsupported numeric type")
		}
	}
	return total / float64(samples.Len())
}

func averageParagraph(samples charrnn.SampleList) float64 {
	sum := 0.0
	for _, x := range samples {
		sum += float64(len(x))
	}
	return sum / float64(samples.Len())
}

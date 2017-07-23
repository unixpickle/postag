package main

import (
	"flag"
	"fmt"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/postag"
	"github.com/unixpickle/wordembed"
)

func main() {
	var inPath string
	var dataPath string
	flag.StringVar(&inPath, "in", "../train/hmm_out", "output file")
	flag.StringVar(&dataPath, "data", "", "path to data")
	flag.Parse()

	if dataPath == "" {
		essentials.Die("Required flag: -data. See -help.")
	}

	data, err := postag.ReadSamples(dataPath, &wordembed.Tokenizer{})
	if err != nil {
		essentials.Die(err)
	}

	model, err := postag.LoadModel(inPath)
	if err != nil {
		essentials.Die(err)
	}

	var totalTokens int
	var correctTokens int
	var totalSequences int
	var workedSequences int

	for _, sample := range data {
		actual := model.Tag(sample.Tokens)
		expected := sample.Tags
		totalTokens += len(expected)
		totalSequences++
		if actual != nil {
			workedSequences++
		}
		for i, a := range actual {
			if expected[i] == a {
				correctTokens++
			}
		}
	}
	fmt.Printf("Got %d/%d for %.02f%% accuracy (%.02f%% of sequences impossible)\n",
		correctTokens, totalTokens,
		100*float64(correctTokens)/float64(totalTokens),
		100*float64(totalSequences-workedSequences)/float64(totalSequences))
}

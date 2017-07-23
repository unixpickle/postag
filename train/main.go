package main

import (
	"flag"
	"log"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/postag"
	"github.com/unixpickle/wordembed"
)

func main() {
	var outPath string
	var dataPath string
	flag.StringVar(&outPath, "out", "hmm_out", "output file")
	flag.StringVar(&dataPath, "data", "", "training data path")
	flag.Parse()

	if dataPath == "" {
		essentials.Die("Required flag: -data. See -help.")
	}

	log.Println("Loading data...")
	data, err := postag.ReadSamples(dataPath, &wordembed.Tokenizer{})
	if err != nil {
		essentials.Die(err)
	}

	log.Println("Training...")
	model := postag.Train(data)

	log.Println("Saving...")
	if err := model.Save(outPath); err != nil {
		essentials.Die(err)
	}
}

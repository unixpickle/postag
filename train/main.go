package main

import (
	"flag"
	"log"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/postag"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/wordembed"
	_ "github.com/unixpickle/wordembed/glove"
)

func main() {
	var embeddingPath string
	var outPath string
	var dataPath string
	flag.StringVar(&embeddingPath, "embedding", "", "GloVe embedding path")
	flag.StringVar(&outPath, "out", "hmm_out", "output file")
	flag.StringVar(&dataPath, "data", "", "training data path")
	flag.Parse()

	if embeddingPath == "" || dataPath == "" {
		essentials.Die("Required flags: -embedding and -data. See -help.")
	}

	log.Println("Loading embedding...")
	var embedding wordembed.Embedding
	if err := serializer.LoadAny(embeddingPath, &embedding); err != nil {
		essentials.Die(err)
	}

	log.Println("Loading data...")
	data, err := postag.ReadSamples(dataPath, &wordembed.Tokenizer{})
	if err != nil {
		essentials.Die(err)
	}

	log.Println("Training...")
	model := postag.Train(data, embedding)

	log.Println("Saving...")
	if err := model.Save(outPath); err != nil {
		essentials.Die(err)
	}
}

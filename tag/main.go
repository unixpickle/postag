package main

import (
	"flag"
	"fmt"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/postag"
	"github.com/unixpickle/wordembed"
	_ "github.com/unixpickle/wordembed/glove"
)

func main() {
	var inPath string
	var phrase string
	flag.StringVar(&inPath, "in", "../train/hmm_out", "output file")
	flag.StringVar(&phrase, "phrase", "", "phrase to tag")
	flag.Parse()

	if phrase == "" {
		essentials.Die("Required flag: -phrase. See -help.")
	}

	model, err := postag.LoadModel(inPath)
	if err != nil {
		essentials.Die(err)
	}

	tokens := (&wordembed.Tokenizer{}).Tokenize(phrase)
	tags := model.Tag(tokens)
	if len(tags) == 0 {
		essentials.Die("impossible token sequence")
	}
	for i, token := range tokens {
		fmt.Println(token, tags[i])
	}
}

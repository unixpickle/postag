package postag

import (
	"bufio"
	"errors"
	"io"
	"os"
	"strings"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/wordembed"
)

// Sample is a sentence from a POS tagging dataset.
type Sample struct {
	Tokens []string
	Tags   []serializer.String
}

// ReadSamples reads the data from a POS dataset.
//
// If a tokenizer is specified, it may be used to split
// up the words into sub-tokens.
func ReadSamples(path string, t *wordembed.Tokenizer) (data []Sample, err error) {
	defer essentials.AddCtxTo("read samples", &err)

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	reader := bufio.NewReader(f)

	var curSample Sample
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		if line == "\n" {
			data = append(data, curSample)
			curSample = Sample{}
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 3 {
			return nil, errors.New("invalid line: " + line)
		}
		tokens := []string{fields[0]}
		if t != nil {
			tokens = t.Tokenize(tokens[0])
		}
		for _, token := range tokens {
			curSample.Tokens = append(curSample.Tokens, token)
			curSample.Tags = append(curSample.Tags, serializer.String(fields[1]))
		}
	}

	return data, nil
}

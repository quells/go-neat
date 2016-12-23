package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	neat "github.com/quells/go-neat"
)

func main() {
	t := time.Now().UTC().UnixNano()
	rand.Seed(t)

	N, T := 50, 500
	fmt.Printf("Generating new population with %d specimens.\n", N)
	p := neat.NewPopulation(3, 1, N)
	fmt.Printf("Starting optimization for %d steps.\n", T)
	p.Optimize(xorEval, T)
	fmt.Println(p.Champion.Genes)

	b := p.Champion
	cases := xorCases()
	for _, c := range cases {
		o, _ := b.ReasonAbout(c.x)
		fmt.Println(c, o)
	}
}

type evalCase struct {
	x []float64
	e float64
}

func xorCases() []evalCase {
	return []evalCase{
		{[]float64{1, 0, 0}, 0},
		{[]float64{1, 0, 1}, 1},
		{[]float64{1, 1, 0}, 1},
		{[]float64{1, 1, 1}, 0},
	}
}

func xorEval(b neat.Brain) float64 {
	cases := xorCases()
	var err float64

	for _, c := range cases {
		o, _ := b.ReasonAbout(c.x)
		err += math.Abs(c.e - o[0])
	}
	return math.Pow(4-err, 2)
}

# go-neat

[NeuroEvolution of Augmenting Topologies](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) (NEAT)

The NEAT algorithm generates neural networks that are not constrained to a fixed structure of nodes and connections.

This implementation is mostly a proof of concept; needs to be highly optimized for production use. Notably, genes and genomes are passed around as strings for debugging purposes.

## Documentation

```golang
type Gene struct{ ... }
    Gene holds information to build a node or connection
```

```golang
type Genome []Gene
    Genome represents the genes for a neural network

func StartingGenome(inputs, outputs int) (Genome, int)
func DecodeGenome(s string) Genome
```

```golang
type Brain struct{ ... }
    Brain represents a neural network

func (b Brain) ReasonAbout(inputs []float64) ([]float64, Brain)

func BuildBrain(genes Genome) Brain
```

```golang
type Population struct{
    Champion *Brain
    ...
}
    Population represents a collection of Genomes that compete to optimize some function.
    Handles random genetic mutation & recombination, fitness calculations, and speciation.

func NewPopulation(inputs, outputs, size int) Population
```

## Usage Example

```golang
import (
    "fmt"
    
    neat "github.com/quells/go-neat"
)

func evaluateSpecimen(b neat.Brain) float64 {
    // Takes 3 inputs, produces 1 output
    ...
}

func main() {
    numSpecimens := 50 // Soft limit on computation requirements
    stepLimit := 500   // Hard limit on iteration count
    
    population := neat.NewPopulation(3, 1, numSpecimens)
    population.Optimize(evaluateSpecimen, stepLimit)
    
    // Logs many status updates
    // Gen 0: 50 specimens in 1 species, 8.71 best score with 4 nodes 3 connections
    // ...
    // Gen 500: 57 specimens in 35 species, 15.96 best score with 6 nodes 8 connections
    
    fmt.Println(population.Champion.Genes)
    
    // n,0,0;n,1,0;n,2,0;n,3,1;c,4,0,3,bff104cecce130fe,1; \
    // c,5,1,3,c015b23c319a5380,1;c,6,2,3,4003ff6a494374a2,0; \
    // c,32,3,3,400c6e86f0d9bee6,1;c,74,1,3,bffc6c4a92abfa3f,0; \
    // n,462,2;c,463,1,4,400a1dafe328ce5a,1;c,464,4,3,3fe0d6497e551b3f,1; \
    // n,588,2;c,589,2,5,3ff0d10b88caab9c,1;c,590,5,3,401251908a79b8ca,1; \
    // c,612,1,5,bff289e75833becd,1;
    
    // ^ output from the xor example, linebreaks added
}
```
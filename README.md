# go-neat

[NeuroEvolution of Augmenting Topologies](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) (NEAT)

The NEAT algorithm generates neural networks that are not constrained to a fixed structure of nodes and connections. This can be useful for developing control systems that can be "trained" to perform some task without making assumptions about how best to perform that task.

This implementation is mostly a proof of concept; needs to be highly optimized for production use. Notably, `Genome` is designed to be passed around as a string for debugging purposes and is frequently encoded and decoded.

## Documentation

A neural network in NEAT is composed of nodes with connections between them. Each node has an accumulator value (`float64`) that is added to by connections from other nodes. Each connection connects two nodes together and has a weight value (`float64`) associated with it. The weight is multiplied by the accumulator of the input node, which is added to the accumulator of the output node. Then, a nonlinear function (sigmoid in this case) is applied to each accumulator. Finally, a decision can be made based on the value of certain pre-selected nodes. This process is performed in `func (b Brain) ReasonAbout`.

Random mutations to a `Genome` include changing connection weights, adding and disabling connections between existing nodes, and adding new nodes. New nodes are added in place of an existing connection to introduce nonlinearity.

```golang
type Gene struct{ ... }
    // Gene holds information to build a node or connection

type Genome []Gene
    // Genome represents the genes for a neural network

func StartingGenome(inputs, outputs int) (Genome, int)
    // StartingGenome produces a Genome with the minimum nodes and connections for a set of inputs and outputs
    // Connection weights are chosen from a uniform random distribution
    // The number of inputs and outputs depends on the fitness evaluation function

func DecodeGenome(s string) Genome
    // DecodeGenome produces a Genome as described by a string
```

```golang
type Brain struct{ ... }
    // Brain represents a neural network

func BuildBrain(genes Genome) Brain
    // BuildBrain returns a Brain instance configured via a 

func (b Brain) ReasonAbout(inputs []float64) ([]float64, Brain)
    // ReasonAbout makes a decision based on some inputs
    // The number of inputs and outputs must remain constant and is determined by the Genome
```

```golang
type Species struct{ ... }
    // Species represents a collection of Brains that share Genome traits
    // A Species will be eliminated if it does not improve after 15 time step

type Population struct{
    Champion *Brain
    ...
}
    // Population represents a collection of Species that compete to optimize some function
    // Handles random genetic mutation & recombination, fitness calculations, and speciation

func NewPopulation(inputs, outputs, size int) Population
    // NewPopulation creates a collection of Brains with the correct number of inputs and outputs to handle a fitness evaluation function

func (p *Population) Optimize(f FitnessEval, numGeneration int)
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
    
    // n,0,0;n,1,0;n,2,0;n,3,1;c,4,0,3,bff104cecce130fe,1...
}
```
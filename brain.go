package neat

import (
	"log"
	"math"
	"strings"
)

// Node represents a neuron in a Brain
type Node struct {
	kind                NodeKind
	accumulator, output float64
}

func (n *Node) clear() {
	n.accumulator = 0
}

func nonLinear(x float64) float64 {
	return 1 / (1 + math.Exp(-5*x))
}

func (n *Node) activate() {
	switch {
	case n.kind == sensorNode:
		return
	default:
		n.output = nonLinear(n.accumulator)
	}
}

// Connection represents a synapse between nodes in a Brain
type Connection struct {
	from, to int
	weight   float64
}

// Brain represents a neural network
type Brain struct {
	nodes           []Node
	connections     []Connection
	Genes           string
	Inputs, Outputs int
	fitness         float64
}

// BuildBrain builds a neural network as described by a Genome
func BuildBrain(genes Genome) Brain {
	nodes, connections := []Node{}, []Connection{}
	var inputs, outputs int
	for _, g := range genes {
		if strings.HasPrefix(g.payload, "n") {
			nodeGene := decodeNodeGene(g.payload)
			node := Node{nodeGene.kind, 0, 0}
			nodes = append(nodes, node)
			if node.kind == sensorNode {
				inputs++
			} else if node.kind == outputNode {
				outputs++
			}
		} else if strings.HasPrefix(g.payload, "c") {
			connGene := decodeConnectionGene(g.payload)
			if connGene.enabled {
				conn := Connection{connGene.from, connGene.to, connGene.weight}
				connections = append(connections, conn)
			}
		} else {
			log.Fatalf("Unknown gene signature: %s", g.payload)
		}
	}

	return Brain{nodes, connections, genes.encode(), inputs, outputs, 0}
}

// ReasonAbout loads an input vector, runs it through the Brain, and returns the output
func (b Brain) ReasonAbout(inputs []float64) ([]float64, Brain) {
	// Clear nodes
	for i := range b.nodes {
		b.nodes[i].clear()
	}

	// Load inputs
	for i := 0; i < b.Inputs; i++ {
		b.nodes[i].output = inputs[i]
	}

	// Apply connections
	for _, c := range b.connections {
		x := b.nodes[c.from].output
		b.nodes[c.to].accumulator += x * c.weight
	}

	// Activate nodes
	for i := range b.nodes {
		b.nodes[i].activate()
	}

	// Read outputs
	output := make([]float64, b.Outputs)
	for o := 0; o < b.Outputs; o++ {
		output[o] = b.nodes[o+b.Inputs].output
	}

	return output, b
}

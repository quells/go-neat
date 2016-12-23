package neat

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
)

// Gene holds information to build a node or connection
type Gene struct {
	mutationID int
	payload    string
}

type byMutationID []Gene

func (b byMutationID) Len() int           { return len(b) }
func (b byMutationID) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byMutationID) Less(i, j int) bool { return b[i].mutationID < b[j].mutationID }

// NodeKind is an enum of the possible kinds of nodes
type NodeKind int

const (
	sensorNode NodeKind = iota
	outputNode
	hiddenNode
)

type nodeGene struct {
	mutationID int
	kind       NodeKind
}

func (g nodeGene) encode() string {
	return fmt.Sprintf("n,%d,%d;", g.mutationID, g.kind)
}

func decodeNodeGene(s string) nodeGene {
	components := strings.Split(strings.Split(s, ";")[0], ",")
	id, _ := strconv.Atoi(components[1])
	kind, _ := strconv.Atoi(components[2])
	return nodeGene{id, NodeKind(kind)}
}

type connectionGene struct {
	mutationID int
	from, to   int
	weight     float64
	enabled    bool
}

func (g connectionGene) encode() string {
	var enabledBit int
	if g.enabled {
		enabledBit = 1
	}
	return fmt.Sprintf("c,%d,%d,%d,%x,%d;", g.mutationID, g.from, g.to, math.Float64bits(g.weight), enabledBit)
}

func decodeConnectionGene(s string) connectionGene {
	components := strings.Split(strings.Split(s, ";")[0], ",")

	id, _ := strconv.Atoi(components[1])
	from, _ := strconv.Atoi(components[2])
	to, _ := strconv.Atoi(components[3])

	b, _ := hex.DecodeString(components[4])
	var w float64
	binary.Read(bytes.NewBuffer(b), binary.BigEndian, &w)

	enabled := false
	if strings.Contains(components[5], "1") {
		enabled = true
	}

	return connectionGene{id, from, to, w, enabled}
}

// Genome represents the genes for a neural net
type Genome []Gene

func (G Genome) encode() string {
	var s string
	for _, gene := range G {
		if !strings.HasSuffix(gene.payload, ";") {
			gene.payload += ";"
		}
		s += gene.payload
	}
	return s
}

func DecodeGenome(s string) Genome {
	genes := strings.Split(s, ";")
	genes = genes[:len(genes)-1]

	extractMutID := func(s string) int {
		components := strings.Split(s, ",")
		id, _ := strconv.Atoi(components[1])
		return id
	}

	G := []Gene{}
	for _, g := range genes {
		G = append(G, Gene{extractMutID(g), g})
	}

	return Genome(G)
}

func encodeGenes(nodes []nodeGene, conns []connectionGene) Genome {
	G := []Gene{}
	for _, node := range nodes {
		g := Gene{node.mutationID, node.encode()}
		G = append(G, g)
	}
	for _, conn := range conns {
		g := Gene{conn.mutationID, conn.encode()}
		G = append(G, g)
	}
	sort.Sort(byMutationID(G))
	return Genome(G)
}

func decodeGenes(G Genome) ([]nodeGene, []connectionGene) {
	nodes, conns := []nodeGene{}, []connectionGene{}
	for _, g := range G {
		if strings.HasPrefix(g.payload, "n") {
			node := decodeNodeGene(g.payload)
			nodes = append(nodes, node)
		} else if strings.HasPrefix(g.payload, "c") {
			conn := decodeConnectionGene(g.payload)
			conns = append(conns, conn)
		}
	}
	return nodes, conns
}

// StartingGenome produces a Genome with the minimum nodes and connections for a set of inputs and outputs
func StartingGenome(inputs, outputs int) (Genome, int) {
	G := []Gene{}
	var mutID int
	for i := 0; i < inputs; i++ {
		node := nodeGene{mutID, sensorNode}
		gene := Gene{mutID, node.encode()}
		G = append(G, gene)
		mutID++
	}
	for i := 0; i < outputs; i++ {
		node := nodeGene{mutID, outputNode}
		gene := Gene{mutID, node.encode()}
		G = append(G, gene)
		mutID++
	}

	for i := 0; i < inputs; i++ {
		for j := 0; j < outputs; j++ {
			w := uniform(-2, 2)
			conn := connectionGene{mutID, i, inputs + j, w, true}
			gene := Gene{mutID, conn.encode()}
			G = append(G, gene)
			mutID++
		}
	}

	return Genome(G), mutID
}

// MutationKind is an enum of the possible kinds of mutation
type MutationKind int

const (
	addNodeMutation MutationKind = iota
	addConnectionMutation
)

// Mutation holds information about a mutation that occurred this generation
type Mutation struct {
	mutationID int
	kind       MutationKind
	from, to   int
}

// Mutate alters an existing Genome to simulation random mutations
func (G Genome) Mutate(nextID int, mtg []Mutation) (Genome, int, []Mutation) {
	nodes, conns := decodeGenes(G)

	if rand.Float64() < 0.8 {
		conns = mutateConnectionWeights(conns)
	}

	if rand.Float64() < 0.03 {
		old, a, n, b, ok := mutateAddNode(nextID, nodes, conns)
		if ok {
			found := false
			for _, m := range mtg {
				if m.kind == addNodeMutation && old.from == m.from && old.to == m.to {
					n.mutationID = m.mutationID
					a.mutationID = m.mutationID + 1
					b.mutationID = m.mutationID + 2
					found = true
					break
				}
			}
			if !found {
				mtg = append(mtg, Mutation{n.mutationID, addNodeMutation, old.from, old.to})
				nextID += 3
			}
			for i, c := range conns {
				if c.mutationID == old.mutationID {
					conns[i] = old
				}
			}
			nodes = append(nodes, n)
			conns = append(conns, a, b)
		}
	}

	if rand.Float64() < 0.05 {
		newConn, ok := mutateAddConnection(nextID, nodes, conns)
		if ok {
			found := false
			for _, m := range mtg {
				if m.kind == addConnectionMutation && newConn.from == m.from && newConn.to == m.to {
					newConn.mutationID = m.mutationID
					found = true
					break
				}
			}
			if !found {
				mtg = append(mtg, Mutation{newConn.mutationID, addConnectionMutation, newConn.from, newConn.to})
				nextID++
			}
			conns = append(conns, newConn)
		}
	}

	G = encodeGenes(nodes, conns)
	return G, nextID, mtg
}

func mutateConnectionWeights(conns []connectionGene) []connectionGene {
	for i, c := range conns {
		if rand.Float64() < 0.9 {
			conns[i].weight = c.weight + uniform(-0.5, 0.5)
		}
	}
	return conns
}

func mutateAddConnection(nextID int, nodes []nodeGene, conns []connectionGene) (connectionGene, bool) {
	var inputs int
	for _, n := range nodes {
		if n.kind == sensorNode {
			inputs++
		}
	}
search:
	for try := 0; try < 10; try++ {
		i := rand.Intn(len(nodes))
		j := rand.Intn(len(nodes)-inputs) + inputs
		for _, c := range conns {
			if c.from == i && c.to == j {
				continue search
			}
			w := uniform(-2, 2)
			return connectionGene{nextID, i, j, w, true}, true
		}
	}
	return connectionGene{}, false
}

func mutateAddNode(nextID int, nodes []nodeGene, conns []connectionGene) (connectionGene, connectionGene, nodeGene, connectionGene, bool) {
	// old, newA, newNode, newB
search:
	for try := 0; try < 10; try++ {
		idx := rand.Intn(len(conns))
		conn := conns[idx]
		if !conn.enabled {
			continue search
		}

		i := conn.from
		j := len(nodes)
		k := conn.to

		conn.enabled = false
		n := nodeGene{nextID, hiddenNode}
		a := connectionGene{nextID + 1, i, j, 1, true}
		b := connectionGene{nextID + 2, j, k, conn.weight, true}

		return conn, a, n, b, true
	}
	return connectionGene{}, connectionGene{}, nodeGene{}, connectionGene{}, false
}

func genomeMismatch(a, b Genome) (float64, float64, float64, float64, float64) {
	A, B := float64(len(a)), float64(len(b))
	N := math.Max(A, B)
	var m, d, e, w float64
	for i := 0; i < int(N); i++ {
		switch {
		case i < len(a) && i < len(b):
			l, r := a[i], b[i]
			if l.mutationID == r.mutationID {
				m++
				if strings.HasPrefix(l.payload, "c") && strings.HasPrefix(r.payload, "c") {
					L, R := decodeConnectionGene(l.payload), decodeConnectionGene(r.payload)
					w += math.Abs(L.weight - R.weight)
				}
			} else {
				d++
			}
		default:
			e++
			var g Gene
			if len(a) > len(b) {
				g = a[i]
			} else {
				g = b[i]
			}
			if strings.HasPrefix(g.payload, "c") {
				w += math.Abs(decodeConnectionGene(g.payload).weight)
			}
		}
	}
	if N < 20 {
		N = 1
	}
	return N, m, d, e, w
}

// CompatibilityDistance is used to differentiate between different species
func CompatibilityDistance(a, b Genome) float64 {
	N, _, d, e, w := genomeMismatch(a, b)
	return (d/N + e/N + 0.4*w)
}

// Sharing calculates whether two Genomes are in the same species
func Sharing(a, b Genome) float64 {
	if CompatibilityDistance(a, b) < 3 {
		return 1
	}
	return 0
}

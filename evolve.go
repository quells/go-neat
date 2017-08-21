package neat

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// Species represents a collection of Brains that share Genome traits
type Species struct {
	members                []Brain
	champion               *Brain
	sharedFitness          float64
	timeWithoutImprovement int
}

// Population represents a collection of Species that compete to optimize some function
type Population struct {
	species  []Species
	Champion *Brain
	nextID   int
}

// FitnessEval is used to evaluate how well a brain can solve a problem
type FitnessEval func(Brain) float64

// NewPopulation creates a collection of Brains with the correct number of inputs and outputs to handle a FitnessEval
func NewPopulation(inputs, outputs, size int) Population {
	members := make([]Brain, size)
	_, nextID := StartingGenome(inputs, outputs)
	for i := range members {
		genes, _ := StartingGenome(inputs, outputs)
		members[i] = BuildBrain(genes)
	}

	species := []Species{Species{members, nil, 0, 0}}

	return Population{species, nil, nextID}
}

func (p Population) size() int {
	N := 0
	for _, s := range p.species {
		N += len(s.members)
	}
	return N
}

// Optimize uses a FitnessEval to identify low-performing candidate Genomes and replace them with new ones based on well-performing candidates
func (p *Population) Optimize(f FitnessEval, numGenerations int) {
	N := p.size()

	p.updateFitnesses(f)
	n, c := p.Champion.nodes, p.Champion.connections
	fmt.Printf("Gen %d: %d specimens in %d species, %.2f best score with %d nodes %d connections\n", 0, p.size(), len(p.species), p.Champion.fitness, len(n), len(c))

	for t := 1; t <= numGenerations; t++ {
		// Should a species go extinct?
		for i := len(p.species) - 1; i >= 0; i-- {
			tooLongWithoutImprovement := p.species[i].timeWithoutImprovement >= 15
			if tooLongWithoutImprovement {
				if i == len(p.species)-1 {
					p.species = p.species[:i]
				} else {
					copy := make([]Species, len(p.species)-2)
					copy = append(p.species[:i], p.species[i+1:]...)
					p.species = copy
				}
			}
		}

		// Update fitness sharing
		for i, s := range p.species {
			s.sharedFitness = 0
			for _, b := range s.members {
				g := DecodeGenome(b.Genes)
				var denom float64
				for _, m := range p.species {
					for _, n := range m.members {
						denom += Sharing(g, DecodeGenome(n.Genes))
					}
				}
				s.sharedFitness += b.fitness / denom
			}
			p.species[i] = s
		}

		// How many offspring should each species get?
		var sumAllFitnesses float64
		for _, s := range p.species {
			sumAllFitnesses += s.sharedFitness
		}
		nextGenerationCounts := make([]int, len(p.species))
		for i, s := range p.species {
			count := s.sharedFitness / sumAllFitnesses * float64(N)
			nextGenerationCounts[i] = int(math.Floor(count + 0.5))
		}
		// fmt.Println(nextGenerationCounts)

		// Cull species populations
		for i, s := range p.species {
			N := len(s.members)
			// if N/2 > 1 {
			// 	copy := make([]Brain, N/2)
			// 	copy = s.members[:N/2]
			// 	s.members = copy
			// }

			for j := N - 1; j > 0; j-- {
				if rand.Float64() < tanhCutoff(j, N) {
					if j == len(s.members)-1 {
						s.members = s.members[:j]
					} else {
						copy := make([]Brain, len(s.members)-2)
						copy = append(s.members[:j], s.members[j+1:]...)
						s.members = copy
					}
				}
			}
			p.species[i] = s
		}

		// Breed species populations
		for i, s := range p.species {
			cap := nextGenerationCounts[i]
			mtg := []Mutation{}
			numParents := len(s.members)
			for {
				if len(s.members) >= cap {
					break
				}
				// Asexual reproduction
				if rand.Float64() < 0.25 {
					parent := DecodeGenome(s.members[rand.Intn(numParents)].Genes)
					var offspring Genome
					offspring, p.nextID, mtg = parent.Mutate(p.nextID, mtg)
					s.members = append(s.members, BuildBrain(offspring))
				} else {
					// Sexual reproduction
					mb, fb := s.members[rand.Intn(numParents)], s.members[rand.Intn(numParents)]
					mg, fg := DecodeGenome(mb.Genes), DecodeGenome(fb.Genes)
					_, matchCount, _, _, _ := genomeMismatch(mg, fg)
					mGenes, fGenes := make([]Gene, int(matchCount)), make([]Gene, int(matchCount))
					for j := 0; j < int(matchCount); j++ {
						// if rand.Float64() < 0.01 {
						// 	if strings.HasPrefix(mg[j].payload, "c") {
						// 		temp := decodeConnectionGene(mg[j].payload)
						// 		temp.enabled = true
						// 		mg[j].payload = temp.encode()
						// 	}
						// 	if strings.HasPrefix(fg[j].payload, "c") {
						// 		temp := decodeConnectionGene(fg[j].payload)
						// 		temp.enabled = true
						// 		fg[j].payload = temp.encode()
						// 	}
						// }
						mGenes[j], fGenes[j] = mg[j], fg[j]
					}
					var otherGenes []Gene
					switch {
					case mb.fitness > fb.fitness && len(mg) > len(fg):
						otherGenes = mg[int(matchCount):]
					case fb.fitness > mb.fitness && len(fg) > len(mg):
						otherGenes = fg[int(matchCount):]
					default:
						otherGenes = []Gene{}
					}
					mGenes, fGenes = append(mGenes, otherGenes...), append(fGenes, otherGenes...)
					aGenes, bGenes := make([]Gene, len(mGenes)), make([]Gene, len(mGenes))
					for j := range aGenes {
						var left, right Gene
						if rand.Float64() < 0.5 {
							left, right = mGenes[j], fGenes[j]
						} else {
							right, left = mGenes[j], fGenes[j]
						}
						aGenes[j], bGenes[j] = left, right
					}
					var aGenome, bGenome Genome
					aGenome, p.nextID, mtg = Genome(aGenes).Mutate(p.nextID, mtg)
					bGenome, p.nextID, mtg = Genome(bGenes).Mutate(p.nextID, mtg)
					s.members = append(s.members, BuildBrain(aGenome), BuildBrain(bGenome))
				}
			}
			p.species[i] = s
		}

		p.updateSpeciation()
		p.updateFitnesses(f)
		// sort.Sort(byChampFitness(p.species))

		n, c := p.Champion.nodes, p.Champion.connections
		fmt.Printf("Gen %d: %d specimens in %d species, %.2f best score with %d nodes %d connections\n", t, p.size(), len(p.species), p.Champion.fitness, len(n), len(c))
		if t%50 == 0 {
			fmt.Println(p.Champion.Genes)
		}
	}
}

type byChampFitness []Species

func (b byChampFitness) Len() int           { return len(b) }
func (b byChampFitness) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byChampFitness) Less(i, j int) bool { return b[i].champion.fitness > b[j].champion.fitness }

type byFitness []Brain

func (b byFitness) Len() int           { return len(b) }
func (b byFitness) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byFitness) Less(i, j int) bool { return b[i].fitness > b[j].fitness }

func (p *Population) updateFitnesses(f FitnessEval) {
	for i, s := range p.species {
		var prevBestFitness float64
		if s.champion == nil {
			prevBestFitness = 0
		} else {
			prevBestFitness = s.champion.fitness
		}

		for j, b := range s.members {
			s.members[j].fitness = f(b)
		}
		sort.Sort(byFitness(s.members))

		for j := range s.members {
			if s.champion == nil {
				s.champion = &s.members[j]
			} else if s.members[j].fitness > s.champion.fitness {
				s.champion = &s.members[j]
			}
		}

		if p.Champion == nil {
			p.Champion = s.champion
		} else if s.champion.fitness > p.Champion.fitness {
			p.Champion = s.champion
		}

		if s.champion.fitness > prevBestFitness {
			s.timeWithoutImprovement = 0
		} else {
			s.timeWithoutImprovement++
		}

		p.species[i] = s
	}
}

func (p *Population) updateSpeciation() {
	newSpecies := []Species{}
	for i, s := range p.species {
		startSize := len(s.members)
		for j, b := range s.members {
			j = startSize - 1 - j
			g := DecodeGenome(s.members[j].Genes)
			delta := CompatibilityDistance(DecodeGenome(s.champion.Genes), g)
			if delta > 3 {
				foundMatchingSpecies := false
			checkNewSpecies:
				for k, ns := range newSpecies {
					delta = CompatibilityDistance(g, DecodeGenome(ns.members[0].Genes))
					if delta < 3 {
						newSpecies[k].members = append(newSpecies[k].members, b)
						foundMatchingSpecies = true
						break checkNewSpecies
					}
				}
				if !foundMatchingSpecies {
					if len(s.members) > 1 {
						ns := Species{[]Brain{b}, nil, 0, 0}
						updatedMembers := make([]Brain, len(s.members)-2)
						if j == len(s.members)-1 {
							updatedMembers = s.members[:j]
						} else {
							updatedMembers = append(s.members[:j], s.members[j+1:]...)
						}
						s.members = updatedMembers
						newSpecies = append(newSpecies, ns)
					}
				}
			}
			p.species[i] = s
		}
	}
	p.species = append(p.species, newSpecies...)
}

func tanhCutoff(i, N int) float64 {
	x, n := float64(i), float64(N)
	m := 5.0
	return 0.5 * (1 + math.Tanh(2*m*x/n-m))
}

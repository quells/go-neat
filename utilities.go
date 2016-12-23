package neat

import "math/rand"

func uniform(lo, hi float64) float64 {
	d := hi - lo
	return lo + d*rand.Float64()
}

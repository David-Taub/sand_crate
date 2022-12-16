
from nptyping import NDArray

Particles = NDArray  # num_of_particles x 2 (X Y)
ParticlesVelocity = NDArray  # num_of_particles x 2 (X Y)
Distances = NDArray  # num_of_particles x 2 (X Y)
ParticlesNeighbors = list[int]

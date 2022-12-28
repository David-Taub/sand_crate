from nptyping import NDArray

Particles = NDArray  # particles x 2 (X Y)
Colliders = NDArray  # particles x 2 (X Y), relative to particle
Velocities = NDArray  # particles x 2 (X Y)
Points = NDArray  # points x 2 (X Y)
Vectors = NDArray  # vectors x 2 (X Y)
Segments = NDArray  # # segments x dots(2) x 2 (X Y)
ParticlesVelocity = NDArray  # num_of_particles x 2 (X Y)
Distances = NDArray  # num_of_particles x 2 (X Y)
ParticlesNeighbors = list[int]

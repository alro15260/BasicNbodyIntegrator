import matplotlib.pyplot as plt
import numpy as np
import updateParticles function from leapfrog.py

def getForce(mi, mj, sep):
    """
    Compute magnitude of gravitational force between two bodies.
    
    Parameters
    ----------
    mi, mj : float
        Particle masses in kg.
    sep : float
        Particle separation (distance between bodies) in m.
        
    Returns
    -------
    force : float
        Gravitational force between bodies in N.
    
    Example
    -------
    >>> # appx. force between Earth and 70 kg person on surface
    >>> mEarth = 6e24
    >>> mPerson = 70
    >>> radiusEarth = 6380000 
    >>> getForce(mEarth, mPerson, radiusEarth)
    688.2302650327729
    """
    G = 6.67e-11                # m3 kg-1 s-2
    return G * mi * mj / sep**2 # N

def magnitude(vec):
    """
    Compute magnitude of any vector with an arbitrary number of elements.
    """
    return np.sqrt(np.sum(vec**2))

def unitVec(vec):
    """
    Create unit vector from a vector with any number of elements.
    """
    # divide vector components by vector magnitude to make unit vector
    mag = magnitude(vec)
    return vec/mag

def getSepVec(pos_i, pos_j):
    """
    Compute separation vector from i to j.
    
    Parameters
    ----------
    pos_i, pos_j : numpy arrays
        Particle positions as 3-element arrays [x, y, z].
        Note that as written, the code on N-D arrays.
        
    Returns
    -------
    sepVec : numpy array
        Separation vector, also as a 3-element array.
        
    Example
    -------
    >>> getSepVec(np.array([-1, -2, -3]), np.array([1, 2, 3])
    [2, 4, 6]
    
    """
    
    # subtract components to get separation vector
    return pos_j - pos_i

def getForceVec(mi, mj, pos_i, pos_j):
    """
    Compute gravitational force vector exerted on particle i by particle j.
    
    Parameters
    ----------
    mi, mj : float
        Particle masses in kg.
    pos_i, pos_j : numpy arrays
        Particle positions in cartesian coordinates in m.
        
    Returns
    -------
    forceVec : list
        Components of gravitational force vector in N.
        
    Example
    -------
    >>> getForceVec(1e10, 1e10, [0, 0, 0], [1, 1, 1])
    [1283.6420984982683, 1283.6420984982683, 1283.6420984982683]
    
    """
    
    # compute magnitude of the force
    sepvec = getSepVec(pos_i, pos_j)            # [m]
    force = getForce(mi, mj, magnitude(sepvec)) # [N]
    
    # get the components of a unit vector in the
    # force direction
    unit_sepvec = unitVec(sepvec)
    
    # return the force as a vector
    return force*unit_sepvec # [N]

def netForces(masses, positions):
    """
    Compute net gravitational force vector exerted on 
    particle i by all particles j, and do this for all i.
    
    Parameters
    ----------
    masses : list of floats
        Particle masses in kg.
    positions : list of numpy arrays
        Particle positions in cartesian coordinates in m.
        
    Returns
    -------
    netForces : list of numpy arrays
        Components of gravitational force vectors in N.
        
    Example
    -------
    >>> # forces between Earth and 70 kg person on surface
    >>> masses = [6e24,70]
    >>> radiusEarth = 6380000 # m
    >>> positions = [np.array([0, 0, 0])*radiusEarth, np.array([1, 1, 1])*radiusEarth]
    >>> netForces(masses,positions)
    [array([   0.        ,    0.        ,  688.23026503]),
     array([   0.        ,    0.        , -688.23026503])]
    
    """
    # empty force list
    force_list = []
    
    # for each body i
    for i in range(len(masses)):

        # zero net force vector; add to this for each j
        net_force_i = np.zeros(3)

        # vector sum all forces on i by bodies j
        for j in range(len(masses)):
            # omit self-interactions (body i on itself):
            if i != j:  
                # vector sum of net force with additional force from body j
                net_force_i += getForceVec(masses[i],masses[j],positions[i],positions[j])

        # add net force to force list:
        force_list.append(net_force_i)
        
    return force_list

def updateParticles(masses, positions, velocities, dt):
    """
    Evolve particles in time via leap-frog integrator scheme. This function
    takes masses, positions, velocities, and a time step dt as

    Parameters
    ----------
    masses : np.ndarray
        1-D array containing masses for all particles, in kg
        It has length N, where N is the number of particles.
    positions : np.ndarray
        2-D array containing (x, y, z) positions for all particles.
        Shape is (N, 3) where N is the number of particles.
    velocities : np.ndarray
        2-D array containing (x, y, z) velocities for all particles.
        Shape is (N, 3) where N is the number of particles.
    dt : float
        Evolve system for time dt (in seconds).

    Returns
    -------
    Updated particle positions and particle velocities, each being a 2-D
    array with shape (N, 3), where N is the number of particles.

    """

    startingPositions = np.array(positions)
    startingVelocities = np.array(velocities)

    # how many particles are there?
    nParticles, nDimensions = startingPositions.shape

    # make sure the three input arrays have consistent shapes
    assert(startingVelocities.shape == startingPositions.shape)
    assert(len(masses) == nParticles)

    # calculate net force vectors on all particles, at the starting position
    startingForces = np.array(netForces(masses, startingPositions))

    # calculate the acceleration due to gravity, at the starting position
    startingAccelerations = startingForces/np.array(masses).reshape(nParticles, 1)

    # calculate the ending position
    nudge = startingVelocities*dt + 0.5*startingAccelerations*dt**2
    endingPositions = startingPositions + nudge

    # calculate net force vectors on all particles, at the ending position
    endingForces = np.array(netForces(masses, endingPositions))

    # calculate the acceleration due to gravity, at the ending position
    endingAccelerations = endingForces/np.array(masses).reshape(nParticles, 1)

    # calculate the ending velocity
    endingVelocities = (startingVelocities +
                        0.5*(endingAccelerations + startingAccelerations)*dt)

    return endingPositions, endingVelocities

# the masses in the list are in kg
testmasses = np.array([1, 1e24])
au = 1.496e+11  # m per AU
# the position vectors in this list are in meters
testpositions = np.array([[ 0,  0,  6731*1000], [0, 0 , 0]])
#the velocity vectors in this list are in m/s
testvelocities = np.array([[0, 0, 0], [0, 0, 0]])
print(testvelocities.shape)
updateParticles(testmasses, testpositions, testvelocities, dt = 1)

def calculateTrajectories(initialMasses, initialPositions, initialVelocities, time, dt):
    '''
    This function is a more general version of the updateParticles function.  This function
    evolves a system over a certain time interval, dt, with step size, t_step, and evolves the 
    system based off of the net gravitational force vector on each body due to each of the other
    bodies in the system.  The net force vector is used to calculate the net acceleration on a
    given body.  From there, one can predict the resulting equations of motion.
    
    Parameters
    ----------
    masses: a 1D array containing the masses of the particles, with nDimensions elements
    initialpositions: a 2D array containing the cooresponding positions of the particles, with
    nParticles * nDimensions elements
    initialvelocities: a 2D array containing the cooresponding velocities of the particles, with
    nParticles * nDimensions elements
    T: a float containing the total time desired to evolve the system
    dt: a float for the size of each time step, in seconds
    
    Returns
    -------
    times: a 1D array for containing the time values with nTimes elements
    positionsatalltimes: a 3D array with the positions at each time containing 
    nParticles * nDimensions * nTimes elements
    velocitiesatalltimes: a 3D array with the velocities at each time containing 
    nParticles * nDimensions * nTimes elements
    '''
    positions = [initialPositions]
    velocities = [initialVelocities]
    times=np.arange(0, time+dt, dt)
    for i in range(len(times[:-1])):
        values = updateParticles(initialMasses, initialPositions, initialVelocities, dt)
        positions.append(values[0])
        velocities.append(values[1])
        initialPositions = values[0]
        initialVelocities = values[1]
    return times, np.array(positions), np.array(velocities)


**Vector field**
Vector field is an assignment of vector to every point in space.

**Electromagnetic field**
The most common description of the electromagnetic field uses two three-dimensional vector fields called the electric field and the magnetic field. These vector fields each have a value defined at every point of space and time and are thus often regarded as functions of the space and time coordinates. As such, they are often written as E(x, y, z, t) (electric field) and B(x, y, z, t) (magnetic field).

Meaning we can write: 
E:R_s^3×R_t⟶R_v^3
Where
 R_s is the set of Real number triples of spatial coordinates: x,y,z 
 R_s is the set of Real number scalars of time
 R_v^3 is the set of Real number vectors, defined on R_s^3, R_t.
 

**Dynamic and static**:
If only the electric field (E) is non-zero, and is constant in time, the field is said to be an electrostatic field. Similarly, if only the magnetic field (B) is non-zero and is constant in time, the field is said to be a magnetostatic field. 

However, if either the electric or magnetic field has a time-dependence, then both fields must be considered together as a coupled electromagnetic field using Maxwell's equations.


**Lorentz force**
In electromagnetism, the Lorentz force is the force exerted on a charged particle by electric and magnetic fields. 

*Point particle*
Lorentz force F on a charged particle (of charge q) in motion (instantaneous velocity v). The E field and B field vary in space and time.
The Lorentz force F acting on a point particle with electric charge q, moving with velocity v, due to an external electric field E and magnetic field B, is given by:

**F** = q(**E** + **v** × **B**)

Here, × is the vector cross product, and all quantities in bold are vectors. 


So now we now that electric and magnetic fields are vector fields and through them, force is exerted on a particle moving at v, with charge q. 

Now we need to ask, does particle influence the field as well? 


**Sources of fields**
Particles are sources of fields.Source terms in Maxwell’s equations depend on the distribution and motion of charges: 

*Charge density* ρ(r,t) creates electric fields (Gauss’s Law)

 ∇⋅E=​ρ​/ε0

*Current density* J(r,t) creates magnetic fields (Ampère–Maxwell law)

∇×B=μ0​J + μ0​ε0∂E​/​∂t


So we can say: 
Given fields → particle motion (via Lorentz force).

Given particle motion → fields (via Maxwell’s equations with sources).

*Small problem*
But wait. Particle influences the field instantaneously, but is also dependent on the field to determine it’s next movement. 

So how does the particle know where to move, when at the same time, using the same coordinates, it influences the field?

For that, we see all change propagates at the speed of light.

**Propagation of change**

Mathematically, the fields of a moving particle are given by the *Liénard–Wiechert* potentials, which depend on the particle’s position and velocity at an earlier “retarded time”:



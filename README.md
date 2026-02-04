# apma4302_fbg2107
Spring2026 METHODS IN COMPUTATIONAL SCI: https://github.com/mspieg/apma4302-methods

# PetSc

```
export PETSC_ARCH=name_of_installation
export PETSC_DIR=/path/to/petsc
```

# 01/29/2026
just initializes PETSc
```
PetscInitialize 
```

rank is the processor number (0 to N-1)
mpiexec -n is n is the number of processes

not necesarily the same as number of cores

## HW 1 Notes
want to do $e_N^x$ which is the Nth order approximation using taylor series
this is  
$$e_N^x = \sum_{n=0}^N \frac{x^n}{n!}$$

$= 1 + x + x^2/2! + \cdots + x^N/N!$

do two things:

1. pass in two things from the command line: x and N over P processes

this takes the calclation and breaks it up over P processes
this should return the same result as just calculating it on one process


we can do Horner's rule, taking the term
$$ 1+ x + x^2/2 + x^3/3! $$

we can re write this using the rule 
$$P(x) = a_0 + x(a_1 + x(a_2 + x(a_3 + \cdots + x(a_{N-1} + x(a_N)))))$$

so we can rewrite the taylor series as
$$ e_N^x = 1 + \frac{x}{1}(1 + \frac{x}{2}(1 + \frac{x}{3}(1 + \cdots (1 + \frac{x}{N}))))$$

Ideally it would be N/P terms per process, but this scales as N 

### to pass in information from command line
we  can use  `ch1>solns>expx.c`

and do `$ mpiexec -np 12 expx -x -1`

make sure you can calculate this from -709 to 709 
also, return the relative error to  machine epsilon,
where machine epsilon is the smallest number such that
1 + eps != 1
we can find this using the following code snippet:

```
double eps = 1.0;
while (1.0 + eps / 2.0 != 1.0) {
    eps /= 2.0;
}
printf("Machine epsilon: %e\n", eps);
```


# 02/03/2026

- git add tag for final submission
- do survery
- for the hw1, make sure to include relative error calculation

## HPC for linear algebra
1. motivate 2pt BVP
2. Petsc, VEC, MAT
3. Direct methodsL LU, SparseLU
4. Iterative methods: Preconditioned (PC objects) Krylov Subspace (KSPs) methods


### Linear 2 pt BVP
$$ \dfrac{- d^2 u}{d x^2} + \gamma u = f(x) $$

for some $x \in [a,b]$
with boundary conditions $u(a) = u_a$, $u(b) = u_b$

we are apprximating $u(x) \sim u \in R^{N+1}$



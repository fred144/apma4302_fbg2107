# HW1

We want want to compute $e^x$ using its taylor series expansion over N terms. We know that the taylor series for $e^x$ is given by
$$e^x = \sum_{n=0}^\infty \frac{x^N}{n!} = 1 + \frac{x^1}{1!} + \frac{x^2}{2!} + \cdots + \frac{x^N}{N!} $$

The code we are given `expx.c` computes this sum in parallel using MPI and PETSc. 
The idea is to split the computation of the series terms across multiple processes. 
Each process computes a subset of the terms and then we combine the results.
Particically, this looks like:
```c
// each process computes its local contribution
localval = 1.0;
for (i = 1; i < rank + 1; i++)
    localval *= x / i;

// sum the contributions over all processes
PetscCall(MPI_Allreduce(&localval, &globalsum, 1, MPIU_REAL, MPIU_SUM,
                        PETSC_COMM_WORLD));
```
however, a limitation is that the work of rank 0 is only computing the first term of the series, rank 1 computes the second term, etc. Such that rank N has to compute N terms. This is unbalanced work. In addition, the complexity of having to multiply x/i for each term is O(N^2) since rank N has to do N multiplications.

To improve this, we can use Horner's rule to rewrite the taylor series in a nested form:
$$ e_N^x = 1 + \frac{x}{1}(1 + \frac{x}{2}(1 + \frac{x}{3}(1 + \cdots (1 + \frac{x}{N}))))$$
and then be able to splid N terms over P processes, ideally N/P terms per process. This would reduce the complexity to O(N) since each process would only have to do N/P multiplications per process. 

The Horner's rule is nested, so we gotta be careful about combingi the results from each process. 

Each process will compute its local contribution using Horner's rule for its assigned terms, and then we will need to combine these contributions.

For the sake of me understanding this, we have
$N = 9$ so 9 terms in total
$P = 3$ so 3 processes/ranks 
which we can evenly split so as to have each process compute 3 terms: 
$$e^x_9 = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + a_4 x^4 + a_5 x^5 + a_6 x^6 + a_7 x^7 + a_8 x^8 + a_9 x^9$$
where
$a_n = \frac{1}{n!}$ 

we can split into blocks of 3 terms each:
```c
Block 0 (rank 0): a0 + a1 x + a2 x^2 + 
Block 1 (rank 1): a3 + a4 x + a5 x^2 +  <-- starts at x^3
Block 2 (rank 2): a6 + a7 x + a8 x^2 +   <-- starts at x^6
```
then do the Horners in each block 
```c
Block 0 (rank 0): a2 + x(a1 + x(a0))
Block 1 (rank 1): a5 + x(a4 + x(a3))
Block 2 (rank 2): a8 + x(a7 + x(a6))
```
then we need to multiply the power to shift the blocks (according to rank):
```c
Block 0 (rank 0): a2 + x(a1 + x(a0)) <-- no shift needed
Block 1 (rank 1): x^3 * (a5 + x(a4 + x(a3)))
Block 2 (rank 2): x^6 * (a8 + x(a7 + x(a6)))
```

which we can then sum together. This way, each process does roughly the same amount of work, and we reduce the overall complexity to O(N/P).

To run the code, we can use the following command (after exporting the PETSC_DIR and PETSC_ARCH variables and making the code):
```bash
mpiexec -n P ./expx -x VALUE -N TERMS
```
where `P` is the number of processes, `VALUE` is the value of x for which we want to compute $e^x$, and `TERMS` is the number of terms N in the taylor series expansion.


Finally, we can also compute the relative error compared to the actual value of $e^x$ using the standard library function `exp(x)` and compare it to our computed value. We can also compute machine epsilon to understand the precision limits of our calculations.


Here are some STDOUTS
```
╭╴ fabg at …/apma4302_fbg2107/pset1 on 󰊢 main () via C v11.4.0-gcc on ☁️   took 2s 
╰─ mpiexec -n 8 ./expx -x -1 -N 10
using blocked Horner: exp(-1.) = 3.67879482162589588334355994447833e-01 (N=10)
using exact: exp(-1.) = 3.67879441171442334024277442949824e-01
relative error: 1.114e-07 = 5.018e+08  [machine eps]

╭╴ fabg at …/apma4302_fbg2107/pset1 on 󰊢 main () via C v11.4.0-gcc on ☁️   took 2s 
╰─ mpiexec -n 8 ./expx -x -1 -N 20
using blocked Horner: exp(-1.) = 3.67879441171442334024277442949824e-01 (N=20)
using exact: exp(-1.) = 3.67879441171442334024277442949824e-01
relative error: 0.000e+00 = 0.000e+00  [machine eps]

╭╴ fabg at …/apma4302_fbg2107/pset1 on 󰊢 main () via C v11.4.0-gcc on ☁️   took 2s 
╰─ mpiexec -n 8 ./expx -x 1 -N 20
using blocked Horner: exp(1.) = 2.71828182845904509079559829842765e+00 (N=20)
using exact: exp(1.) = 2.71828182845904509079559829842765e+00
relative error: 0.000e+00 = 0.000e+00  [machine eps]

╭╴ fabg at …/apma4302_fbg2107/pset1 on 󰊢 main () via C v11.4.0-gcc on ☁️   took 2s 
╰─ mpiexec -n 8 ./expx -x 10 -N 20
using blocked Horner: exp(10.) = 1.73562780882554008599072403740138e+01 (N=20)
using exact: exp(10.) = 2.20264657948067178949713706970215e+04
relative error: 9.992e-01 = 4.500e+15  [machine eps]

╭╴ fabg at …/apma4302_fbg2107/pset1 on 󰊢 main () via C v11.4.0-gcc on ☁️   took 2s 
╰─ mpiexec -n 8 ./expx -x 10 -N 100
using blocked Horner: exp(10.) = 1.53475159531893023086013272404671e+04 (N=100)
using exact: exp(10.) = 2.20264657948067178949713706970215e+04
relative error: 3.032e-01 = 1.366e+15  [machine eps]

╭╴ fabg at …/apma4302_fbg2107/pset1 on 󰊢 main () via C v11.4.0-gcc on ☁️   took 2s 
╰─ mpiexec -n 8 ./expx -x 10 -N 1000
using blocked Horner: exp(10.) = 2.20264657948067288089077919721603e+04 (N=1000)
using exact: exp(10.) = 2.20264657948067178949713706970215e+04
relative error: 4.955e-16 = 2.231e+00  [machine eps]
```

Errors are larger for larger x, as expected since the series converges more slowly for larger x. Increasing N reduces the error however. Negatives are handled by using identity. 


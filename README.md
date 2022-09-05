# Binary Quadratic Programming (BQP)

This script implements the generator of symmetric BQP instances described in Section 5.1 of [1].

### Mathematical Program

Let $i,n$ be natural numbers.

Let $A_0$ denote a $n \times n$ real (possibly indefinite) symmetric matrix.

Let $a_i$ denote a vector of dimension $n$ for all $i \in I_{eq}$.
        
Let $b$ denote a vector of dimension $|I_{eq}|$.
        
Let $x$ represent a $n$ dimensional vector of binary decision variables.

The BQP instances of interest match the following program:

$minimize \quad x^TA_0\ x$

$subject\ to \quad a_i^Tx = b_i, \qquad \forall i \in I_{eq};$

$\qquad\qquad\quad\ x \in \\{0,1\\}^n$.

Where $a^T$ represents the transpose of vector $a$.

### Environment and Usage

The script uses essentially Numpy [2] to code the generator.

`python3 bqp_generator.py <instance_size> <minimum_partition_element_size>`

E.g.: `python3 bqp_generator.py 15 5`.

The instances can be solved using the auxiliary file __bqp.mod__, where the program above is coded in AMPL [3].

### Output

A ".dat" file for each BQP instance created.

### References

[1]: Gustavo Dias, Leo Liberti. Exploiting Symmetries in Mathematical Programming via Orbital Independence. Annals of Operations Research, Vol 298, Number 1, pg 149–182, March 2021. DOI: https://doi.org/10.1007/s10479-019-03145-x

[2]: Numpy. URL: https://numpy.org/

[3]: AMPL. URL: https://ampl.com

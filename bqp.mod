###############################################################################
###
###   BINARY QUADRATIC PROGRAM (BQP)
###
###   The compact form of BQPs is the following: 
###
###   Minimize    x^T*A_0*x
###   Subject to  x^T*A_i*x  = b_i for i in Ieq;
###               x^T*A_i*x <= b_i for i in Iin;
###               x in {0,1}^n.
###
###   Where: - n is a positive integer;
###          - I := {0} U Ieq U Iin is a set of non-negative integers;
###          - A_i is a n x n real matrix for i in I;
###          - b is a |Ieq|+|Iin| dimensional real vector;
###          - x is a n dimensional binary vector of decision variables.
###
###   Let N:={1,...,n}. The BQP model above is written in AMPL[1]. For
###   simplicity, it explicitly exploits the following math properties:
###    (P1) x[i]*x[j] = x[j]*x[i] for all i,j in N;
###    (P2) x[i]*x[i] = x[i] for x[i] in {0,1}, i in N.
###
###   References
###   [1] AMPL: https://ampl.com
###############################################################################

set N;
set I   ordered;
set Ieq within I ordered default {};
set Iin within I ordered default {};

var x   {N} binary;

param A {I,N,N} default 0;
param b {I}     default 0;

minimize obj_fun:         sum{i in N, j in N : i<j} A[first(I),i,j]*x[i]*x[j] + 
                        + sum{i in N, j in N : i>j} A[first(I),i,j]*x[j]*x[i] + 
                        + sum{i in N} A[first(I),i,i]*x[i];

subject to equality {m in Ieq}:  sum{i in N, j in N : i<j} A[m,i,j]*x[i]*x[j] + 
                               + sum{i in N, j in N : i>j} A[m,i,j]*x[j]*x[i] + 
                               + sum{i in N} A[m,i,i]*x[i] =  b[m];

subject to inequality {m in Iin}: sum{i in N, j in N : i<j} A[m,i,j]*x[i]*x[j]+ 
                                + sum{i in N, j in N : i>j} A[m,i,j]*x[j]*x[i]+ 
                                + sum{i in N} A[m,i,i]*x[i] <= b[m];

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

CFLAGS += -pedantic -std=c99
Q4DIR  = q4_stdouts
H5DIR  = bvp_hdf5_files

# Q3 runs
# bvp 
bvp: bvp.o
	-${CLINKER} -o bvp bvp.o ${PETSC_LIB}
	${RM} bvp.o

# Q3 b runs solves the BVP with default options in options_file
run_bvp:
	./bvp -options_file options_file

# parallel version (bjacobi replaces icc for parallel)
run_bvp_parallel:
	mpiexec -np 4 ./bvp -options_file options_file 

# Q3 c runs convergence test 
convergence:
	chmod +x run_3c.sh
	./run_3c.sh

convergence_parallel:
	chmod +x run_3c.sh
	./run_3c.sh 4

# Q4 runs 
# each part writes to q4_stdouts/q4X.out via tee.
# also produce bvp_solution.h5 — moved to H5DIR with
# a descriptive name so they don't overwrite each other.

# (a) richardson + jacobi
run_q4a:
	@mkdir -p $(Q4DIR) $(H5DIR)
	./bvp -options_file options_file \
		-ksp_type richardson -pc_type jacobi \
		-ksp_converged_reason -ksp_monitor \
		2>&1 | tee $(Q4DIR)/q4a.out
	@mv -f bvp_solution.h5 $(H5DIR)/q4a_solution.h5 2>/dev/null || true

# (b) unpreconditioned CG, 1 processor
run_q4b:
	@mkdir -p $(Q4DIR) $(H5DIR)
	./bvp -options_file options_file \
		-ksp_type cg -pc_type none \
		-ksp_converged_reason -ksp_monitor \
		2>&1 | tee $(Q4DIR)/q4b.out
	@mv -f bvp_solution.h5 $(H5DIR)/q4b_solution.h5 2>/dev/null || true

# (c) unpreconditioned CG with c=0
run_q4c:
	@mkdir -p $(Q4DIR) $(H5DIR)
	./bvp -options_file options_file \
		-bvp_c 0 -ksp_type cg -pc_type none \
		-ksp_converged_reason -ksp_monitor \
		2>&1 | tee $(Q4DIR)/q4c.out
	@mv -f bvp_solution.h5 $(H5DIR)/q4c_solution.h5 2>/dev/null || true

# (d) ICC preconditioned CG, 1 processor
run_q4d:
	@mkdir -p $(Q4DIR) $(H5DIR)
	./bvp -options_file options_file \
		-ksp_type cg -pc_type icc \
		-ksp_converged_reason -ksp_monitor \
		2>&1 | tee $(Q4DIR)/q4d.out
	@mv -f bvp_solution.h5 $(H5DIR)/q4d_solution.h5 2>/dev/null || true

# (e) block Jacobi + ICC, 4 processors
run_q4e:
	@mkdir -p $(Q4DIR) $(H5DIR)
	mpiexec -np 4 ./bvp -options_file options_file \
		-ksp_type cg -pc_type bjacobi -sub_pc_type icc \
		-ksp_converged_reason -ksp_monitor \
		2>&1 | tee $(Q4DIR)/q4e.out
	@mv -f bvp_solution.h5 $(H5DIR)/q4e_solution.h5 2>/dev/null || true

# (f) MUMPS direct solver — serial and parallel
run_q4f_serial:
	@mkdir -p $(Q4DIR) $(H5DIR)
	./bvp -options_file options_file \
		-ksp_type preonly -pc_type lu \
		-pc_factor_mat_solver_type mumps \
		-ksp_converged_reason \
		2>&1 | tee $(Q4DIR)/q4f_serial.out
	@mv -f bvp_solution.h5 $(H5DIR)/q4f_serial_solution.h5 2>/dev/null || true

run_q4f_parallel:
	@mkdir -p $(Q4DIR) $(H5DIR)
	mpiexec -np 4 ./bvp -options_file options_file \
		-ksp_type preonly -pc_type lu \
		-pc_factor_mat_solver_type mumps \
		-ksp_converged_reason \
		2>&1 | tee $(Q4DIR)/q4f_parallel.out
	@mv -f bvp_solution.h5 $(H5DIR)/q4f_parallel_solution.h5 2>/dev/null || true

# run all Q4 parts in sequence
run_q4: run_q4a run_q4b run_q4c run_q4d run_q4e \
        run_q4f_serial run_q4f_parallel

# cleanup cleanup, phony means these aren't real files
.PHONY: clean tri bvp \
        run_bvp run_bvp_parallel \
        convergence convergence_parallel \
        run_q4a run_q4b run_q4c run_q4d run_q4e \
        run_q4f_serial run_q4f_parallel run_q4

clean::
	@rm -f tri bvp *.o *~
	@rm -f bvp_solution.h5 bvp_k*.h5
	@rm -f convergence_data.txt convergence.png
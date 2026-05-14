include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

euler1d: euler1d.o
	-${CLINKER} -o euler1d euler1d.o ${PETSC_LIB}
	${RM} euler1d.o

# --- Sod shock tube ---
run_sod:
	mpiexec -np 4 ./euler1d -options_file options_file -problem sod

run_sod_lf:
	mpiexec -np 4 ./euler1d -options_file options_file -problem sod -riemann_type lf -limiter_type none

# convergence study: N = 100 200 400 800
run_sod_convergence:
	@for N in 100 200 400 800; do \
	  echo "=== N=$$N ==="; \
	  mpiexec -np 4 ./euler1d -da_grid_x $$N -problem sod -t_end 0.2 \
	    -riemann_type hllc -limiter_type minmod -output_freq 99999; \
	done

# --- Lax problem ---
run_lax:
	mpiexec -np 4 ./euler1d -options_file options_file -problem lax -t_end 0.13

# --- Vacuum / 123 problem ---
run_vacuum:
	mpiexec -np 4 ./euler1d -options_file options_file -problem vacuum -t_end 0.15

.PHONY: clean run_sod run_sod_lf run_sod_convergence run_lax run_vacuum

clean::
	@rm -f *~ euler1d *.o results_t*.csv

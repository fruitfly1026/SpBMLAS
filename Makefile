CC=g++
#FLAG=-g -Wall
#FLAG=-O3
#LDFLAG=-O3
FLAG=-O3 #-msse4.2 -fp-model source 
LDFLAG=-O3 #-msse4.2 -fp-model source

OBJS=driver.o mmio.o

.cpp.o:
	${CC} -o $@ -c ${FLAG} $<
.s.o:
	as -o $@ $< 

spmv-gcc: ${OBJS}
	${CC}  ${LDFLAG} -o $@ $^

.PHONY:clean
clean: 
	find ./ -name "*.o" -delete
	rm spmv-gcc 


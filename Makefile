CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

all: svm-train svm-predict 

# lib: cvm.o
# 	if [ "$(OS)" = "Darwin" ]; then \
# 		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libcvm.so.$(SHVER)"; \
# 	else \
# 		SHARED_LIB_FLAG="-shared -Wl,-soname,libcvm.so.$(SHVER)"; \
# 	fi; \
# 	$(CXX) $${SHARED_LIB_FLAG} cvm.o -o libcvm.so.$(SHVER)

svm-predict: svm-predict.cpp cvm.o utility.o sgraph.o svm.o   cvm_core.o bvm.o
	$(CXX) $(CFLAGS) svm-predict.cpp  cvm.o utility.o sgraph.o svm.o  cvm_core.o bvm.o  -o svm-predict -lm
svm-train: svm-train.cpp cvm.o utility.o sgraph.o svm.o   cvm_core.o bvm.o
	$(CXX) $(CFLAGS) svm-train.cpp cvm.o utility.o sgraph.o svm.o  cvm_core.o bvm.o -o svm-train -lm

%.o : %.cpp
	$(CC) $(CFLAGS) -c $<
	
clean:
	rm -f *~ *.o  svm-train svm-predict 

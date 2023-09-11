ALL: test
include=-I../include
test:
	hipcc test.cu ${include} --std=c++20 --no-offload-arch=gfx1030 --offload-arch=gfx90a
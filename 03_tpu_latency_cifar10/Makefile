CC := g++
CFLAGS := -std=c++11 -I/usr/include -I/usr/local/include/opencv4
INCLUDEPATHS := -I${HOME}/EAI_TfLite/tensorflow_edgetpu_prebuilt/include \
                -I${HOME}/EAI_TfLite

LDFLAGS := -pthread -ltensorflow-lite -lflatbuffers \
           -l:libedgetpu.so.1.0 -ldl -lrt \
           -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

LDPATH := -L${HOME}/EAI_TfLite/tensorflow_edgetpu_prebuilt/lib \
          -L${HOME}/EAI_TfLite/libs/armv7a

SRCS := tpu_latency_cifar10.cc
OBJS := $(SRCS:.cc=.o)
EXEC := tpu_latency_cifar10

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -o $@ $^ $(LDPATH) $(LDFLAGS)  

%.o: %.cc
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -c $< -o $@   

clean : 
	rm -f $(OBJS) $(EXEC)	

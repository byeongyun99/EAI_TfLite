CC := g++
CFLAGS := -std=c++11 -I/usr/include
INCLUDEPATHS := -I${HOME}/tensorflow -I${HOME}/EAI_TfLite
LDFLAGS := -pthread -ltensorflow-lite -lflatbuffers -l:libedgetpu.so.1.0 -ldl
					 
LDPATH := -L${HOME}/tensorflow/tensorflow/lite/tools/make/gen/bbb_armv7l/lib\
	  -L${HOME}/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build\
		-L${HOME}/EAI_TfLite/libs/armv7a

SRCS := minimal_tpu.cc
OBJS := $(SRCS:.cc=.o)
EXEC := minimal_tpu

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -o $@ $^ $(LDPATH) $(LDFLAGS)  

%.o: %.cc
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -c $< -o $@   

clean : 
	rm -f $(OBJS) $(EXEC)	

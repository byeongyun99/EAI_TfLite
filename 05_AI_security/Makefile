
CXX := g++
ROOT := $(abspath ..)

SRC := adversarial_attack.cc

# CPU TensorFlow Lite 2.4.1
CPU_TF := $(HOME)/tensorflow
CPU_TFLITE := $(CPU_TF)/tensorflow/lite/tools/make/gen/bbb_armv7l/lib/libtensorflow-lite.a
CPU_FLAT := $(CPU_TF)/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a

# Edge TPU compatible TensorFlow Lite
TPU_TF := $(ROOT)/tensorflow_edgetpu_prebuilt
TPU_TFLITE := $(TPU_TF)/lib/libtensorflow-lite.a
TPU_FLAT := $(TPU_TF)/lib/libflatbuffers.a

# Edge TPU runtime
EDGETPU := $(ROOT)/libs/armv7a/libedgetpu.so.1.0

CXXFLAGS := -std=c++11 -O2 \
            -I$(ROOT) \
            -I/usr/local/include/opencv4

CPU_INCLUDES := -I$(CPU_TF) \
                -I$(CPU_TF)/tensorflow/lite/tools/make/downloads/flatbuffers/include

TPU_INCLUDES := -I$(TPU_TF)/include

LIBS := -pthread -ldl -lrt -lm \
        -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

RPATH := -Wl,-rpath,'$$ORIGIN/../libs/armv7a'

.PHONY: all clean cpu tpu

all: cpu tpu

cpu: adversarial_attack_cpu

tpu: adversarial_attack_tpu

adversarial_attack_cpu: $(SRC) $(CPU_TFLITE) $(CPU_FLAT)
	$(CXX) $(CXXFLAGS) $(CPU_INCLUDES) $(SRC) -o $@ \
	$(CPU_TFLITE) $(CPU_FLAT) $(EDGETPU) $(LIBS) $(RPATH)

adversarial_attack_tpu: $(SRC) $(TPU_TFLITE) $(TPU_FLAT)
	$(CXX) $(CXXFLAGS) $(TPU_INCLUDES) $(SRC) -o $@ \
	$(TPU_TFLITE) $(TPU_FLAT) $(EDGETPU) $(LIBS) $(RPATH)

clean:
	rm -f adversarial_attack_cpu adversarial_attack_tpu adversarial_attack.o adversarial_attack


$(warning mnist demo Makefile has been included)

MNIST_DEMO_SRCS := \
tensorflow/lite/micro/examples/micro_detection/main.cc \
tensorflow/lite/micro/examples/micro_detection/model/mnist_model.cc

MNIST_DEMO_HDRS := \
tensorflow/lite/micro/examples/micro_detection/model/mnist_model.h \
tensorflow/lite/micro/examples/micro_detection/model/mnist_test_data.h

MNIST_DEMO_TEST_SRCS := \
tensorflow/lite/micro/examples/micro_detection/mnist_demo_test.cc \
tensorflow/lite/micro/examples/micro_detection/model/mnist_model.cc

MNIST_DEMO_TEST_HDRS := \
tensorflow/lite/micro/examples/micro_detection/model/mnist_model.h \
tensorflow/lite/micro/examples/micro_detection/model/mnist_test_data.h

# Find any platform-specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/micro_detection/*/Makefile.inc)

# Tests the character classifier model with a seven.
$(eval $(call microlite_test,micro_detection_test,\
$(MNIST_DEMO_TEST_SRCS),$(MNIST_DEMO_TEST_HDRS)))

# Builds an MNIST character classifier demo binary.
$(eval $(call microlite_test,micro_detection,\
$(MNIST_DEMO_SRCS),$(MNIST_DEMO_HDRS)))

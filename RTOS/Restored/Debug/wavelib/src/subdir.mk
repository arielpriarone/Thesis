################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (11.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../wavelib/src/conv.c \
../wavelib/src/cwt.c \
../wavelib/src/cwtmath.c \
../wavelib/src/hsfft.c \
../wavelib/src/real.c \
../wavelib/src/wavefilt.c \
../wavelib/src/wavefunc.c \
../wavelib/src/wavelib.c \
../wavelib/src/wtmath.c 

OBJS += \
./wavelib/src/conv.o \
./wavelib/src/cwt.o \
./wavelib/src/cwtmath.o \
./wavelib/src/hsfft.o \
./wavelib/src/real.o \
./wavelib/src/wavefilt.o \
./wavelib/src/wavefunc.o \
./wavelib/src/wavelib.o \
./wavelib/src/wtmath.o 

C_DEPS += \
./wavelib/src/conv.d \
./wavelib/src/cwt.d \
./wavelib/src/cwtmath.d \
./wavelib/src/hsfft.d \
./wavelib/src/real.d \
./wavelib/src/wavefilt.d \
./wavelib/src/wavefunc.d \
./wavelib/src/wavelib.d \
./wavelib/src/wtmath.d 


# Each subdirectory must supply rules for building sources it contributes
wavelib/src/%.o wavelib/src/%.su wavelib/src/%.cyclo: ../wavelib/src/%.c wavelib/src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F767xx -c -I../Core/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../Drivers/CMSIS/Include -I"C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/Restored/wavelib/header" -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-wavelib-2f-src

clean-wavelib-2f-src:
	-$(RM) ./wavelib/src/conv.cyclo ./wavelib/src/conv.d ./wavelib/src/conv.o ./wavelib/src/conv.su ./wavelib/src/cwt.cyclo ./wavelib/src/cwt.d ./wavelib/src/cwt.o ./wavelib/src/cwt.su ./wavelib/src/cwtmath.cyclo ./wavelib/src/cwtmath.d ./wavelib/src/cwtmath.o ./wavelib/src/cwtmath.su ./wavelib/src/hsfft.cyclo ./wavelib/src/hsfft.d ./wavelib/src/hsfft.o ./wavelib/src/hsfft.su ./wavelib/src/real.cyclo ./wavelib/src/real.d ./wavelib/src/real.o ./wavelib/src/real.su ./wavelib/src/wavefilt.cyclo ./wavelib/src/wavefilt.d ./wavelib/src/wavefilt.o ./wavelib/src/wavefilt.su ./wavelib/src/wavefunc.cyclo ./wavelib/src/wavefunc.d ./wavelib/src/wavefunc.o ./wavelib/src/wavefunc.su ./wavelib/src/wavelib.cyclo ./wavelib/src/wavelib.d ./wavelib/src/wavelib.o ./wavelib/src/wavelib.su ./wavelib/src/wtmath.cyclo ./wavelib/src/wtmath.d ./wavelib/src/wtmath.o ./wavelib/src/wtmath.su

.PHONY: clean-wavelib-2f-src


################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (11.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
S_SRCS += \
C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/ADC_SingleConversion_TriggerSW_DMA/SW4STM32/startup_stm32f767xx.s 

OBJS += \
./Example/SW4STM32/startup_stm32f767xx.o 

S_DEPS += \
./Example/SW4STM32/startup_stm32f767xx.d 


# Each subdirectory must supply rules for building sources it contributes
Example/SW4STM32/startup_stm32f767xx.o: C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/ADC_SingleConversion_TriggerSW_DMA/SW4STM32/startup_stm32f767xx.s Example/SW4STM32/subdir.mk
	arm-none-eabi-gcc -mcpu=cortex-m7 -g3 -c -x assembler-with-cpp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@" "$<"

clean: clean-Example-2f-SW4STM32

clean-Example-2f-SW4STM32:
	-$(RM) ./Example/SW4STM32/startup_stm32f767xx.d ./Example/SW4STM32/startup_stm32f767xx.o

.PHONY: clean-Example-2f-SW4STM32


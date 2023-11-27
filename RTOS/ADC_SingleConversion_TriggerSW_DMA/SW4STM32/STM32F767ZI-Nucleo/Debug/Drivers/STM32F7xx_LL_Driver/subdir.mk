################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (11.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/ADC_SingleConversion_TriggerSW_DMA/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_ll_utils.c 

OBJS += \
./Drivers/STM32F7xx_LL_Driver/stm32f7xx_ll_utils.o 

C_DEPS += \
./Drivers/STM32F7xx_LL_Driver/stm32f7xx_ll_utils.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/STM32F7xx_LL_Driver/stm32f7xx_ll_utils.o: C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/ADC_SingleConversion_TriggerSW_DMA/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_ll_utils.c Drivers/STM32F7xx_LL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DSTM32F767xx -DUSE_STM32F7XX_NUCLEO -DUSE_FULL_LL_DRIVER -DHSE_VALUE=8000000U -c -I../../../Inc -I../../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../../Drivers/CMSIS/Include -Os -ffunction-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-STM32F7xx_LL_Driver

clean-Drivers-2f-STM32F7xx_LL_Driver:
	-$(RM) ./Drivers/STM32F7xx_LL_Driver/stm32f7xx_ll_utils.cyclo ./Drivers/STM32F7xx_LL_Driver/stm32f7xx_ll_utils.d ./Drivers/STM32F7xx_LL_Driver/stm32f7xx_ll_utils.o ./Drivers/STM32F7xx_LL_Driver/stm32f7xx_ll_utils.su

.PHONY: clean-Drivers-2f-STM32F7xx_LL_Driver


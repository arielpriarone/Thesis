################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (11.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_Printf/Drivers/BSP/STM32F7xx_Nucleo_144/stm32f7xx_nucleo_144.c 

OBJS += \
./Drivers/BSP/STM32F767ZI-Nucleo/stm32f7xx_nucleo_144.o 

C_DEPS += \
./Drivers/BSP/STM32F767ZI-Nucleo/stm32f7xx_nucleo_144.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/BSP/STM32F767ZI-Nucleo/stm32f7xx_nucleo_144.o: C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_Printf/Drivers/BSP/STM32F7xx_Nucleo_144/stm32f7xx_nucleo_144.c Drivers/BSP/STM32F767ZI-Nucleo/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DUSE_HAL_DRIVER -DSTM32F767xx -DUSE_STM32F7XX_NUCLEO_144 -c -I../../../Inc -I../../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../../Drivers/CMSIS/Include -I../../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../../Drivers/BSP/STM32F7xx_Nucleo_144 -I../../../Drivers/BSP/Components/Common -I../../../Utilities/Log -I../../../Utilities/Fonts -I../../../Utilities/CPU -Os -ffunction-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-BSP-2f-STM32F767ZI-2d-Nucleo

clean-Drivers-2f-BSP-2f-STM32F767ZI-2d-Nucleo:
	-$(RM) ./Drivers/BSP/STM32F767ZI-Nucleo/stm32f7xx_nucleo_144.cyclo ./Drivers/BSP/STM32F767ZI-Nucleo/stm32f7xx_nucleo_144.d ./Drivers/BSP/STM32F767ZI-Nucleo/stm32f7xx_nucleo_144.o ./Drivers/BSP/STM32F767ZI-Nucleo/stm32f7xx_nucleo_144.su

.PHONY: clean-Drivers-2f-BSP-2f-STM32F767ZI-2d-Nucleo


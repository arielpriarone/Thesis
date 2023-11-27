################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (11.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal.c \
C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_cortex.c \
C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_dma.c \
C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_gpio.c \
C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_pwr.c \
C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_pwr_ex.c \
C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_rcc.c \
C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_rcc_ex.c \
C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_uart.c \
C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_usart.c 

OBJS += \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal.o \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_cortex.o \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_dma.o \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_gpio.o \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_pwr.o \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_pwr_ex.o \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_rcc.o \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_rcc_ex.o \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_uart.o \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_usart.o 

C_DEPS += \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal.d \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_cortex.d \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_dma.d \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_gpio.d \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_pwr.d \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_pwr_ex.d \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_rcc.d \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_rcc_ex.d \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_uart.d \
./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_usart.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal.o: C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal.c Drivers/STM32F7xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DSTM32F767xx -DUSE_HAL_DRIVER -DUSE_FULL_LL_DRIVER -DUSE_STM32F7XX_NUCLEO_144 -c -I../../../Inc -I../../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../../Drivers/BSP/STM32F7xx_Nucleo_144 -I../../../Drivers/BSP/Components/Common -I../../../Drivers/CMSIS/Include -Os -ffunction-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_cortex.o: C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_cortex.c Drivers/STM32F7xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DSTM32F767xx -DUSE_HAL_DRIVER -DUSE_FULL_LL_DRIVER -DUSE_STM32F7XX_NUCLEO_144 -c -I../../../Inc -I../../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../../Drivers/BSP/STM32F7xx_Nucleo_144 -I../../../Drivers/BSP/Components/Common -I../../../Drivers/CMSIS/Include -Os -ffunction-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_dma.o: C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_dma.c Drivers/STM32F7xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DSTM32F767xx -DUSE_HAL_DRIVER -DUSE_FULL_LL_DRIVER -DUSE_STM32F7XX_NUCLEO_144 -c -I../../../Inc -I../../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../../Drivers/BSP/STM32F7xx_Nucleo_144 -I../../../Drivers/BSP/Components/Common -I../../../Drivers/CMSIS/Include -Os -ffunction-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_gpio.o: C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_gpio.c Drivers/STM32F7xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DSTM32F767xx -DUSE_HAL_DRIVER -DUSE_FULL_LL_DRIVER -DUSE_STM32F7XX_NUCLEO_144 -c -I../../../Inc -I../../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../../Drivers/BSP/STM32F7xx_Nucleo_144 -I../../../Drivers/BSP/Components/Common -I../../../Drivers/CMSIS/Include -Os -ffunction-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_pwr.o: C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_pwr.c Drivers/STM32F7xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DSTM32F767xx -DUSE_HAL_DRIVER -DUSE_FULL_LL_DRIVER -DUSE_STM32F7XX_NUCLEO_144 -c -I../../../Inc -I../../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../../Drivers/BSP/STM32F7xx_Nucleo_144 -I../../../Drivers/BSP/Components/Common -I../../../Drivers/CMSIS/Include -Os -ffunction-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_pwr_ex.o: C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_pwr_ex.c Drivers/STM32F7xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DSTM32F767xx -DUSE_HAL_DRIVER -DUSE_FULL_LL_DRIVER -DUSE_STM32F7XX_NUCLEO_144 -c -I../../../Inc -I../../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../../Drivers/BSP/STM32F7xx_Nucleo_144 -I../../../Drivers/BSP/Components/Common -I../../../Drivers/CMSIS/Include -Os -ffunction-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_rcc.o: C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_rcc.c Drivers/STM32F7xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DSTM32F767xx -DUSE_HAL_DRIVER -DUSE_FULL_LL_DRIVER -DUSE_STM32F7XX_NUCLEO_144 -c -I../../../Inc -I../../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../../Drivers/BSP/STM32F7xx_Nucleo_144 -I../../../Drivers/BSP/Components/Common -I../../../Drivers/CMSIS/Include -Os -ffunction-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_rcc_ex.o: C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_rcc_ex.c Drivers/STM32F7xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DSTM32F767xx -DUSE_HAL_DRIVER -DUSE_FULL_LL_DRIVER -DUSE_STM32F7XX_NUCLEO_144 -c -I../../../Inc -I../../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../../Drivers/BSP/STM32F7xx_Nucleo_144 -I../../../Drivers/BSP/Components/Common -I../../../Drivers/CMSIS/Include -Os -ffunction-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_uart.o: C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_uart.c Drivers/STM32F7xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DSTM32F767xx -DUSE_HAL_DRIVER -DUSE_FULL_LL_DRIVER -DUSE_STM32F7XX_NUCLEO_144 -c -I../../../Inc -I../../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../../Drivers/BSP/STM32F7xx_Nucleo_144 -I../../../Drivers/BSP/Components/Common -I../../../Drivers/CMSIS/Include -Os -ffunction-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_usart.o: C:/Users/ariel/Documents/Courses/Tesi/Code/RTOS/UART_HyperTerminal_IT/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_usart.c Drivers/STM32F7xx_HAL_Driver/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DSTM32F767xx -DUSE_HAL_DRIVER -DUSE_FULL_LL_DRIVER -DUSE_STM32F7XX_NUCLEO_144 -c -I../../../Inc -I../../../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../../../Drivers/STM32F7xx_HAL_Driver/Inc -I../../../Drivers/BSP/STM32F7xx_Nucleo_144 -I../../../Drivers/BSP/Components/Common -I../../../Drivers/CMSIS/Include -Os -ffunction-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-STM32F7xx_HAL_Driver

clean-Drivers-2f-STM32F7xx_HAL_Driver:
	-$(RM) ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal.cyclo ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal.d ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal.o ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal.su ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_cortex.cyclo ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_cortex.d ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_cortex.o ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_cortex.su ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_dma.cyclo ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_dma.d ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_dma.o ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_dma.su ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_gpio.cyclo ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_gpio.d ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_gpio.o ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_gpio.su ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_pwr.cyclo ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_pwr.d ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_pwr.o ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_pwr.su ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_pwr_ex.cyclo ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_pwr_ex.d ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_pwr_ex.o ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_pwr_ex.su ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_rcc.cyclo ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_rcc.d ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_rcc.o ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_rcc.su ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_rcc_ex.cyclo ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_rcc_ex.d ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_rcc_ex.o ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_rcc_ex.su ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_uart.cyclo ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_uart.d ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_uart.o ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_uart.su ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_usart.cyclo ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_usart.d ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_usart.o ./Drivers/STM32F7xx_HAL_Driver/stm32f7xx_hal_usart.su

.PHONY: clean-Drivers-2f-STM32F7xx_HAL_Driver


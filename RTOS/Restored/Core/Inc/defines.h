#define FALSE 0				// FALSE value definition
#define TRUE 1				// TRUE value definition
#define TREE_DEPTH 6 		// DEPTH OF THE PACKET TRASFORM: NODES BOTTOM LEVEL = 2^6 = 64
#define TD_FEAT 3			// TIME-DOMAIN FEATURES (MEAN, POWER, STD)
#define ADC_BUF_LEN	6000	// SNAPSHOT TIME-DOMAIN LENGTH (5000 SANPLES AT 5KHZ -> 5S RECORDING)
#define LED_RED GPIO_PIN_14 // RED LED ADDRESS AT GPIO B
#define LED_BLU GPIO_PIN_7 	// RED LED ADDRESS AT GPIO B
#define LED_GRE GPIO_PIN_0 	// RED LED ADDRESS AT GPIO B
#define VERBOSE FALSE		// SET THE VERBOSITY MODE
#define OPMODE TRAIN		// SET THE OPERATION MODE: TRAIN or EVALUATE
#define Use7Features TRUE	// use only first 7 features for evaluation

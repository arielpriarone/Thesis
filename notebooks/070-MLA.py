import src
# %% Manchine Learning Agent
# The MLA is the core of the framework. It is responsible for the training and evaluation of the ML models.
# It is also responsible for the data preparation and the data flow.
# The MLA is a class that inherits from the DB_Manager class.
# The MLA is initialized with a configuration file and a type (novelty/fault).
# The MLA has a mode (evaluate/train/retrain) that cannot be changed at runtime. - for now

HealtyAgent = src.models.MLA(configStr='../config.yaml', type='novelty')
HealtyAgent._standardize_features(HealtyAgent.col_healthy_train)

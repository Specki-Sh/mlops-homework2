class Config:
    RANDOM_STATE = 42
    TRAIN_DATA_PATH = "./data/train.csv"
    TEST_DATA_PATH = "./data/test.csv"
    CAT_FEATURES = ['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2',
                    'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC',
                    'ROLE_FAMILY', 'ROLE_CODE']
    VAL_SIZE = 0.2
    N_TRIALS = 20
    EXPERIMENT_NAME = "employee_access_prediction"
    ARTIFACTS_PATH = "./artifacts"
    
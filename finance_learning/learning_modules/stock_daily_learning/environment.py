class DataEnvironment:
    input_size = 7
    sequence_len = 60
    evaluate_len = 5
    validation_rate = 0.20
    test_rate = 0.20
    rise_rate = 0.3
    fall_rate = -0.04
    output_size = 3

class ModelEnvironment:
    epoch = 100
    batch_size = 10
    lstm_size = 14
    lstm_depth = 5
    learning_rate = 0.002

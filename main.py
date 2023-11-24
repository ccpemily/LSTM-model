import model

if __name__ == "__main__":
    module = model.lstm_module(model.DEFAULT_MODULE_PARAMS)
    model.train(module, model.DEFAULT_DATASET, model.DEFAULT_MODULE_PARAMS)
    result = model.predict(module, model.DEFAULT_DATASET, model.DEFAULT_MODULE_PARAMS)
    print(result)
    pass
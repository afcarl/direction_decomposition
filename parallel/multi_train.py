import pipeline

def train(model, inputs, targets, lr, batch_size, iters):
    model = pipeline.Trainer(model, lr, batch_size)
    model.train(inputs, targets, iters=iters)
    return model

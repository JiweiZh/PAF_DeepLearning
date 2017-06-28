import network as nw
import mnist_loader as ml


training_data, validation_data, test_data = ml.load_data_wrapper()
net = nw.Network([nw.nb_px**2,30,10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

training_data2, validation_data2, test_data2 = ml.load_data_wrapper()

print(net.evaluate(nw.resizeall(list(test_data2))))
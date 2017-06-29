import network as nw
import mnist_loader as ml
import modifications as md

epochs=0
q=''


print("==========================================================================")
print("==========================================================================")
print("==================WELCOME TO THIS NEW SIMULATION SESSION==================")
print("==========================================================================")
print("==========================================================================")
while not 1<=epochs<=100:
    print("\n++++++++++++++ Let\'s set the learning time of our program ! ++++++++++++++\n")
    try:
        epochs=int(input("How many epochs of learning do you want the program to run ? (type a number between 1 and 100)\n"))
    except:
        print('Invalid Number')

while not (q=='yes' or q=='no'):
    print("\n==========================================================================")
    q=input("Do you want to reduce the images resolution ? (type \'yes\' or \'no\')\n")
if q=='yes':
    print("\n==========================================================================")
    try:
        nw.nb_px=int(input("The current images are 28*28 pixels, what resolution do you want ? (type a number between 1 and 27)\n"))
    except ValueError:
        print('Invalid Number')


print("\n==========================================================================")
print("==========================================================================")
print("====================== The computer is learning... =======================")
print("==========================================================================")
print("==========================================================================\n")
training_data, validation_data, test_data = ml.load_data_wrapper()
net = nw.Network([nw.nb_px**2,30,10])
net.SGD(training_data, epochs, 10, 3.0, test_data=test_data)



while True:
    _, _, test_data2 = ml.load_data_wrapper()
    test=nw.resizeall(list(test_data2),nw.nb_px)
    q=''
    number_of_errors=-1
    bits_per_pixel=-1
    print("\n==========================================================================")
    print("==========================================================================\n")

    while not (q=='no' or q=='yes'):
        q=input("Do you want to continue ? (Type \'yes\' to make a new simulation, \'no\' otherwise)\n")
    if q=='no':
        break
    else:
        print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+++++++++++++ Let's set the number of errors in the images ! +++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        while not (0<=number_of_errors<=nw.nb_px**2):
            try:
                number_of_errors=int(input("Which average number of random pixels do you want ? (type a number between 0 and {})\n".format(nw.nb_px**2)))
            except ValueError:
                print('Invalid Number')
        print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+++++++++ Let's set the number of bits per pixel in the images ! +++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        while not (0<=bits_per_pixel<=8):
            try:
                bits_per_pixel=int(input("How many bits do you want the pixels to be represented on ? (type a number between 1 and 8)\n"))
            except ValueError:
                print('Invalid Number')
        #test=md.new_resolution(test,bits_per_pixel)
        for k in range(len(test)):
            try:
                im=test[k][0]
                im=md.error2(im,number_of_errors,nw.nb_px**2)
                test[k]=(md.new_resolution(im,bits_per_pixel),test[k][1])
            except IndexError:
                print('ie')
        print("\n\n==========================================================================")
        print("==========================================================================")
        print("==========================================================================")
        print("=============================== Parameters ===============================")
        print("==========================================================================\n")
        print("number of learning epochs       :",epochs)
        print("resolution                      : {0}*{0}".format(nw.nb_px))
        print("average number of random pixels :",number_of_errors)
        print("number of bits per pixel        :",bits_per_pixel)
        print("\n==========================================================================")
        print("================================ RESULTS =================================")
        print("==========================================================================\n")
        print('test accuracy                   : {}'.format(net.evaluate(test)/len(test)))
from sklearn.datasets import load_digits

if __name__ == '__main__':
    mnist_data = load_digits()
    X = mnist_data['data']
    y = mnist_data['target']
    print(len(X))
    print(type(X))
    print(X.shape)
    print(y.shape)
    output_file = open("../mnist_data.txt", 'w')
    desc = "File description: samples count: " + str(X.shape[0]) + ". Each of the following lines contains one sample."
    desc += " Each sample is represented by 64 int numbers separated by ''."
    desc += " At the end of each line is int representing sample class. There are 10 classes represented by 0, ..., 9."

    output_file.write(desc + '\n')
    for i in range(len(X)):
        for j in range(X.shape[1]):
            output_file.write(str(int(X[i, j])) + ' ')
        output_file.write(str(y[i]) + '\n')

    output_file.close()

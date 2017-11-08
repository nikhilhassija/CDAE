import numpy
numpy.random.seed(0)

import CDAE2
# import movie_lens
import metrics

row_num = 6040
col_num = 3952


# data
f1 = open("u1.base","r")
f2 = open("u1.test","r")


train_x = numpy.zeros((row_num,col_num), dtype=int)

for line in f1:
    split_line = line.split("::")
    train_x[int(split_line[0])-1][int(split_line[1])-1]=int(split_line[2])

test_x = numpy.zeros((row_num,col_num), dtype=int)

for line in f2:
    split_line = line.split("::")
    test_x[int(split_line[0])-1][int(split_line[1])-1]=int(split_line[2])

f1.close()
f2.close()

train_x_users = numpy.asarray([i for i in range(row_num)])
test_x_users = numpy.asarray([i for i in range(row_num)])

# user sparse matrix
user_X = []
user_file = open('users.dat','r')
age = [1,18,25,35,45,50,56]
for line in user_file:
    temp = line.strip().split("::")
    if(temp[1]=="M"):
        l_sex = [1,0]
    else:
        l_sex = [0,1]
    l_age = [0]*8
    for i in range(len(age)):
        if(int(temp[2])==age[i]):
            l_age[i]=1
    l_occupation = [0]*21
    l_occupation[int(temp[3])]=1
    temp_arr = l_age + l_sex + l_occupation
    user_X.append(temp_arr)
user_file.close()

# model
model = CDAE.create(I=train_x.shape[1], U=user_X.shape[1], K=50,
                    hidden_activation='relu', output_activation='sigmoid', q=0.50, l=0.01)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

# train
history = model.fit(x=[train_x, user_X], y=train_x,
                    batch_size=128, nb_epoch=1000, verbose=1,
                    validation_data=[[test_x, user_X], test_x])

model.save_weights("model2.h5")
# model.load_weights("model2.h5")

# predict
pred = model.predict(x=[train_x, train_x_users])
pred = pred * (train_x == 0) # remove watched items from predictions
pred = numpy.argsort(pred)

for n in range(1,11):
    sr = metrics.success_rate(pred[:, -n:], test_x)
    print("Success Rate at {:d}: {:f}".format(n, sr))
    p = metrics.precision(pred[:, -n:], test_x)
    print("Precision at {:d}: {:f}".format(n, p))
    r = metrics.recall(pred[:, -n:], test_x)
    print("Recall at {:d}: {:f}".format(n, r))
    Map = metrics.Map(pred[:, -n:], test_x)
    print("Map at {:d}: {:f}".format(n, Map))
    print

import numpy
numpy.random.seed(0)

import CDAE
# import movie_lens
import metrics
import sys

# row_num = 943
# col_num = 1682


row_num = 6040
col_num = 3952


foldx = sys.argv[1]
delim = ","


f1 = open("u{}.base".format(foldx),"r")
f2 = open("u{}.test".format(foldx),"r")

train_x = numpy.zeros((row_num,col_num), dtype=int)

for line in f1:
	split_line = line.split(delim)
	if int(split_line[2]) > 3:
		train_x[int(split_line[0])-1][int(split_line[1])-1] = 1

test_x = numpy.zeros((row_num,col_num), dtype=int)

for line in f2:
	split_line = line.split(delim)
	if int(split_line[2]) > 3:
		test_x[int(split_line[0])-1][int(split_line[1])-1] = 1

f1.close()
f2.close()

# data
train_x_users = numpy.asarray([i for i in range(row_num)])
test_x_users = numpy.asarray([i for i in range(row_num)])

# model
model = CDAE.create(I=train_x.shape[1], U=row_num, K=50,
					hidden_activation='relu', output_activation='sigmoid', q=0.50, l=0.01)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

# train
history = model.fit(x=[train_x, train_x_users], y=train_x,
					batch_size=128, nb_epoch=1000, verbose=1,
					validation_data=[[train_x, train_x_users], train_x])

model.save_weights("model.h5")
# model.load_weights("model.h5")

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


'''
Success Rate at 1: 27.783669
Success Rate at 2: 39.236479
Success Rate at 3: 45.281018
Success Rate at 4: 49.310710
Success Rate at 5: 51.219512
Success Rate at 6: 53.234358
Success Rate at 7: 54.188759
Success Rate at 8: 55.673383
Success Rate at 9: 56.733828
Success Rate at 10: 57.688229
'''
import numpy

def success_rate(pred, true):
	cnt = 0
	for i in range(pred.shape[0]):
		t = numpy.where(true[i] == 1) # true set
		ary = numpy.intersect1d(pred[i], t)
		if ary.size > 0:
			cnt += 1
	return cnt * 100 / pred.shape[0]

def precision(pred, true):
	p = 0
	for i in range(pred.shape[0]):
		t = numpy.where(true[i] == 1) # true set
		ary = numpy.intersect1d(pred[i], t)
		p += ary.size
	p /= pred.shape[1]
	return p * 100 / pred.shape[0]

def recall(pred, true):
	p = 0

	for i in range(pred.shape[0]):
		true_set = numpy.where(true[i] == 1) # true set
		
		ary = numpy.intersect1d(pred[i], true_set)

		p += ary.size / len(true_set)

	return p * 100 / true.shape[0]

# n   = ored,shape[1]
# MAP = sigma(AP_user)
# AP  = (sigma_K^N (precision @ k * rel(k))) / (min(N, true_set.shape[0])

def Map(pred, true):
	MAP = 0

	for user in range(pred.shape[0]):
		true_set = numpy.where(true[user] == 1)

		AP = 0

		for k in range(1, pred.shape[1] + 1):
			first_k = pred[user, -k:]

			prec = numpy.intersect1d(first_k, true_set)
			prck = prec.size / k

			AP += prck * numpy.any(prec == first_k[0]).astype(float)

		AP = AP / (min(pred.shape[1], len(true_set)))

		MAP += AP

	return MAP / pred.shape[0]
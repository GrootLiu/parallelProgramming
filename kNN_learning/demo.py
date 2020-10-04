import csv
import random

# 读取数据
with open('Prostate_Cancer.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = [row for row in reader]

# 分组
random.shuffle(data)
n = len(data) // 3
test_set = data[0:n]
train_set = data[n:]


# kNN
# 距离
def distance(d1, d2):
    res = 0
    for key in ('radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'symmetry', 'fractal_dimension'):
        res += (float(d1[key]) - float(d2[key])) ** 2
    return res ** 0.5


K = 4


def knn(d):
    # 1. 距离
    res = [
        {'result': train['diagnosis_result'], 'distance': distance(d, train)}
        for train in train_set
    ]

    # 2. 排序——升序
    res = sorted(res, key=lambda item: item['distance'])

    # 3. 取前K个
    res2 = res[0:K]

    # 4. 加权平均
    result = {'B': 0, 'M': 0}
    # 总距离
    sum = 0
    for r in res2:
        sum += r['distance']
    for r in res2:
        result[r['result']] += (1 - r['distance'] / sum)

    if result['B'] > result['M']:
        return 'B'
    else:
        return 'M'


# 验证
correct = 0
for test in test_set:
    result = test['diagnosis_result']
    result2 = knn(test)

    if result == result2:
        correct += 1

print(correct)
print("{:.2f}%".format(100*correct/len(test_set)))

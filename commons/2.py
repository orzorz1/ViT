import numpy as np

data = np.array([13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70])

# 箱光滑
bin_size = 3
binned_data = []
for i in range(0, len(data), bin_size):
    bin_mean = np.mean(data[i:i+bin_size])
    binned_data.extend([bin_mean] * min(bin_size, len(data) - i))
# 计算IQR
Q1 = np.percentile(binned_data, 25)
Q3 = np.percentile(binned_data, 75)
IQR = Q3 - Q1

# 计算Z-分数
mean = np.mean(binned_data)
std_dev = np.std(binned_data)
z_scores = [(x - mean) / std_dev for x in binned_data]

# 获取离群点
outliers_iqr = [x for x in binned_data if x < (Q1 - 1.5 * IQR) or x > (Q3 + 1.5 * IQR)]
outliers_z_score = [binned_data[i] for i in range(len(binned_data)) if z_scores[i] > 3 or z_scores[i] < -3]

print(outliers_iqr)
print(outliers_z_score)
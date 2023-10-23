import torch


def restore_shape(input2d, batch_size, deep, width, height):
    # 先将 input2d 分割为 batch_size 个组
    batches = torch.chunk(input2d, batch_size)

    # 初始化一个列表来存储每个 batch 的结果
    restored_batches = []

    for batch in batches:
        # 初始化一个张量来存储当前 batch 的所有层
        restored_batch = torch.zeros((deep, width, height), dtype=batch.dtype, device=batch.device)

        # 循环遍历当前 batch 的所有切片，并将它们加到相应的层中
        for i, slice in enumerate(batch):
            if i == 0:
                restored_batch[i:i + 2, :, :] += slice[0:2, :, :]
            elif i == deep - 1:
                restored_batch[i - 1:i + 1, :, :] += slice[1:3, :, :]
            else:
                restored_batch[i - 1:i + 2, :, :] += slice

        restored_batches.append(restored_batch)

    # 将所有恢复的 batch 堆叠起来
    output = torch.stack(restored_batches)

    return output
# 获取input2d的维度信息
input2d = torch.randn([16, 3, 224, 224])
num_slide_batch, _, width, height = input2d.shape

# 假设你已经有了batch_size和deep的值，然后调用函数恢复原始形状
batch_size = 1
deep = 8
output = restore_shape(input2d, batch_size, deep, width, height)
print(output.shape)
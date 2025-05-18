import tensorflow as tf

# 替换为你的事件文件路径
event_file = 'data/onehot_example_dataset/val/frames/events.out.tfevents.1743735607.txt'

for event in tf.compat.v1.train.summary_iterator(event_file):
    for value in event.summary.value:
        if value.tag == 'train_loss_step':
            print(f"Step: {event.step}, Train Loss: {value.simple_value}")
        elif value.tag == 'train_loss_epoch':
            print(f"Epoch: {event.step}, Train Loss: {value.simple_value}")
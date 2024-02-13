def convert_onehot(num_classes, label):
    onehot_label = [0 for _ in range(num_classes)]
    if isinstance(label,list):
        for l in label:
            onehot_label[int(l)]=1
    else:
        onehot_label[int(label)] = 1
    return onehot_label


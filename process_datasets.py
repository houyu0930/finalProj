from tqdm import tqdm

train_path = '/home/houyu/learning/FinalProject/datasets/E-c/E-c-En-train.txt'
dev_path = '/home/houyu/learning/FinalProject/datasets/E-c/E-c-En-dev.txt'
test_path = '/home/houyu/learning/FinalProject/datasets/E-c/E-c-En-test-gold.txt'

new_train = '/home/houyu/learning/FinalProject/datasets/new/E-c-En-train.txt'
new_dev = '/home/houyu/learning/FinalProject/datasets/new/E-c-En-dev.txt'
new_test = '/home/houyu/learning/FinalProject/datasets/new/E-c-En-test-gold.txt'


def gen_newdata(ori_path, new_path):

    with open(ori_path, 'r') as fin:
        with open(new_path, 'w') as fout:
            flag = 0
            for line in tqdm(fin.readlines()):
                items = line[:len(line)-1].split('\t')
                # print(items)
                if flag == 0:
                    items.append('neutral')
                    new_line = '\t'.join(items)
                    new_line = new_line + '\n'
                    fout.write(new_line)
                    flag = 1
                else:
                    cnt = 0
                    for item in items[2:]:
                        cnt += int(item)
                    if cnt == 0:
                        items.append('1')
                    else:
                        items.append('0')
                    new_line = '\t'.join(items)
                    new_line = new_line + '\n'
                    fout.write(new_line)
                    # print(line)
            # break


def main():
    gen_newdata(dev_path, new_dev)
    gen_newdata(train_path, new_train)
    gen_newdata(test_path, new_test)

if __name__ == '__main__':
    main()
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--numbers", type=str, default=None)
    opts = parser.parse_args()
    
    def str2list(s):
        if type(s) is list:
            return s
        items = ''.join(s.strip()).split(',')
        ret = []
        for i in items:
            ret.append(float(i))
        return ret
    
    numbers = str2list(opts.numbers)
    sum = 0.
    for n in numbers:
        sum += n
    print('{:.4f}'.format(sum/len(numbers))) 